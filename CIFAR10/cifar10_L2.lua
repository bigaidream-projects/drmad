--[[
Multiple meta-iterations for DrMAD on CIFAR10
]]

require 'torch'
require 'sys'
require 'image'

local root = '../'

local grad = require 'autograd'
local utils = require(root .. 'models/utils.lua')
local optim = require 'optim'
local dl = require 'dataload'
local xlua = require 'xlua'
local c = require 'trepl.colorize'
dofile './provider.lua'


opt = lapp[[
   -s,--save            (default "logs")    subdirectory to save logs
   -b,--batchSize         (default 128)      batch size
   -r,--learningRate      (default 1)      learning rate
   --learningRateDecay      (default 1e-7)    learning rate decay
   --weightDecay          (default 0.0005)    weightDecay
   -m,--momentum          (default 0.9)       momentum
   --epoch_step         (default 25)      epoch step
   --model            (default vgg)       model name
   --max_epoch          (default 300)       maximum number of iterations
   --backend            (default cunn)        backend
   --mode             (default L2)        L2/L1/learningrate
   --type             (default cuda)      cuda/float/cl
   --numMeta            (default 3)         #episode
   --hLR              (default 0.0001)     learningrate of hyperparameter
   --initHyper          (default 0.001)      initial value for hyperparameter
]]

print(c.blue "==> " .. "parameter:")
print(opt)

grad.optimize(true)


local function sgd_m(opfunc, x, config, state)
   -- (0) get/update state
   local config = config or {}
   local state = state or config
   local lr = config.learningRate or 1e-3
   local lrd = config.learningRateDecay or 0
   local wd = config.weightDecay or 0
   local mom = config.momentum or 0
   local damp = config.dampening or mom
   local nesterov = config.nesterov or false
   local lrs = config.learningRates
   local wds = config.weightDecays
   local forward_only = config.forward_only or nil
   state.evalCounter = state.evalCounter or 0
   local nevals = state.evalCounter
   assert(not nesterov or (mom > 0 and damp == 0), "Nesterov momentum requires a momentum and zero dampening")


   -- (1) evaluate f(x) and df/dx
   local fx,dfdx = opfunc(x)

   --print(dfdx)

   -- (2) weight decay with single or individual parameters
   if wd ~= 0 then
      for i = 1, #dfdx do
         dfdx[i]:add(wd, x[i])
      end
   elseif wds then
      if not state.decayParameters then
         state.decayParameters = torch.Tensor():typeAs(x):resizeAs(dfdx)
      end
      state.decayParameters:copy(wds):cmul(x)
      dfdx:add(state.decayParameters)
   end

   -- (3) apply momentum
   if mom ~= 0 then
      if not state.dfdx then
         state.dfdx = {}
         for i = 1, #dfdx do
            state.dfdx[i] = torch.Tensor():typeAs(dfdx[i]):resizeAs(dfdx[i]):copy(dfdx[i])
         end
      else
         for i = 1, #dfdx do
            state.dfdx[i]:mul(mom):add(1-damp, dfdx[i])
         end
      end
      if nesterov then
         dfdx:add(mom, state.dfdx)
      else
         dfdx = state.dfdx
      end
   end

   -- (4) learning rate decay (annealing)
   local clr = lr / (1 + nevals*lrd)

   -- (5) parameter update with single or individual learning rates
   if lrs then
      if not state.deltaParameters then
         state.deltaParameters = {}
         for i = 1, #dfdx do
            state.deltaParameters[i] = torch.Tensor():typeAs(x[i]):resizeAs(dfdx[i])
         end
      end
      for i = 1, #dfdx do
         state.deltaParameters[i]:copy(lrs[i]):cmul(dfdx[i])
         x[i]:add(-clr[i], state.deltaParameters[i])
      end
   else
      for i = 1, #dfdx do
         x[i]:add(-clr, dfdx[i])
      end
   end

   -- (6) update evaluation counter
   state.evalCounter = state.evalCounter + 1

   -- return x*, f(x) before optimization
--   if not forward_only then
--      return x, {fx}
--   end


   return x, {fx}
end

-- Load in MNIST
print(c.blue '==>' ..' loading data')

provider = torch.load 'provider.t7'
provider.trainData.data = provider.trainData.data:float()
provider.testData.data = provider.testData.data:float()

local trainset, validset, testset = dl.loadCIFAR10()
local classes = testset.classes
local confusionMatrix = optim.ConfusionMatrix(classes)
print(c.blue '    completed!')


local predict, model, modelf, dfTrain, params, all_params, initParams, finalParams, params_l2, params_velocity, gethessian
local params_proj, dHyperProj
local parameters, gradParameters, grads

local function cast(t)
   if opt.type == 'cuda' then
      require 'cunn'
      return t:cuda()
   elseif opt.type == 'float' then
      return t:float()
   elseif opt.type == 'cl' then
      require 'clnn'
      return t:cl()
   else
      error('Unknown type '..opt.type)
   end
end




confusion = optim.ConfusionMatrix(10)

print('Will save at '..opt.save)
paths.mkdir(opt.save)
testLogger = optim.Logger(paths.concat(opt.save, 'test.log'))
testLogger:setNames{'% mean class accuracy (train set)', '% mean class accuracy (test set)'}
testLogger.showPlot = false


print(c.blue'==>' ..' setting criterion')
criterion = cast(nn.CrossEntropyCriterion())



local function init(iter)
   ----
   --- build VGG net.
   ----

   if iter == 1 then
      -- load model
      print(c.blue '==>' ..' configuring model')
      model = cast(dofile(root .. 'models/'..opt.model..'.lua'))
      -- cast a model using functionalize
      modelf, params = grad.functionalize(model)
      parameters, gradParameters = model:parameters()

      local Lossf = grad.nn.CrossEntropyCriterion()

      -- define training function
      local function fTrain(params, x, y)
         local prediction = modelf(params, x)
         return Lossf(prediction, y), prediction
      end
      dfTrain = grad(fTrain)

      -- a simple unit test
      local X = cast(torch.Tensor(4, 3, 32, 32):fill(0.5))
      local Y = cast(torch.Tensor(1, 4):fill(0))
      local grads, l, p = dfTrain(params, X, Y)
      if (grads) then
        print(c.green '    Auto Diff works!')
      end

      -- build auto-hessian
      local function fhessian(params, X, Y)
         local grads, loss, predition = dfTrain(params, X, Y)
         return loss
      end
      gethessian = grad(fhessian)
      local hessian, loss = gethessian(params, X, Y)
      if (hessian) then
         print(c.green '    Hessian seems works!')
      end

      print(c.blue '    completed!')
   end

   print(c.blue '==>' ..' initializing model')
   --print(params[1])
--   utils.MSRinit(model)
   --print(params[1])

   print(c.blue '    completed!')

   -- copy initial weights for later computation
   initParams = utils.deepcopy(params)
end



local optimState = {
   learningRate = opt.learningRate,
   weightDecay = opt.weightDecay,
   momentum = opt.momentum,
   learningRateDecay = opt.learningRateDecay,
}


local function train()
   model:training()
   local epoch = epoch or 1

   -- drop learning rate every "epoch_step" epochs
   if epoch % opt.epoch_step == 0 then optimState.learningRate = optimState.learningRate/2 end

   print(c.blue '==>'.." online epoch # " .. epoch .. ' [batchSize = ' .. opt.batchSize .. ']')

   local targets = cast(torch.FloatTensor(opt.batchSize))
   local indices = torch.randperm(provider.trainData.data:size(1)):long():split(opt.batchSize)
   -- remove last element so that all the batches have equal size
   indices[#indices] = nil

   local tic = torch.tic()
   for t,v in ipairs(indices) do
      xlua.progress(t, #indices)

      local inputs = provider.trainData.data:index(1,v):cuda()
      targets:copy(provider.trainData.labels:index(1,v)):cuda()

      --- [autograd] begin
      local feval = function(x)
         if x~=params then params:copy(x) end
         for j = 1, #gradParameters do
            gradParameters[j]:zero()
         end
         local loss, prediction
         grads, loss, prediction = dfTrain(params, inputs, targets)
         confusionMatrix:batchAdd(prediction, targets)
         print(c.red 'loss: ', loss)
         return loss, gradParameters
      end
      sgd_m(feval, params, optimState)
      --- [autograd] end


      --- [orginal] begin
--      local feval = function(x)
--         if x ~= parameters then parameters:copy(x) end
--         for j = 1, #gradParameters do
--            gradParameters[j]:zero()
--         end
--         local outputs = model:forward(inputs)
--         local f = criterion:forward(outputs, targets)
--         local df_do = criterion:backward(outputs, targets)
--         model:backward(inputs, df_do)
--         print("loss: ", f)
--         confusion:batchAdd(outputs, targets)
--         return f,gradParameters
--      end
--      sgd_m(feval, parameters, optimState)
      --- [orginal] end

   end

   confusion:updateValids()
   print(('Train accuracy: '..c.cyan'%.2f'..' %%\t time: %.2f s'):format(
      confusion.totalValid * 100, torch.toc(tic)))

   train_acc = confusion.totalValid * 100

   confusion:zero()
   epoch = epoch + 1
end



local function train_meta(iter)

    -----------------------------------
    -- [[Meta training]]
    -----------------------------------


    for epoch = 1, opt.max_epoch do
       train()
   end

   -----------------------
   -- [[Reverse mode hyper-parameter training:
   -- to get gradient w.r.t. hyper-parameters]]
   -----------------------



end

-----------------------------
-- entry point
-----------------------------

local time = sys.clock()


for i = 1, opt.numMeta do
    init(i)
    --    print("wtf", model)
    train_meta(i)
end

time = sys.clock() - time
print(time)