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


opt = lapp[[
   -s,--save            (default "logs")    subdirectory to save logs
   -b,--batchSize         (default 64)      batch size
   -r,--learningRate      (default 0.0001)      learning rate
   --learningRateDecay      (default 1e-7)    learning rate decay
   --weightDecay          (default 0.0005)    weightDecay
   -m,--momentum          (default 0.9)       momentum
   --epoch_step         (default 25)      epoch step
   --model            (default vgg)       model name
   --max_epoch          (default 300)       maximum number of iterations
   --backend            (default nn)        backend
   --mode             (default L2)        L2/L1/learningrate
   --type             (default cuda)      cuda/float/cl
   --numMeta            (default 3)         #episode
   --hLR              (default 0.0001)     learningrate of hyperparameter
   --initHyper          (default 0.001)      initial value for hyperparameter
]]

print(c.blue "==> " .. "parameter:")
print(opt)

grad.optimize(true)

-- Load in MNIST

print(c.blue '==>' ..' loading data')
local trainset, validset, testset = dl.loadCIFAR10()
local classes = testset.classes
local confusionMatrix = optim.ConfusionMatrix(classes)
print(c.blue '    completed!')


local predict, model, modelf, dfTrain, params, all_params, initParams, finalParams, params_l2, params_velocity
local params_proj, dHyperProj


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


-- create params that has a same size as elementary parameters
local function full_create(params)
   local full_params = {}
   for i = 1, #params do
      full_params[i] = params[i]:clone():fill(0)
   end
   return full_params
end

-- create params that has a same size as elementary weight parameters
-- i.e. ignore bias parameters
local function L2_norm_create(params, initHyper)
   local hyper_L2 = {}
   for i = 1, #params do
      -- dimension = 1 is bias, do not need L2_reg
      if (params[i]:nDimension() > 1) then
        hyper_L2[i] = params[i]:clone():fill(initHyper)
      end
   end
   return hyper_L2
end

local function L2_norm(params, params_l2)
--   local penalty = torch.sum(params[1]) * params_l2[1]
   local penalty = 0
   for i = 1, #params do
       --dimension = 1 is bias, do not need L2_reg
       if (params[i]:nDimension() > 1) then
         --print(i)
         penalty = penalty + torch.sum(torch.cmul(params[i], params_l2[i]))
       end
   end
    return penalty
end


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

      params_l2 = L2_norm_create(params, opt.initHyper)
      params_velocity = full_create(params)
      params_proj = full_create(params)

      local Lossf = grad.nn.MSECriterion()

      -- define training function
      local function fTrain(params, x, y)
         --print(params.elementary)
         --print(params.l2)
         local prediction = modelf(params.elementary, x)
         local penalty = L2_norm(params.elementary, params.l2)
         return Lossf(prediction, y) + penalty, prediction
      end

      dfTrain = grad(fTrain)

      -- a simple unit test
      local X = cast(torch.Tensor(4, 3, 32, 32):fill(0.5))
      local Y = cast(torch.Tensor(4, 10):fill(0))

      all_params = {
         elementary = params,
         l2 = params_l2,
         velocity = params_velocity
      }

      local dparams, l, p = dfTrain(all_params, X, Y)

      if (l) then
        print(c.green '    Auto Diff works!')
      end

      print(c.blue '    completed!')
   end

   print(c.blue '==>' ..' initializing model')
   --print(params[1])
   utils.MSRinit(model)
   --print(params[1])

   print(c.blue '    completed!')

   -- copy initial weights for later computation
   initParams = utils.deepcopy(params)
end


local function gradProj(params, input, target, Proj, dV)
--   local grads, loss, prediction = dfTrain(params, input, target)
--   proj_1 = proj_1 + torch.cmul(grads.W[1] , DV_1)
--   proj_2 = proj_2 + torch.cmul(grads.W[2] , DV_2)
--   proj_3 = proj_3 + torch.cmul(grads.W[3] , DV_3)
--   local loss = torch.sum(proj_1) + torch.sum(proj_2) + torch.sum(proj_3)
--   return loss

end

local function train_meta(iter)

    -----------------------------------
    -- [[Meta training]]
    -----------------------------------

    -- Train a neural network to get final parameters

   for epoch = 1, opt.max_epoch do
      print(c.blue '==>' ..' Meta episode #' .. iter .. ', Training epoch #' .. epoch)
      for i, inputs, targets in trainset:subiter(opt.batchSize) do

         local function makesample(inputs, targets)
            local t_ = torch.FloatTensor(targets:size(1), 10):zero()
            for j = 1, targets:size(1) do
               t_[targets[j]] = 1
            end
            return cast(inputs), cast(t_)
         end

         local X, Y = makesample(inputs, targets)
         local grads, loss, prediction = dfTrain(all_params, X, Y)

         -- update parameter
         for i = 1, #grads do
            params_velocity[i] = params_velocity[i]:mul(opt.learningRateDecay) - grads[i]:mul(1 - opt.learningRateDecay)
            params[i] = params[i] + opt.learningRate * params_velocity[i]
         end

         -- Log performance:
         confusionMatrix:batchAdd(prediction, Y)
         if i % 1000 == 0 then
            print("Epoch "..epoch)
            print(confusionMatrix)
            confusionMatrix:zero()
         end
         print(c.red 'loss: ', loss)
      end
   end

   -- copy final parameters after convergence
   finalParams = utils.deepcopy(params)
   finalParams = nn.utils.recursiveCopy(finalParams, params)

   -----------------------
   -- [[Reverse mode hyper-parameter training:
   -- to get gradient w.r.t. hyper-parameters]]
   -----------------------
    dHyperProj = grad(gradProj)



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