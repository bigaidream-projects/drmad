--[[
One meta-iteration for DrMAD on MNIST
March 5, Jie Fu, http://bigaidream.github.io/contact.html
MIT license

Modified from torch-autograd's example, train-mnist-mlp.lua
]]

-- L1 penalty for now, due to the lack of torch.dot support needed by L2 penalty
-- Purely stochastic training on purpose, to test the linear subspace hypothesis

-- Import libs
require 'torch'
local grad = require 'autograd'
local util = require 'autograd.util'
local lossFuns = require 'autograd.loss'
local optim = require 'optim'
local debugger = require('fb.debugger')

grad.optimize(true)
package.path = package.path .. ";/home/jie/Documents/lua_workspace/torch-autograd/source/examples/?.lua"
-- Load in MNIST
local fullData, testData, classes = require('get-mnist')()
trainData = {
  size = fullData.size,
  x = fullData.x[{{1, 50000}}],
  y = fullData.y[{{1, 50000}}]
}

validData = {
  size = fullData.size,
  x = fullData.x[{{50001, 60000}}],
  y = fullData.y[{{50001, 60000}}]
}


local inputSize = trainData.x[1]:nElement()
--debugger.enter()
local confusionMatrix = optim.ConfusionMatrix(classes)

-- What model to train:
local predict,fTrain,params

-- Define our neural net
function predict(params, input, target)
   local h1 = torch.tanh(input * params.W[1] + params.B[1])
   local h2 = torch.tanh(h1 * params.W[2] + params.B[2])
   local h3 = h2 * params.W[3] + params.B[3]
   local out = util.logSoftMax(h3)
   return out
end

-- Define training loss
function fTrain(params, input, target)
   local prediction = predict(params, input, target)
   local loss = lossFuns.logMultinomialLoss(prediction, target)
   local penalty1 = torch.sum(torch.abs(torch.cmul(params.W[1], params.HY[1])))
   local penalty2 = torch.sum(torch.abs(torch.cmul(params.W[2], params.HY[2])))
   local penalty3 = torch.sum(torch.abs(torch.cmul(params.W[3], params.HY[3])))
   loss = loss + penalty1 + penalty2 + penalty3
   return loss, prediction
end



-- Define elementary parameters
-- [-1/sqrt(#output), 1/sqrt(#output)]
torch.manualSeed(0)
local W1 = torch.FloatTensor(inputSize,50):uniform(-1/math.sqrt(50),1/math.sqrt(50))
local B1 = torch.FloatTensor(50):fill(0)
local W2 = torch.FloatTensor(50,50):uniform(-1/math.sqrt(50),1/math.sqrt(50))
local B2 = torch.FloatTensor(50):fill(0)
local W3 = torch.FloatTensor(50,#classes):uniform(-1/math.sqrt(#classes),1/math.sqrt(#classes))
local B3 = torch.FloatTensor(#classes):fill(0)

local initHyper = 0.001
local HY1 = torch.FloatTensor(inputSize,50):fill(initHyper)
local HY2 = torch.FloatTensor(50,50):fill(initHyper)
local HY3 = torch.FloatTensor(50,#classes):fill(initHyper)

-- Trainable parameters and hyperparameters:
params = {
   W = {W1, W2, W3},
   B = {B1, B2, B3},
   HY = {HY1, HY2, HY3}
}


-- copy initial weights
-- TODO: deep copy
initParams = params


-- Get the gradients closure magically:
local dfTrain = grad(fTrain, { optimize = true })

------------------------------------
-- Forward pass
-----------------------------------

-- elementary learning rate
local eLr = 0.01
local numEpoch = 100
-- Train a neural network to get final parameters
for epoch = 1,numEpoch do
   print('Forward Training Epoch #'..epoch)
   for i = 1,trainData.size do
      -- Next sample:
      local x = trainData.x[i]:view(1,inputSize)
      local y = torch.view(trainData.y[i], 1, 10)

      -- Grads:
      local grads, loss, prediction = dfTrain(params, x, y)

      -- Update weights and biases at each layer
      for i=1,#params.W do
         params.W[i] = params.W[i] - grads.W[i] * eLr
         params.B[i] = params.B[i] - grads.B[i] * eLr
      end

      -- Log performance:
      confusionMatrix:add(prediction[1], y[1])
      if i % 1000 == 0 then
         print(confusionMatrix)
         print(epoch)
         confusionMatrix:zero()
      end
   end
end

-- copy final parameters after convergence
-- TODO: deep copy
finalParams = params


----------------------
-- Backward pass
-----------------------

-- Define validation loss
function fValid(params)
  --confusionMatrix:zero()
  local loss
  print('==> testing on valid set:')
    for t = 1,validData.size do
      -- disp progress
      xlua.progress(t, validData.size)
    
      -- get new sample
      local input = validData.x[t]
    --  if opt.type == 'double' then input = input:double()
    --  elseif opt.type == 'cuda' then input = input:cuda() end
      local target = validData.y[t]
      local prediction = predict(params, input, target)
      local loss = lossFuns.logMultinomialLoss(prediction, target)
      
    --  local pred = model:forward(input)
    --  confusionMatrix:add(pred, target)
    end
  loss = loss/(validData.size)
  return loss
end




-------------------------------------

-- Initialize derivative w.r.t. velocity
local DV1 = torch.FloatTensor(inputSize,50):fill(0)
local DV2 = torch.FloatTensor(50,50):fill(0)
local DV3 = torch.FloatTensor(50,#classes):fill(0)
local DV = {DV1, DV2, DV3}


-- Initialize derivative w.r.t. hyperparameters
local DHY1 = torch.FloatTensor(inputSize,50):fill(0)
local DHY2 = torch.FloatTensor(50,50):fill(0)
local DHY3 = torch.FloatTensor(50,#classes):fill(0)
local DHY = {DHY1, DHY2, DHY3}

-- Initialize derivative w.r.t. weights
local DW1 = torch.FloatTensor(inputSize,50):fill(0)
local DW2 = torch.FloatTensor(50,50):fill(0)
local DW3 = torch.FloatTensor(50,#classes):fill(0)
local DW = {DW1, DW2, DW3}



-- Get gradient of validation loss w.r.t. finalParams
local dfValid = grad(fValid, { optimize = true })
local validGrads, validLoss = dfValid(params)

function gradProj(params, input, target)
    local proj = 0
    local grads, loss, prediction = dfTrain(params, input, target)
    for i = 1,#params.W do
       proj = proj + torch.dot(grads.W[i], DV[i])
    end
    return proj
end

local dHVP = grad(gradProj, { optimize = true })

-- Backpropagate the valid errors
numIter = numEpoch*(trainData.size)
local beta = torch.linspace(0.001, 0.999, numIer)
-- learning rate for hyperparameters
local hLr

for epoch = 1,numEpoch do


   print('Backword Training Epoch #'..epoch)
   for i = 1,trainData.size do
      -- Next sample:
      local x = trainData.x[i]:view(1,inputSize)
      local y = torch.view(trainData.y[i], 1, 10)
      for j=1,#params.W do
         params.W[j] = torch.mul(initParams.W[j], (1-beta[-numEpoch*(i+epoch)]) )+torch.mul(finalParams.W[j], beta[-numEpoch*(i+epoch)])
         DV[j] = DV[j] + validDW[j]* eLr
      end
      grads, loss = dHVP(params, x, y)      
      for j=1,#params.W do
         validGrads.W[j] = validGrads.W[j] -torch.mul(grads.W[j], (1.0 - hLr))
         DHY[j] = DHY[j] - torch.mul(grads.HY[j], (1.0 - hLr))
         DV[j] = torch.mul(DV[j], hLr)
      end

      -- Log performance:
      confusionMatrix:add(prediction[1], y[1])
      if i % 1000 == 0 then
         print(confusionMatrix)
         print(epoch)
         confusionMatrix:zero()
      end
   end
end

