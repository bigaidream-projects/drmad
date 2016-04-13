--[[
One meta-iteration for DrMAD on MNIST
April 13, 2016
MIT license

Modified from torch-autograd's example, train-mnist-mlp.lua
]]

-- Purely stochastic training on purpose, to test the linear subspace hypothesis
-- STATUS: something wrong with hypergradients, line 257

-- Import libs
local th = require 'torch'
local grad = require 'autograd'
local util = require 'autograd.util'
local lossFuns = require 'autograd.loss'
local optim = require 'optim'
-- using https://github.com/slembcke/debugger.lua
--local debugger = require('debugger')

grad.optimize(true)
package.path = package.path .. ";/home/jie/d2/github/bigaidream-projects/drmad/hypergrad_lua/?.lua"


-- Load in MNIST
local fullData, testData, classes = require('get-mnist')()
trainData = {
    size = 50000,
    x = fullData.x[{ { 1, 50000 } }],
    y = fullData.y[{ { 1, 50000 } }]
}

validData = {
    size = 10000,
    x = fullData.x[{ { 50001, 60000 } }],
    y = fullData.y[{ { 50001, 60000 } }]
}

transValidData = {
    size = 10000,
    x = th.FloatTensor(10000, 1, 1024):fill(0),
    y = th.FloatTensor(10000, 1, 10):fill(0)
}

local inputSize = trainData.x[1]:nElement()
local confusionMatrix = optim.ConfusionMatrix(classes)

-- What model to train:
local predict, fTrain, params

-- Define our neural net
function predict(params, input)
    local h1 = th.tanh(input * params.W[1] + params.B[1])
    local h2 = th.tanh(h1 * params.W[2] + params.B[2])
    local h3 = h2 * params.W[3] + params.B[3]
    local out = util.logSoftMax(h3)
    return out
end

-- Define training loss
function fTrain(params, input, target)
    local prediction = predict(params, input)
    local loss = lossFuns.logMultinomialLoss(prediction, target)
    local penalty1 = th.sum(th.cmul(th.cmul(params.W[1], params.HY[1]), params.W[1]))
    local penalty2 = th.sum(th.cmul(th.cmul(params.W[2], params.HY[2]), params.W[2]))
    local penalty3 = th.sum(th.cmul(th.cmul(params.W[3], params.HY[3]), params.W[3]))
    loss = loss + penalty1 + penalty2 + penalty3
    return loss, prediction
end


-- Define elementary parameters
-- [-1/sqrt(#output), 1/sqrt(#output)]
th.manualSeed(0)
local W1 = th.FloatTensor(inputSize, 50):uniform(-1 / math.sqrt(50), 1 / math.sqrt(50))
local B1 = th.FloatTensor(50):fill(0)
local W2 = th.FloatTensor(50, 50):uniform(-1 / math.sqrt(50), 1 / math.sqrt(50))
local B2 = th.FloatTensor(50):fill(0)
local W3 = th.FloatTensor(50, #classes):uniform(-1 / math.sqrt(#classes), 1 / math.sqrt(#classes))
local B3 = th.FloatTensor(#classes):fill(0)

local initHyper = 0.001
local HY1 = th.FloatTensor(inputSize, 50):fill(initHyper)
local HY2 = th.FloatTensor(50, 50):fill(initHyper)
local HY3 = th.FloatTensor(50, #classes):fill(initHyper)



-- Trainable parameters and hyperparameters:
params = {
    W = { W1, W2, W3 },
    B = { B1, B2, B3 },
    HY = { HY1, HY2, HY3 }
}

local deepcopy = require 'deepcopy'

-- copy initial weights
initParams = deepcopy(params)

-- Get the gradients closure magically:
local dfTrain = grad(fTrain, { optimize = true })

------------------------------------
-- [[Forward pass]]
-----------------------------------

-- elementary learning rate
local eLr = 0.01
local numEpoch = 1
-- Train a neural network to get final parameters
for epoch = 1, numEpoch do
    print('Forward Training Epoch #' .. epoch)
    for i = 1, trainData.size do
        -- Next sample:
        local x = trainData.x[i]:view(1, inputSize)
        local y = th.view(trainData.y[i], 1, 10)

        -- Grads:
        local grads, loss, prediction = dfTrain(params, x, y)

        -- Update weights and biases at each layer
        for i = 1, #params.W do
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
finalParams = deepcopy(params)


----------------------
-- [[Backward pass]]
-----------------------

local shallowcopy = require 'shallowcopy'
-- Transform validation data

for t = 1, validData.size do
    transValidData.x[t] = shallowcopy(validData.x[t]:view(1, inputSize))
    transValidData.y[t] = shallowcopy(th.view(validData.y[t], 1, 10))
end

-- Define validation loss
validLoss = 0

function fValid(params, input, target)
    local prediction = predict(params, input)
    local loss = lossFuns.logMultinomialLoss(prediction, target)
    return loss, prediction
end
local dfValid = grad(fValid, { optimize = true })

-- Initialize validGrads
local VW1 = th.FloatTensor(inputSize, 50):fill(0)
local VB1 = th.FloatTensor(50):fill(0)
local VW2 = th.FloatTensor(50, 50):fill(0)
local VB2 = th.FloatTensor(50):fill(0)
local VW3 = th.FloatTensor(50, #classes):fill(0)
local VB3 = th.FloatTensor(#classes):fill(0)

validGrads = {
    W = { VW1, VW2, VW3 },
    B = { VB1, VB2, VB3 }
}

-- Get gradient of validation loss w.r.th. finalParams
-- Test network to get validation gradients w.r.t weights
for epoch = 1, numEpoch do
    print('Forward Training Epoch #' .. epoch)
    for i = 1, transValidData.size do
        -- Next sample:
        local x = transValidData.x[i]:view(1, inputSize)
        local y = th.view(transValidData.y[i], 1, 10)

        -- Grads:
        local grads, loss, prediction = dfValid(params, x, y)
        for i = 1, #params.W do
            validGrads.W[i] = validGrads.W[i] + grads.W[i]
            validGrads.B[i] = validGrads.B[i] - grads.B[i]
        end
    end
end

-- Get average validation gradients w.r.t weights and biases
for i = 1, #params.W do
    validGrads.W[i] = validGrads.W[i]/numEpoch
    validGrads.B[i] = validGrads.B[i]/numEpoch
end

-------------------------------------

-- Initialize derivative w.r.th. hyperparameters
local DHY1 = th.FloatTensor(inputSize, 50):fill(0)
local DHY2 = th.FloatTensor(50, 50):fill(0)
local DHY3 = th.FloatTensor(50, #classes):fill(0)
local DHY = { DHY1, DHY2, DHY3 }


local nLayers = 3
local proj1 = th.zero(th.FloatTensor(inputSize, 50))
local proj2 = th.zero(th.FloatTensor(50, 50))
local proj3 = th.zero(th.FloatTensor(50, #classes))


-- Initialize derivative w.r.t. velocity
local DV1 = th.FloatTensor(inputSize, 50):fill(0)
local DV2 = th.FloatTensor(50, 50):fill(0)
local DV3 = th.FloatTensor(50, #classes):fill(0)
local DV = { DV1, DV2, DV3 }


-- https://github.com/twitter/torch-autograd/issues/66
-- torch-autograd needs to track all variables
function gradProj(params, input, target, proj_1, proj_2, proj_3, DV_1, DV_2, DV_3)
    local grads, loss, prediction = dfTrain(params, input, target)
    proj_1 = proj_1 + th.cmul(grads.W[1] , DV_1)
    proj_2 = proj_2 + th.cmul(grads.W[2] , DV_2)
    proj_3 = proj_3 + th.cmul(grads.W[3] , DV_3)
    local loss = th.sum(proj_1) + th.sum(proj_2) + th.sum(proj_3)
    return loss
end
local dHVP = grad(gradProj)

----------------------------------------------
-- Backpropagate the validation errors
numIter = numEpoch * (trainData.size)
local beta = th.linspace(0.001, 0.999, numIter)

-- learning rate for hyperparameters
local hLr = 0.01

for epoch = 1, numEpoch do

    print('Backword Training Epoch #' .. epoch)
    for i = 1, trainData.size do
        -- Next sample:
        local x = trainData.x[i]:view(1, inputSize)
        local y = th.view(trainData.y[i], 1, 10)
        for j = 1, nLayers do
            params.W[j] = th.mul(initParams.W[j], (1 - beta[i + (numEpoch * (epoch-1))])) +
                    th.mul(finalParams.W[j], beta[i + (numEpoch * (epoch-1))])
            DV[j] = DV[j] + validGrads.W[j] * eLr
        end
        local grads, loss = dHVP(params, x, y, proj1, proj2, proj3, DV1, DV2, DV3)
        for j = 1, nLayers do
            validGrads.W[j] = validGrads.W[j] - th.mul(grads.W[j], (1.0 - hLr))
            -- grads w.r.t. HY are all zeros
            DHY[j] = DHY[j] - th.mul(grads.HY[j], (1.0 - hLr))
            debugger()
            DV[j] = th.mul(DV[j], hLr)
        end
    end
end

print(DHY2)