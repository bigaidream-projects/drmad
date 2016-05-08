--[[
One meta-iteration for DrMAD on MNIST
MIT license

Modified from torch-autograd's example, train-mnist-mlp.lua
]]

-- Purely stochastic training on purpose,
-- to test the linear subspace hypothesis, epochSize=-1

-- Import libs
require 'cutorch'
local grad = require 'autograd'
local util = require 'autograd.util'
local lossFuns = require 'autograd.loss'
local optim = require 'optim'
local dl = require 'dataload'
local xlua = require 'xlua'
cutorch.setDevice(2)
grad.optimize(true)

-- Load in MNIST
local trainset, validset, testset = dl.loadMNIST()
local transValidData = {
    size = 10000,
    x = torch.CudaTensor(10000, 1, 28*28):fill(0),
    y = torch.CudaTensor(10000, 1, 10):fill(0)
}
local inputSize = trainset.inputs[1]:nElement()
local classes = testset.classes
local confusionMatrix = optim.ConfusionMatrix(classes)

local initHyper = 0.001
local predict, fTrain, params

-- initialize hyperparameters as global variables
-- to be shared across different meta-iterations
local HY1 = torch.CudaTensor(inputSize, 50):fill(initHyper)
local HY2 = torch.CudaTensor(50, 50):fill(initHyper)
local HY3 = torch.CudaTensor(50, #classes):fill(initHyper)

local function train_meta()
    --[[
    One meta-iteration to get directives w.r.t. hyperparameters
    ]]
    -- What model to train:


    -- Define our neural net
    function predict(params, input)
        local h1 = torch.tanh(input * params.W[1] + params.B[1])
        local h2 = torch.tanh(h1 * params.W[2] + params.B[2])
        local h3 = h2 * params.W[3] + params.B[3]
        local out = util.logSoftMax(h3)
        return out
    end

    -- Define training loss
    function fTrain(params, input, target)
        local prediction = predict(params, input)
        local loss = lossFuns.logMultinomialLoss(prediction, target)
        local penalty1 = torch.sum(torch.cmul(torch.cmul(params.W[1], params.HY[1]), params.W[1]))
        local penalty2 = torch.sum(torch.cmul(torch.cmul(params.W[2], params.HY[2]), params.W[2]))
        local penalty3 = torch.sum(torch.cmul(torch.cmul(params.W[3], params.HY[3]), params.W[3]))
        loss = loss + penalty1 + penalty2 + penalty3
        return loss, prediction
    end


    -- Define elementary parameters
    -- [-1/sqrt(#output), 1/sqrt(#output)]
    torch.manualSeed(0)
    local W1 = torch.CudaTensor(inputSize, 50):uniform(-1 / math.sqrt(50), 1 / math.sqrt(50))
    local B1 = torch.CudaTensor(50):fill(0)
    local W2 = torch.CudaTensor(50, 50):uniform(-1 / math.sqrt(50), 1 / math.sqrt(50))
    local B2 = torch.CudaTensor(50):fill(0)
    local W3 = torch.CudaTensor(50, #classes):uniform(-1 / math.sqrt(#classes), 1 / math.sqrt(#classes))
    local B3 = torch.CudaTensor(#classes):fill(0)



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
    -- set it small to avoid NaN issue
    local eLr = 0.0001

    local numEpoch = 1
    local epochSize = -1

    -- weight decay for elementary parameters
    local gamma = 0.1
    -- Train a neural network to get final parameters
    local y_ = torch.CudaTensor(10)
    local function makesample(inputs, targets)
        assert(inputs:size(1) == 1)
        assert(inputs:dim() == 4)
        --assert(torch.type(inputs) == 'torch.CudaTensor')
        local x = inputs:view(1, -1)
        y_:zero()
        y_[targets[1]] = 1 -- onehot
        return x, y_:view(1, 10)
    end

    for epoch = 1, numEpoch do
        print('Forward Training Epoch #' .. epoch)
        for i, inputs, targets in trainset:subiter(1, epochSize) do
            -- Next sample:
            local x, y = makesample(inputs:cuda(), targets:cuda())

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
                print("Epoch "..epoch)
                print(confusionMatrix)
                confusionMatrix:zero()
            end
        end
    end

    -- copy final parameters after convergence
    finalParams = deepcopy(params)

    ----------------------
    -- [[Backward pass]]
    -----------------------

    -- Transform validation data

    transValidData.y:zero()
    for t, inputs, targets in validset:subiter(1, epochSize) do
        inputs = inputs:cuda()
        targets = targets:cuda()
        transValidData.x[t]:copy(inputs:view(-1))
        transValidData.y[{t,1,targets[1]}] = 1 -- onehot
    end

    -- Define validation loss
    local validLoss = 0

    function fValid(params, input, target)
        local prediction = predict(params, input)
        local loss = lossFuns.logMultinomialLoss(prediction, target)
        return loss, prediction
    end
    local dfValid = grad(fValid, { optimize = true })

    -- Initialize validGrads
    local VW1 = torch.CudaTensor(inputSize, 50):fill(0)
    local VB1 = torch.CudaTensor(50):fill(0)
    local VW2 = torch.CudaTensor(50, 50):fill(0)
    local VB2 = torch.CudaTensor(50):fill(0)
    local VW3 = torch.CudaTensor(50, #classes):fill(0)
    local VB3 = torch.CudaTensor(#classes):fill(0)

    local validGrads = {
        W = { VW1, VW2, VW3 },
        B = { VB1, VB2, VB3 }
    }

    -- Get gradient of validation loss w.r.th. finalParams
    -- Test network to get validation gradients w.r.t weights
    for epoch = 1, numEpoch do
        print('Forward Training Epoch #' .. epoch)
        for i = 1, epochSize == -1 and transValidData.size or epochSize do
            -- Next sample:
            local x = transValidData.x[i]:view(1, inputSize)
            local y = torch.view(transValidData.y[i], 1, 10)

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
    DHY1 = torch.CudaTensor(inputSize, 50):fill(0)
    DHY2 = torch.CudaTensor(50, 50):fill(0)
    DHY3 = torch.CudaTensor(50, #classes):fill(0)
    DHY = { DHY1, DHY2, DHY3 }


    local nLayers = 3
    local proj1 = torch.CudaTensor(inputSize, 50):zero()
    local proj2 = torch.CudaTensor(50, 50):zero()
    local proj3 = torch.CudaTensor(50, #classes):zero()


    -- Initialize derivative w.r.t. velocity
    local DV1 = torch.CudaTensor(inputSize, 50):fill(0)
    local DV2 = torch.CudaTensor(50, 50):fill(0)
    local DV3 = torch.CudaTensor(50, #classes):fill(0)
    local DV = { DV1, DV2, DV3 }


    -- https://github.com/twitter/torch-autograd/issues/66
    -- torch-autograd needs to track all variables
    local function gradProj(params, input, target, proj_1, proj_2, proj_3, DV_1, DV_2, DV_3)
        local grads, loss, prediction = dfTrain(params, input, target)
        proj_1 = proj_1 + torch.cmul(grads.W[1] , DV_1)
        proj_2 = proj_2 + torch.cmul(grads.W[2] , DV_2)
        proj_3 = proj_3 + torch.cmul(grads.W[3] , DV_3)
        local loss = torch.sum(proj_1) + torch.sum(proj_2) + torch.sum(proj_3)
        return loss
    end
    local dHVP = grad(gradProj)

    ----------------------------------------------
    -- Backpropagate the validation errors
    local numIter = numEpoch * (epochSize == -1 and trainset:size() or epochSize)
    local beta = torch.linspace(0.001, 0.999, numIter)


    local buffer
    for epoch = 1, numEpoch do

        print('Backword Training Epoch #' .. epoch)
        for i, inputs, targets in trainset:subiter(1, epochSize) do
            -- Next sample:
            local x, y = makesample(inputs:cuda(), targets:cuda())

            for j = 1, nLayers do
                params.W[j]:mul(initParams.W[j], 1 - beta[i + (numEpoch * (epoch-1))])
                buffer = buffer or initParams.W[j].new()
                buffer:mul(finalParams.W[j], beta[i + (numEpoch * (epoch-1))])
                params.W[j]:add(buffer)
                DV[j]:add(eLr, validGrads.W[j])
            end

            local grads, loss = dHVP(params, x, y, proj1, proj2, proj3, DV1, DV2, DV3)
            --        print("loss", loss)
            for j = 1, nLayers do
                buffer = buffer or DHY[j].new()

                buffer:mul(grads.W[j], 1.0 - gamma)
                validGrads.W[j]:add(-1, buffer)

                buffer:mul(grads.HY[j], 1.0 - gamma)
                DHY[j]:add(-1, buffer)

                DV[j]:mul(DV[j], gamma)
            end
            --xlua.progress(i, trainset:size())
        end
    end
    return DHY
end

-----------------------------
-- entry point
------------------------

-- Hyperparameter learning rate, cannot be too huge
-- this is a super-parameter...
local hLr = 0.0001
local numMeta = 3

for i = 1, numMeta do
    local dhy = train_meta()
    for j = 1, #params.W do
        dhy[j]:mul(-hLr)
        params.HY[j]:add(dhy[j])
    end
end

for i, hy in ipairs(params.HY) do
    print("HY "..i, hy:sum())
end