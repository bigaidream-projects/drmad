--
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
--  The training loop and learning rate schedule
--

local optim = require 'optim'
local autograd = require 'autograd'
local nn = require 'nn'

require('../optimizer/DrMAD.lua')
local drmad = DrMAD()


local function L2_calc(params, l2)
   local penalty = 0
   assert(params)
   assert(l2)
   for i = 1, #params do
      if params[i]:size(1) > 1 then
--         print(penalty)
         penalty = penalty + torch.sum(torch.cmul(params[i], params[i]))
      end
   end
--   print("penalty: ", penalty)
--   print("penalty: ", penalty:float())
   penalty = torch.sum(penalty * l2)
   return penalty
end


local M = {}
local Trainer = torch.class('resnet.Trainer', M)

function Trainer:__init(model, criterion, opt, optimState)
   self.model = model
   --tmp, self.gradpramas = model:parameters()
   --print(self.gradpramas[1])
   self.criterion = criterion
   self.optimState = optimState or {
      learningRate = opt.LR,
      learningRateDecay = 1e-7,
      momentum = opt.momentum,
      --      nesterov = true,
      --      dampening = 0.0,
      --      weightDecay = opt.weightDecay,
      hyper = {},
   }
   self.opt = opt
   self.params, self.gradParams = model:parameters()
   print('params size: ' .. #self.gradParams)

   self.all_params = {
      elementary = self.params,
      meta = {
         L2 = torch.CudaTensor(1):fill(opt.weightDecay),
      }
   }

   self.modelf = autograd.functionalize(model)
   --parameters, gradParameters = model:parameters()

   self.loss = autograd.nn.CrossEntropyCriterion()

   --- define training function
   local function fTrain(params, x, y)
      local prediction = self.modelf(params.elementary, x)
      local loss = self.loss(prediction, y) + L2_calc(params.elementary, params.meta.L2)
      return loss, prediction
   end

   self.dfTrain = autograd(fTrain)

   --- exactly the same of training function, but remove penalty
   local function fValid(params, x, y)
      local prediction = self.modelf(params.elementary, x)
      return self.loss(prediction, y), prediction
   end


   self.dfValid = autograd(fValid)

   self.valid_grads = {}
   nn.utils.recursiveResizeAs(self.valid_grads, self.all_params)
   print("self.all_params: ", self.all_params)

   print("self.valid_grads: ", self.valid_grads)

      --   for i = 1, #self.params do
--      self.valid_grads.elementary[i] = torch.Tensor():typeAs(self.params[i]):resizeAs(self.params[i]):fill(0)
--   end
--   self.valid_grads.meta.L2 = 0


   --- define hessian function
   --- to calc dWdW and dmetadW
   local function fHessian(hessian_params, x, y)
      local grads, loss = self.dfTrain(hessian_params.elementary, x, y)
      local res = 0
      for i = 1, #grads.elementary do
         res = res + torch.sum(grads.elementary[i])
      end
      res = res + grads.meta.L2
      return torch.sum(res)                 -- you just need to sum up everything, a grammar sugar
   end

   self.dfHessian = autograd(fHessian)
end

function Trainer:train(epoch, dataloader)
   -- Trains the model for a single epoch
   self.optimState.learningRate = self:learningRate(epoch)

   local timer = torch.Timer()
   local dataTimer = torch.Timer()

   local function feval()
      return self.criterion.output, self.gradParams --, self.gradParams
   end

   local function feval_autograd(x)
      return self.autograd_output, self.gradParams
   end

   local trainSize = dataloader:size()
   local top1Sum, top5Sum, lossSum = 0.0, 0.0, 0.0
   local N = 0

   print('=> Training epoch # ' .. epoch)
   -- set the batch norm to training mode
   self.model:training()
   for n, sample in dataloader:run() do
      local dataTime = dataTimer:time().real

      -- Copy input and target to the GPU
      self:copyInputs(sample)
      local batchSize = self.target:size(1)


      --- [original] begin
      -- print('using origin version')
      -- local output = self.model:forward(self.input):float()
      -- local loss = self.criterion:forward(self.model.output, self.target)
      --
      -- self.model:zeroGradParameters()
      -- self.criterion:backward(self.model.output, self.target)
      -- self.model:backward(self.input, self.criterion.gradInput)
      --
      -- sgd_m(feval, self.params, self.optimState)
      --- [original] end


      --- [autograd] begin
      print('using autograd version')
      self.model:zeroGradParameters()
      local tmp, output
      tmp, self.autograd_output, output = self.dfTrain(self.all_params, self.input, self.target)
      output:float()
      local loss = self.autograd_output
      drmad.sgd(feval_autograd, self.params, self.optimState) --, self.dfhessian, self.dfValid)
      --- [autograd] end


      local top1, top5 = self:computeScore(output, sample.target, 1)

      --      print(top1, top5)

      top1Sum = top1Sum + top1 * batchSize
      top5Sum = top5Sum + top5 * batchSize
      lossSum = lossSum + loss * batchSize
      N = N + batchSize


      --      print('loss: ', loss)
      print((' | Epoch: [%d][%d/%d]    Time %.3f  Data %.3f  Err %1.4f  top1 %7.3f  top5 %7.3f'):format(epoch, n, trainSize, timer:time().real, dataTime, loss, top1, top5))

      -- check that the storage didn't get changed do to an unfortunate getParameters call
      --assert(self.params:storage() == self.model:parameters()[1]:storage())

      timer:reset()
      dataTimer:reset()
   end

   return top1Sum / N, top5Sum / N, lossSum / N
end


function Trainer:valid(epoch, dataloader)
   -- Computes the top-1 and top-5 err on the validation set

   local timer = torch.Timer()
   local dataTimer = torch.Timer()
   local size = dataloader:size()

   local nCrops = self.opt.tenCrop and 10 or 1
   local top1Sum, top5Sum = 0.0, 0.0
   local N = 0

   local valid_loss = 0
   nn.utils.recursiveFill(self.valid_grads, 0)

   self.model:evaluate()
   for n, sample in dataloader:run() do
      local dataTime = dataTimer:time().real

      -- Copy input and target to the GPU
      self:copyInputs(sample)

      local grads, output, loss
      grads, loss, output = self.dfValid(self.all_params, self.input, self.target)
      local batchSize = output:size(1) / nCrops

      --      self.valid_grads

      --      local output = self.model:forward(self.input):float()
      --      local batchSize = output:size(1) / nCrops
      --      local loss = self.criterion:forward(self.model.output, self.target)
      valid_loss = valid_loss + loss
      local top1, top5 = self:computeScore(output, sample.target, nCrops)
      top1Sum = top1Sum + top1 * batchSize
      top5Sum = top5Sum + top5 * batchSize
      N = N + batchSize
      timer:reset()
      dataTimer:reset()
   end
   self.model:training()

   print((' * Finished epoch # %d     top1: %7.3f  top5: %7.3f\n'):format(epoch, top1Sum / N, top5Sum / N))

   return top1Sum / N, top5Sum / N
end

function Trainer:test(epoch, dataloader)
   -- Computes the top-1 and top-5 err on the validation set

   local timer = torch.Timer()
   local dataTimer = torch.Timer()
   local size = dataloader:size()

   local nCrops = self.opt.tenCrop and 10 or 1
   local top1Sum, top5Sum = 0.0, 0.0
   local N = 0

   self.model:evaluate()
   for n, sample in dataloader:run() do
      local dataTime = dataTimer:time().real

      -- Copy input and target to the GPU
      self:copyInputs(sample)

      local output = self.model:forward(self.input):float()
      local batchSize = output:size(1) / nCrops
      local loss = self.criterion:forward(self.model.output, self.target)

      local top1, top5 = self:computeScore(output, sample.target, nCrops)
      top1Sum = top1Sum + top1 * batchSize
      top5Sum = top5Sum + top5 * batchSize
      N = N + batchSize

      print((' | Test: [%d][%d/%d]    Time %.3f  Data %.3f  top1 %7.3f (%7.3f)  top5 %7.3f (%7.3f)'):format(epoch, n, size, timer:time().real, dataTime, top1, top1Sum / N, top5, top5Sum / N))

      timer:reset()
      dataTimer:reset()
   end
   self.model:training()

   print((' * Finished epoch # %d     top1: %7.3f  top5: %7.3f\n'):format(epoch, top1Sum / N, top5Sum / N))

   return top1Sum / N, top5Sum / N
end

function Trainer:computeScore(output, target, nCrops)
   if nCrops > 1 then
      -- Sum over crops
      output = output:view(output:size(1) / nCrops, nCrops, output:size(2))
      --:exp():sum(2):squeeze(2)
   end

   -- Coputes the top1 and top5 error rate
   local batchSize = output:size(1)

   local _, predictions = output:float():sort(2, true) -- descending

   -- Find which predictions match the target
   local correct = predictions:eq(target:long():view(batchSize, 1):expandAs(output))

   -- Top-1 score
   local top1 = 1.0 - (correct:narrow(2, 1, 1):sum() / batchSize)

   -- Top-5 score, if there are at least 5 classes
   local len = math.min(5, correct:size(2))
   local top5 = 1.0 - (correct:narrow(2, 1, len):sum() / batchSize)

   return top1 * 100, top5 * 100
end

function Trainer:copyInputs(sample)
   -- Copies the input to a CUDA tensor, if using 1 GPU, or to pinned memory,
   -- if using DataParallelTable. The target is always copied to a CUDA tensor
   self.input = self.input or (self.opt.nGPU == 1
           and torch.CudaTensor()
           or cutorch.createCudaHostTensor())
   self.target = self.target or torch.CudaTensor()

   self.input:resize(sample.input:size()):copy(sample.input)
   self.target:resize(sample.target:size()):copy(sample.target)
end

function Trainer:learningRate(epoch)
   -- Training schedule
   local decay = 0
   if self.opt.dataset == 'imagenet' then
      decay = math.floor((epoch - 1) / 30)
   elseif self.opt.dataset == 'cifar10' then
      decay = epoch >= 122 and 2 or epoch >= 81 and 1 or 0
   elseif self.opt.dataset == 'cifar100' then
      decay = epoch >= 122 and 2 or epoch >= 81 and 1 or 0
   end
   return self.opt.LR * math.pow(0.1, decay)
end

return M.Trainer
