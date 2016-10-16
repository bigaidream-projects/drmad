--
-- Created by IntelliJ IDEA.
-- User: taineleau
-- Date: 9/19/16
-- Time: 23:20
-- Follow the style of optim.
-- DrMAD is a optimizer can do hyper-parameter optimization
--

--[[ A plain implementation of SGD
ARGS:
- `opfunc` : a function that takes a single input (X), the point
             of a evaluation, and returns f(X) and df/dX
- `x`      : the initial point
- `config` : a table with configuration parameters for the optimizer
- `config.forward_only`      : Ture if in the forward_mode
- `config.learningRate`      : learning rate
- `config.learningRateDecay` : learning rate decay
- `config.weightDecay`       : weight decay
- `config.weightDecays`      : vector of individual weight decays
- `config.momentum`          : momentum
- `config.dampening`         : dampening for momentum
- `config.nesterov`          : enables Nesterov momentum
- `config.learningRates`     : vector of individual learning rates
- `config.forward_only`      : wether conduct a hyper-gradient or not
- `state`  : a table describing the state of the optimizer; after each
             call the state is modified
- `state.evalCounter`        : evaluation counter (optional: 0, by default)
- `state.hyper`              :
RETURN:
- `x`     : the new x vector
- `f(x)`  : the function, evaluated before the update
(Clement Farabet, 2012)
]]

local DrMAD = torch.class('DrMAD')

function DrMAD:sgd(opfunc, x, config, state)
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
   local forward_only = config.forward_only or true
   state.evalCounter = state.evalCounter or 0
   local nevals = state.evalCounter
   assert(not nesterov or (mom > 0 and damp == 0), "Nesterov momentum requires a momentum and zero dampening")


   -- (1) evaluate f(x) and df/dx
   local fx, dfdx = opfunc(x)

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
         for i = 1, #dfdx do
            dfdx[i]:add(mom, state.dfdx[i])
         end
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

   -- In AutoDiff, `forward` means the usual training procedure,
   -- `backward` means the procedure that calculating hyper-gradients.
   -- DrMAD postulates weights to be (t/N) * initial weight + (1 - 1/N) * final weight.
   -- and get everything about dW. (dW)^2, dWdA.
   -- DrMAD.sgd() should not know how to calc the hessian.

   if forward_only == true then
      return x, {fx}
   end

   --------------------
   --- backward begin!
   --------------------
   -- refer: https://github.com/HIPS/hypergrad/blob/master/hypergrad/optimizers.py#L229

--   local d_x = dfdx
--   local h = state.hyper
--   local function grad_proj(x, y) return torch.mul(x, y) end
--   local l_hvp_x, l_hvp_meta = hessianopfunc(all_params, x)
--   h.d_alpha[h.i] = tablemul(d_x, v)
--   -- do not need to recover w
--   d_v = d_v + (alpha * d_x)
--   d_betas[i] =
--   return x, {fx}
end


function DrMAD:sgd_backward()
   --------------------
   --- backward begin!
   --------------------
   -- refer: https://github.com/HIPS/hypergrad/blob/master/hypergrad/optimizers.py#L229

end