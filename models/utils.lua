-- copy from: https://github.com/szagoruyko/wide-residual-networks

local utils = {}

local c = require 'trepl.colorize'

function utils.MSRinit(model)
    for k,v in pairs(model:findModules('nn.SpatialConvolution')) do
        local n = v.kW*v.kH*v.nInputPlane
        v.weight:normal(0,math.sqrt(2/n))
        if v.bias then v.bias:zero() end
    end
    print(c.red '    apply MSR init.')
end

function utils.FCinit(model)
    for k,v in pairs(model:findModules'nn.Linear') do
        v.bias:zero()
    end
end

function utils.DisableBias(model)
    for i,v in ipairs(model:findModules'nn.SpatialConvolution') do
        v.bias = nil
        v.gradBias = nil
    end
end

function utils.testModel(model)
    model:float()
    local imageSize = opt and opt.imageSize or 32
    local input = torch.randn(1,3,imageSize,imageSize):type(model._type)
    print('forward output',{model:forward(input)})
    print('backward output',{model:backward(input,model.output)})
    model:reset()
end

function utils.makeDataParallelTable(model, nGPU)
    if nGPU > 1 then
        local gpus = torch.range(1, nGPU):totable()
        local fastest, benchmark = cudnn.fastest, cudnn.benchmark

        local dpt = nn.DataParallelTable(1, true, true)
        :add(model, gpus)
        :threads(function()
            local cudnn = require 'cudnn'
            cudnn.fastest, cudnn.benchmark = fastest, benchmark
        end)
        dpt.gradInput = nil

        model = dpt:cuda()
    end
    return model
end

function utils.deepcopy(orig)
    local orig_type = type(orig)
    local copy
    if orig_type == 'table' then
        copy = {}
        for orig_key, orig_value in next, orig, nil do
            copy[utils.deepcopy(orig_key)] = utils.deepcopy(orig_value)
        end
        setmetatable(copy, utils.deepcopy(getmetatable(orig)))
    else -- number, string, boolean, etc
    copy = orig
    end
    return copy
end

return utils