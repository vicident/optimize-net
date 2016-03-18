local optnet = require 'optnet.env'
local models = require 'optnet.models'

local use_cudnn = true

if use_cudnn then
  require 'cudnn'
  require 'cunn'
end

local countUsedMemory = optnet.countUsedMemory

local optest = torch.TestSuite()
local tester = torch.Tester()

local backward_tol = 1e-6

local function resizeAndConvert(input, type)
  local res
  local s = 64
  if torch.isTensor(input) then
    local iSize = torch.Tensor(input:size():totable())[{{2,-1}}]
    res = torch.rand(s,table.unpack(iSize:totable())):type(type)
  else
    res = {}
    for k, v in ipairs(input) do
      res[k] = resizeAndConvert(v,type)
    end
  end
  return res
end

local function cudnnSetDeterministic(net)
  net:apply(function(m)
    if m.setMode then m:setMode(1, 1, 1) end
  end)
end

local function genericTestForward(model,opts)
  local net, input = models[model](opts)
  net:evaluate()

  if use_cudnn then
    cudnn.convert(net,cudnn);
    net:cuda();

    input = resizeAndConvert(input,'torch.CudaTensor')
  end

  local out_orig = net:forward(input):clone()

  local mems1 = optnet.countUsedMemory(net)

  optnet.optimizeMemory(net, input)

  local out = net:forward(input):clone()
  local mems2 = countUsedMemory(net)
  tester:eq(out_orig, out, 'Outputs differ after optimization of '..model)

  local mem1 = mems1.total_size
  local mem2 = mems2.total_size

  local omem1 = mems1.outputs
  local omem2 = mems2.outputs

  local bmem1 = mems1.buffers
  local bmem2 = mems2.buffers

  local pmem1 = mems1.params
  local pmem2 = mems2.params

  tester:assertle(mem2, mem1, 'Optimized model uses more memory! '..
  'Before: '.. mem1..' bytes, After: '..mem2..' bytes')
  print('Memory use')
  print('Total',  mem1/1024/1024,  mem2/1024/1024, 1-mem2/mem1)
  print('Outputs',omem1/1024/1024,omem2/1024/1024, 1-omem2/omem1)
  print('Buffers',bmem1/1024/1024,bmem2/1024/1024, 1-bmem2/bmem1)
  print('Params', pmem1/1024/1024,pmem2/1024/1024, 1-pmem2/pmem1)
end

function optest.basic()
  genericTestForward('basic1')
end

function optest.basic_conv()
  genericTestForward('basic2')
end

function optest.basic_concat()
  genericTestForward('basic_concat')
end

function optest.alexnet()
  genericTestForward('alexnet')
end

function optest.googlenet()
  genericTestForward('googlenet')
end

function optest.vgg()
  genericTestForward('vgg')
end

function optest.resnet20()
  local opts = {dataset='cifar10',depth=20}
  genericTestForward('resnet', opts)
end

function optest.resnet32()
  local opts = {dataset='cifar10',depth=32}
  genericTestForward('resnet', opts)
end

function optest.resnet56()
  local opts = {dataset='cifar10',depth=56}
  genericTestForward('resnet', opts)
end

function optest.resnet110()
  local opts = {dataset='cifar10',depth=110}
  genericTestForward('resnet', opts)
end


-------------------------------------------------
-- Backward
-------------------------------------------------

-- reuse this function
local function recursiveClone(out)
  if torch.isTensor(out) then
    return out:clone()
  else
    local res = {}
    for k, v in ipairs(out) do
      res[k] = recursiveClone(v)
    end
    return res
  end
end


local function genericTestBackward(model,opts)
  local net, input = models[model](opts)
  net:training()

  if use_cudnn then
    cudnn.convert(net,cudnn);
    cudnnSetDeterministic(net)
    net:cuda();

    input = resizeAndConvert(input,'torch.CudaTensor')
  end

  local out_orig = recursiveClone(net:forward(input))
  local grad_orig = recursiveClone(out_orig)
  net:zeroGradParameters()
  local gradInput_orig = recursiveClone(net:backward(input, grad_orig))
  local _, gradParams_orig = net:getParameters()
  gradParams_orig = gradParams_orig:clone()

  local mems1 = optnet.countUsedMemory(net)

  optnet.optimizeMemory(net, input, {mode='training'})

  local out = recursiveClone(net:forward(input))
  local grad = recursiveClone(out)
  net:zeroGradParameters()
  local gradInput = recursiveClone(net:backward(input, grad))
  local _, gradParams = net:getParameters()
  gradParams = gradParams:clone()

  local mems2 = countUsedMemory(net)
  tester:eq(out_orig, out, 'Outputs differ after optimization of '..model)
  tester:eq(gradInput_orig, gradInput, backward_tol, 'GradInputs differ after optimization of '..model)
  tester:eq(gradParams_orig, gradParams, backward_tol, 'GradParams differ after optimization of '..model)

  local mem1 = mems1.total_size
  local mem2 = mems2.total_size

  local omem1 = mems1.outputs
  local omem2 = mems2.outputs

  local imem1 = mems1.gradInputs
  local imem2 = mems2.gradInputs

  local bmem1 = mems1.buffers
  local bmem2 = mems2.buffers

  local pmem1 = mems1.params
  local pmem2 = mems2.params

  tester:assertle(mem2, mem1, 'Optimized model uses more memory! '..
  'Before: '.. mem1..' bytes, After: '..mem2..' bytes')
  print('Memory use')
  print('Total',  mem1/1024/1024,  mem2/1024/1024, 1-mem2/mem1)
  print('Outputs',omem1/1024/1024,omem2/1024/1024, 1-omem2/omem1)
  print('gradInputs',imem1/1024/1024,imem2/1024/1024, 1-imem2/imem1)
  print('Buffers',bmem1/1024/1024,bmem2/1024/1024, 1-bmem2/bmem1)
  print('Params', pmem1/1024/1024,pmem2/1024/1024, 1-pmem2/pmem1)
end

function optest.basic_backward()
  genericTestBackward('basic1')
end

function optest.basic_conv_backward()
  genericTestBackward('basic2')
end

function optest.basic_conv2_backward()
  genericTestBackward('basic3')
end

function optest.basic_concat_backward()
  genericTestBackward('basic_concat')
end

function optest.alexnet_backward()
  genericTestBackward('alexnet')
end

function optest.vgg_backward()
  genericTestBackward('vgg')
end

function optest.googlenet_backward()
  genericTestBackward('googlenet')
end

function optest.resnet20_backward()
  local opts = {dataset='cifar10',depth=20}
  genericTestBackward('resnet', opts)
end

function optest.resnet32_backward()
  local opts = {dataset='cifar10',depth=32}
  genericTestBackward('resnet', opts)
end

function optest.resnet56_backward()
  local opts = {dataset='cifar10',depth=56}
  genericTestBackward('resnet', opts)
end

function optest.resnet110_backward()
  local opts = {dataset='cifar10',depth=110}
  genericTestBackward('resnet', opts)
end

tester:add(optest)

function optnet.test(tests)
  tester:run(tests)
  return tester
end
