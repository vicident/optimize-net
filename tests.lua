local optnet = require 'optnet.env'
local models = require 'optnet.models'

local countUsedMemory = optnet.countUsedMemory

local optest = torch.TestSuite()
local tester = torch.Tester()

local function genericTestForward(model,opts)
  local net, input = models[model](opts)
  net:evaluate()
  local out_orig = net:forward(input):clone()

  local mems1 = optnet.countUsedMemory(net, input)

  optnet.optimizeMemory(net, input)

  local out = net:forward(input):clone()
  local mems2 = countUsedMemory(net, input)
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

tester:add(optest)

function optnet.test(tests)
  tester:run(tests)
  return tester
end
