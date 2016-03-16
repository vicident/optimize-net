local optnet = require 'optnet.env'
local utils = require 'optnet.utils'
local keepTrack = utils.keepTrack

function optnet.countUsedMemory(net, input, opts)
  opts = opts or {}
  local func = opts.func or 'updateOutput'
  net[func](net, input)
  local tensors = {outputs={},buffers={},params={}}
  local function entry_fun(t)
    return t
  end
  local function new_func(m)
    local basefunc = m[func]
    m[func] = function(self, input)
      --keepTrack(input, tensors, entry_fun)
      keepTrack(self.output, tensors.outputs, entry_fun)
      for k, v in pairs(self) do
        if torch.isTensor(v) and
           k ~= 'weight' and k ~= 'bias' and
           k ~= 'gradWeight' and k ~= 'gradBias' and
           k ~= 'output' and k ~= 'gradInput' then
          keepTrack(v, tensors.buffers, entry_fun)
        end
      end
      for _, k in ipairs({'weight', 'bias', 'gradWeight','gradBias'}) do
        if self[k] then
          keepTrack(self[k], tensors.params, entry_fun)
        end
      end
      return basefunc(self, input)
    end
  end
  net:apply(new_func)
  net[func](net, input)
  -- clean up the modified function
  net:apply(function(x)
    x[func] = nil
  end)
  local total_size = 0
  local sizes = {}
  for typeTensor, subTensors in pairs(tensors) do
    sizes[typeTensor] = 0
    for k,v in pairs(subTensors) do
      local size = v:storage():size()*v:elementSize()
      total_size = total_size + size
      sizes[typeTensor] = sizes[typeTensor] + size
    end
  end
  sizes.total_size = total_size
  return sizes
end
