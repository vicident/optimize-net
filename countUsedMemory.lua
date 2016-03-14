local optnet = require 'optnet.env'
local utils = require 'optnet.utils'
local keepTrack = utils.keepTrack

function optnet.countUsedMemory(net, input, opts)
  opts = opts or {}
  local countBuffers = opts.countBuffers or false
  local func = opts.func or 'updateOutput'
  net[func](net, input)
  local tensors = {}
  local function entry_fun(t)
    return t
  end
  local function new_func(m)
    local basefunc = m[func]
    m[func] = function(self, input)
      --keepTrack(input, tensors, entry_fun)
      keepTrack(self.output, tensors, entry_fun)
      if countBuffers then
        for k, v in pairs(self) do
          if torch.isTensor(v) then
            keepTrack(v, tensors, entry_fun)
          end
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
  for k,v in pairs(tensors) do
    local size = v:storage():size()*v:elementSize()
    total_size = total_size + size
  end
  return total_size--/(1024*1024) -- MB
end
