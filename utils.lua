local utils = {}

local function keepTrack(t, track, entry_fun, fun, ...)
  if torch.isTensor(t) and t:storage() then
    local ptr = torch.pointer(t:storage())
    if not track[ptr] then
      track[ptr] = entry_fun(t, ...)
    end
    if fun then
      fun(t,track,...)
    end
    return
  end
  if torch.type(t) == 'table' then
    for k, v in ipairs(t) do
      keepTrack(v, track, entry_fun, fun, ...)
    end
  end
end
utils.keepTrack = keepTrack

function utils.usedMemory(net, input, func)
  local func = func or 'updateOutput'
  net[func](net, input)
  local tensors = {}
  local function entry_fun(t)
    return t
  end
  local function new_func(m)
    local basefunc = m[func]
    m[func] = function(self, input)
      keepTrack(input, tensors, entry_fun)
      keepTrack(self.output, tensors, entry_fun)
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

return utils
