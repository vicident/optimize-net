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

return utils
