require 'nn'

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

function usedMemory(net, input, func)
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


local kNotUsed = 10000---1
local kNotDefined = 0
local kMinimumForSharing = 2
local kAlwaysLive = 10000

local function analyse(net, input, func)
  local func = func or 'updateOutput'
  local grad
  if func == 'backward' then
    -- need to run forward before backward
    grad = net['forward'](net, input)
  end
  -- do a pass over the network to initialize its fields
  net[func](net, input, grad)

  local track = {}
  local analysis = {}

  local function entry_fun(t, args)
    local ptr = torch.pointer(t:storage())
    local info = {used=kNotUsed, defined=kNotDefined,
                  name=args.name, ptr=ptr, tensor=t}
    table.insert(analysis, info)
    return info
  end

  local function fun(t, track, args)
    local ptr = torch.pointer(t:storage())
    local val = track[ptr][args.var]
    if val == args.notUsed then
      track[ptr][args.var] = args.c
    else
      track[ptr][args.var] = args.f(args.c,val)
    end
  end

  local c = 1
  local function apply_func(m)
    local basefunc = m[func]
    local base_opts = {
      analysis=analysis, c=c, name=tostring(m),
      kNotUsed=kNotUsed, kNotDefined=kNotDefined
    }
    m[func] = function(self, input)
      --local opts = {}; for k, v in pairs(base_opts) do opts[k] = v; end
      --opts.var = 'used'; opts.f = math.max; opts.notUsed = kNotUsed
      keepTrack(input, track, entry_fun, fun,-- opts)--[[
            {var='used', c=c, f=math.max,
             notUsed=kNotUsed, name=tostring(m)})--]]

      --opts = {}; for k, v in pairs(base_opts) do opts[k] = v; end
      --opts.var = 'defined'; opts.f = math.min; opts.notUsed = kNotDefined
      keepTrack(self.output, track, entry_fun, fun,-- opts)--[[
            {var='defined',c=c, f=math.min,
             notUsed=kNotDefined, name=tostring(m)})--]]
      c = c + 1
      return basefunc(self,input)
    end
  end
  net:apply(apply_func)
  net[func](net, input, grad)
  local function trackInputs(t)
    if torch.isTensor(t) then
      local f = function(a,b) return a end
      keepTrack(t, track, entry_fun, fun,
        {var='used', c=kAlwaysLive,
         f=f, notUsed=0, name='input'})
      keepTrack(t, track, entry_fun, fun,
        {var='defined', c=-kAlwaysLive,
         f=f, notUsed=0, name='input'})
    else
      for k,v in ipairs(t) do
        trackInputs(v)
      end
    end
  end
  trackInputs(input)
  -- clean up the modified function
  net:apply(function(x)
    x[func] = nil
  end)
  return analysis
end

local function isCompatible(candidate, assignment)
  if candidate.used == kNotUsed then
    return false
  end
  if candidate.tensor:numel() < kMinimumForSharing then
    return false
  end
  local a_used = assignment[#assignment].used
  return candidate.defined > a_used
end

local function assign(net, analysis)
  table.sort(analysis, function(a,b)
    local x = a.used
    local y = b.used
    return x < y
  end)
  local assignments = {}
  for _,candidate in ipairs(analysis) do
    local assigned = false
    for _, assignment in ipairs(assignments) do
      if isCompatible(candidate, assignment) then
        table.insert(assignment,candidate)
        assigned = true
        break
      end
    end
    if not assigned then
      table.insert(assignments, {candidate})
    end
  end
  return assignments
end

local function applyAssignments(net, assignments)
  for _, assignment in ipairs(assignments) do
    local storage
    for k, v in ipairs(assignment) do
      if v.used == kAlwaysLive and v.defined == -kAlwaysLive then
        break
      end
      storage = storage or v.tensor.new(1):storage()
      v.tensor:set(storage)
    end
  end
end

function optimizeMemory(net, input, opts)
  local analysis = analyse(net, input)
--  print('Analysis')
--  print(analysis)
  local assignments = assign(net,analysis)
--  print('Assignments')
--  print(assignments)
  applyAssignments(net, assignments)
end

function removeOptimization(net)
  local function rem(m)
    if torch.isTensor(m) then
      m:set()
    end
    if torch.type(m) == 'table' then
      for k, v in ipairs(m) do
        rem(v)
      end
    end
  end
  
  net:apply(function(m)
    rem(m.output)
    rem(m.gradInput)
  end)
end


