require 'nn'


function usedMemory(net, input, func)
  local func = func or 'updateOutput'
  net[func](net, input)
  local tensors = {}
  local function keepTrack(t)
    if torch.isTensor(t) and t:storage() then
      local ptr = torch.pointer(t:storage())
      if not tensors[ptr] then
        tensors[ptr] = t
      end
      return
    end
    if torch.type(t) == 'table' then
      for k, v in ipairs(t) do
        keepTrack(v)
      end
    end
  end
  local function new_func(m)
    local basefunc = m[func]
    m[func] = function(self, input)
      keepTrack(input)
      keepTrack(self.output)
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
  local analysis = {}
  local analysis2 = {}
  local func = func or 'updateOutput'
  net[func](net, input)
  local c = 1
  local function keepTrack(t, var, c, name, f, notUsed)
    if torch.isTensor(t) and t:storage() then
      local ptr = torch.pointer(t:storage())
      if not analysis[ptr] then
        --analysis[ptr] = {[var]=c, name=name, ptr=ptr, tensor=t}
        analysis[ptr] = {used=kNotUsed,defined=kNotDefined, name=name, ptr=ptr, tensor=t}
        table.insert(analysis2,analysis[ptr])
      end
      local val = analysis[ptr][var]
      if val == notUsed then
        analysis[ptr][var] = c
      else
        analysis[ptr][var] = f(c,val)
      end
      return
    end
    if torch.type(t) == 'table' then
      for k, v in ipairs(t) do
        keepTrack(v, var, c, name, f, notUsed)
      end
    end
  end
  local function new_func(m)
    local basefunc = m[func]
    m[func] = function(self, input)
      --if torch.typename(m) ~= 'nn.Sequential' then
      keepTrack(input, 'used', c, tostring(m), math.max, kNotUsed)
      keepTrack(self.output, 'defined', c, tostring(m), math.min, kNotDefined)
      c = c + 1
      --end
      return basefunc(self,input)
    end
  end
  net:apply(new_func)
  net[func](net, input)
  local function trackInputs(t)
    if torch.isTensor(t) then
      local f = function(a,b) return a end
      keepTrack(t, 'used', kAlwaysLive, 'input', f, 0)
      keepTrack(t, 'defined', -kAlwaysLive, 'input', f, 0)
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
  return analysis2
end

local function isCompatible(candidate, assignment)
  if candidate.used == kNotUsed then
    return false
  end
  if candidate.tensor:numel() < kMinimumForSharing then
    return false
  end
  local a_used = assignment[#assignment].used-- or -1
  return candidate.defined > a_used
end

local function assign(net, analysis)
  table.sort(analysis, function(a,b)
    local x = a.used-- or -1
    local y = b.used-- or -1
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

function optimizeMemory(net, input)
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
  end)
end


