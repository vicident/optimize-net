require 'nn'

local optnet = require 'optnet.env'
require 'optnet.countUsedMemory'
require 'optnet.tests'

local utils = require 'optnet.utils'

local kNotUsed = 10000---1
local kNotDefined = 0
local kMinimumForSharing = 2
local kAlwaysLive = 10000

local function analyse(net, input, func)
  local func = func or 'updateOutput'

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
    m[func] = function(self, input)
      local opts = {
        analysis=analysis, c=c, name=tostring(m),
        kNotUsed=kNotUsed, kNotDefined=kNotDefined
      }

      -- always keep track of the input
      opts.var = 'used'; opts.f = math.max; opts.notUsed = kNotUsed
      utils.keepTrack(input, track, entry_fun, fun, opts)

      if not m.modules then
        -- always keep track of the outputs of non-containers
        opts.var = 'defined'; opts.f = math.min; opts.notUsed = kNotDefined
        utils.keepTrack(self.output, track, entry_fun, fun, opts)
      elseif torch.typename(m) == 'nn.Concat' or
        torch.typename(m) == 'nn.Parallel' or
        torch.typename(m) == 'nn.DepthConcat' then

        -- for containers that do some operations on the input, need to keep
        -- track of each output of its branches uppon entry on the module,
        -- as well as to keep track of it's own output (as it's a non-trivial
        -- operation on the childs output, contrary to nn.Sequential for
        -- example)
        opts.var = 'defined'; opts.f = math.min; opts.notUsed = kNotDefined
        utils.keepTrack(self.output, track, entry_fun, fun, opts)

        for i,branch in ipairs(m.modules) do
          local last_module = branch:get(branch:size())
          local out = last_module.output
          opts.var = 'defined'; opts.f = math.min; opts.notUsed = kNotDefined
          utils.keepTrack(out, track, entry_fun, fun, opts)
        end
      end
      c = c + 1
      return basefunc(self,input)
    end
  end
  net:apply(apply_func)
  net[func](net, input, grad)
  local function trackInputs(t)
    if torch.isTensor(t) then
      local f = function(a,b) return a end
      utils.keepTrack(t, track, entry_fun, fun,
        {var='used', c=kAlwaysLive,
         f=f, notUsed=0, name='input'})
      utils.keepTrack(t, track, entry_fun, fun,
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

  -- disable backward pass if in evaluation mode
  if func == 'updateOutput' then
    net:apply(function(m)
      m.updateGradInput = function(self, input, gradInput)
        error([[Backward pass disabled!
          You are using inference optimization.
          Call optnet.removeOptimization(net) to enable backward again]])
      end
    end)
  end
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

local function defaultValue(var, val)
  if var == nil then
    var = val
  end
  return var
end

-- set to inplace modules that allows it
local function setInplace(net, opts)
  local inplace = defaultValue(opts.inplace, true)
 
  if inplace then
    net:apply(function(m)
      if m.inplace ~= nil then
        -- inplace is not always supported for threshold,
        -- depending on the values. Disabling it for the moment
        if torch.typename(m) ~= 'nn.Threshold' then
          m.inplace = true
        end
      end
    end)
  end
end

local reusableBuffers = {
['nn.SpatialConvolution'] = {{'finput','fgradInput'},{}},
['nn.SpatialConvolutionMM'] = {{'finput','fgradInput'},{}},
['nn.Normalize'] = {{'norm','buffer','normp','_indices'},{}},
['nn.SpatialCrossMapLRN'] = {{'scale'},{}},
['nn.SpatialMaxPooling'] = {{'indices'},{}},
}
-- basic reusing scheme: keeps a list of all possible buffers
-- that can be reused in evaluation mode and also in training
-- mode.
local function reuseStateBuffers(net, opts)
  local reuseBuffers = defaultValue(opts.reuseBuffers, true)
  if reuseBuffers then
    local reusedBuffers = {}
    net:apply(function(m)
      local name = torch.typename(m)
      if reusableBuffers[name] then
        local rb = reusableBuffers[name][1]
        for k, v in ipairs(rb) do
          if m[v] then
            reusedBuffers[name..','..v] = reusedBuffers[name..','..v] or m[v]:storage()
            if reusedBuffers[name..','..v] then
              m[v]:set(reusedBuffers[name..','..v])
            end
          end
        end
      end
    end)
  end
end

-- needed for cudnn
local function resetInputDescriptors(net)
  net:apply(function(m)
    if torch.typename(m):find('cudnn') and
       torch.typename(m.iSize) == 'torch.LongStorage' then
      m.iSize:fill(0)
    end
  end)
end

function optnet.optimizeMemory(net, input, opts)
  opts = opts or {}
  local func = defaultValue(opts.func,'forward')

  local grad
  if func == 'backward' then
    -- need to run forward before backward
    grad = net['forward'](net, input)
  end
  -- do a pass over the network to initialize its fields
  net[func](net, input, grad)

  setInplace(net, opts)
  reuseStateBuffers(net, opts)

  -- share outputs
  local analysis = analyse(net, input)
  local assignments = assign(net,analysis)
  applyAssignments(net, assignments)
  resetInputDescriptors(net)
end

function optnet.removeOptimization(net)
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
    local name = torch.typename(m)
    if reusableBuffers[name] then
      local rb = reusableBuffers[name][1]
      for k, v in ipairs(rb) do
        if m[v] then
          m[v]:set()
        end
      end
    end

    resetInputDescriptors(net)
    -- remove backward blocking
    m.updateGradInput = nil
  end)
end

return optnet

