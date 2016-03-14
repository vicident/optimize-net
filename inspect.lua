require 'graph'
require 'trepl'

models = dofile 'models.lua'

--net,input = models.resnet({dataset='cifar10',depth=20})
--net,input = models.alexnet()
--net,input = models.basic1()
net,input = models.googlenet()

-- taken from http://www.graphviz.org/doc/info/colors.html
local colorNames = {
  "aliceblue","antiquewhite","antiquewhite1","antiquewhite2","antiquewhite3",
  "antiquewhite4","aquamarine","aquamarine1","aquamarine2","aquamarine3",
  "aquamarine4","azure","azure1","azure2","azure3",
  "azure4","beige","bisque","bisque1","bisque2",
  "bisque3","bisque4","black","blanchedalmond","blue",
  "blue1","blue2","blue3","blue4","blueviolet",
  "brown","brown1","brown2","brown3","brown4",
  "burlywood","burlywood1","burlywood2","burlywood3","burlywood4",
  "cadetblue","cadetblue1","cadetblue2","cadetblue3","cadetblue4",
  "chartreuse","chartreuse1","chartreuse2","chartreuse3","chartreuse4",
  "chocolate","chocolate1","chocolate2","chocolate3","chocolate4",
  "coral","coral1","coral2","coral3","coral4",
  "cornflowerblue","cornsilk","cornsilk1","cornsilk2","cornsilk3",
  "cornsilk4","crimson","cyan","cyan1","cyan2",
  "cyan3","cyan4","darkgoldenrod","darkgoldenrod1","darkgoldenrod2",
  "darkgoldenrod3","darkgoldenrod4","darkgreen","darkkhaki","darkolivegreen",
  "darkolivegreen1","darkolivegreen2","darkolivegreen3","darkolivegreen4","darkorange",
  "darkorange1","darkorange2","darkorange3","darkorange4","darkorchid",
  "darkorchid1","darkorchid2","darkorchid3","darkorchid4","darksalmon",
  "darkseagreen","darkseagreen1","darkseagreen2","darkseagreen3","darkseagreen4",
  "darkslateblue","darkslategray","darkslategray1","darkslategray2","darkslategray3",
  "darkslategray4","darkslategrey","darkturquoise","darkviolet","deeppink",
  "deeppink1","deeppink2","deeppink3","deeppink4","deepskyblue",
  "deepskyblue1","deepskyblue2","deepskyblue3","deepskyblue4","dimgray",
  "dimgrey","dodgerblue","dodgerblue1","dodgerblue2","dodgerblue3",
  "dodgerblue4","firebrick","firebrick1","firebrick2","firebrick3",
  "firebrick4","floralwhite","forestgreen","gainsboro","ghostwhite",
  "gold","gold1","gold2","gold3","gold4",
  "goldenrod","goldenrod1","goldenrod2","goldenrod3","goldenrod4"
}


local storageHash = {}

local nodes = {}

local function createNode(name, data)
  local storageId
  for k, v in ipairs(storageHash) do
    if v == data then
      storageId = k
    end
  end
  local node = graph.Node("Storage id: "..storageId)
  function node:graphNodeName()
    return name
  end
  function node:graphNodeAttributes()
    return {color=colorNames[storageHash[data]]}
  end
  return node
end

local function createInputNode(input, name)
  name = name or 'Input'
  if torch.isTensor(input) then
    local ptr = torch.pointer(input)
    local storagePtr = torch.pointer(input:storage())
    if not storageHash[storagePtr] then
      storageHash[storagePtr] = torch.random(1,#colorNames)
      table.insert(storageHash, storagePtr)
    end
    nodes[ptr] = createNode(name,storagePtr)
  else
    for k,v in ipairs(input) do
      createInputNode(nodes, v, name..' '..k)
    end
  end
end

createInputNode(input)

local g = graph.Graph()

local function addEdge(p,c,name)
  if torch.isTensor(c) and torch.isTensor(p) then
    local cc = torch.pointer(c)
    local childStoragePtr = torch.pointer(c:storage())
    if not storageHash[childStoragePtr] then
      storageHash[childStoragePtr] = torch.random(1,#colorNames)
      table.insert(storageHash, childStoragePtr)
    end
    local pp = torch.pointer(p)
    nodes[cc] = nodes[cc] or createNode(name,childStoragePtr)
    local parent = nodes[pp]
    assert(parent, 'Parent node inexistant for module '.. name)
    g:add(graph.Edge(parent,nodes[cc]))
  elseif torch.isTensor(p) then
    for k,v in ipairs(c) do
      addEdge(p,v,name)
    end
  else
    for k,v in ipairs(p) do
      addEdge(v,c,name)
    end
  end
end

local function apply_func(m)
  local oldf = m.updateOutput
  m.updateOutput = function(self, input)
    if not m.modules then
      local name = tostring(m)
      if m.inplace then -- handle it differently to avoid loops ?
        addEdge(input,self.output,name)
      else
        addEdge(input,self.output,name)
      end
    elseif torch.typename(m) == 'nn.Concat' or 
      torch.typename(m) == 'nn.Parallel' or 
      torch.typename(m) == 'nn.DepthConcat' then
      -- those containers effectively do some computation, so they have their
      -- place in the graph
      for i,branch in ipairs(m.modules) do
        local last_module = branch:get(branch:size())
        local out = last_module.output
        local ptr = torch.pointer(out)
        local storagePtr = torch.pointer(out:storage())
        if not storageHash[storagePtr] then
          storageHash[storagePtr] = torch.random(1,#colorNames)
          table.insert(storageHash, storagePtr)
        end
        local name = torch.typename(last_module)
        nodes[ptr] = nodes[ptr] or createNode(name,storagePtr)
        addEdge(out, self.output, torch.typename(m))
      end
    end
    return oldf(self, input)
  end
end




net:forward(input)
--createInputNode(nodes, net.output, 'Output')
net:apply(apply_func)
net:forward(input)


graph.dot(g)

