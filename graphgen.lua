require 'graph'

-- taken from http://www.graphviz.org/doc/info/colors.html
local colorNames = {
  "aliceblue","antiquewhite","antiquewhite1","antiquewhite2","antiquewhite3",
  "antiquewhite4","aquamarine","aquamarine1","aquamarine2","aquamarine3",
  "aquamarine4","azure","azure1","azure2","azure3",
  "azure4","beige","bisque","bisque1","bisque2",
  "bisque3","bisque4","blanchedalmond","blue",
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


local function generateGraph(net, input, opts)

  local storageHash = {}
  local nodes = {}

  local g = graph.Graph()

  -- basic function for creating an annotated nn.Node to suit our purposes
  -- gives the same color for the same storage.
  -- note that two colors being the same does not imply the same storage
  -- as we have a limited number of colors
  local function createNode(name, tensor)
    local data = torch.pointer(tensor:storage())
    local storageId
    if not storageHash[data] then
      storageHash[data] = torch.random(1,#colorNames)
      table.insert(storageHash, data)
    end
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
      return {
         color=colorNames[storageHash[data]], 
         style = 'filled',
         shape = 'box',
         fontsize = 10,
      }
    end
    return node
  end
  
  -- generate input/output nodes
  local function createBoundaryNode(input, name)
    if torch.isTensor(input) then
      local ptr = torch.pointer(input)
      nodes[ptr] = createNode(name,input)
    else
      for k,v in ipairs(input) do
        createBoundaryNode(nodes, v, name..' '..k)
      end
    end
  end

  -- create edge "from" -> "to", creating "to" on the way with "name"
  -- the edges can be seen as linking modules, but in fact it links the output
  -- tensor of each module
  local function addEdge(from, to, name)
    if torch.isTensor(to) and torch.isTensor(from) then
      local fromPtr = torch.pointer(from)
      local toPtr = torch.pointer(to)

      nodes[toPtr] = nodes[toPtr] or createNode(name,to)
      
      assert(nodes[fromPtr], 'Parent node inexistant for module '.. name)
      
      -- insert edge
      g:add(graph.Edge(nodes[fromPtr],nodes[toPtr]))
      
    elseif torch.isTensor(from) then
      for k,v in ipairs(to) do
        addEdge(from, v, name)
      end
    else
      for k,v in ipairs(from) do
        addEdge(v, to, name)
      end
    end
  end

  -- go over the network keeping track of the input/output for each module
  -- we overwrite the updateOutput for that.
  local function apply_func(m)
    local basefunc = m.updateOutput
    m.updateOutput = function(self, input)
      if not m.modules then
        local name = tostring(m)
        if m.inplace then -- handle it differently ?
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

          local name = torch.typename(last_module)
          nodes[ptr] = nodes[ptr] or createNode(name,out)
          addEdge(out, self.output, torch.typename(m))
        end
      end
      return basefunc(self, input)
    end
  end

  createBoundaryNode(input, 'Input')

  -- fill the states from each tensor
  net:forward(input)
  
  --createInputNode(nodes, net.output, 'Output')
  
  -- overwriting the standard functions to generate our graph
  net:apply(apply_func)
  -- generate the graph
  net:forward(input)

  -- clean up the modified function
  net:apply(function(x)
    x.updateOutput = nil
  end)

  return g
end

return generateGraph

