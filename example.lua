optnet = require 'optnet'
generateGraph = require 'optnet.graphgen'
models = require 'optnet.models'

modelname = 'googlenet'
net, input = models[modelname]()

g = generateGraph(net, input)
graph.dot(g, modelname, modelname)

optnet.optimizeMemory(net, input)

g = generateGraph(net, input)
graph.dot(g, modelname..'_optimized', modelname..'_optimized')
