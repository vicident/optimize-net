#OptNet - reducing memory usage in torch neural networks

Memory optimizations for torch neural networks.

Heavily inspired from the `Optimizer` from https://github.com/facebook/fb-caffe-exts

## How does it work ?

It goes over the network and verify which buffers can be reused.
Currently, it only supports evaluation mode, but training mode will soon be included.

Here is a list of currently tested modules (numbers are for CPU version, with batch size of 1, in the format (total memory used, memory used for the outputs)):

| Network | before optimization | after optimization | Relative save |
| ------- | :--------: | :-------: | :------: |
|alexnet | (972MB, 6MB) | (933MB, 1.5MB) | (4%, 75%) |
|vgg16 | (2311MB, 69MB) | (2119MB, 30MB) | (8%, 55%) |
|googlenet | (505MB, 69MB) | (337MB, 30MB) | (33%, 57%) |
|resnet 110 (cifar)| (113MB, 16MB) | (32MB, 4MB) | (72%, 73%) |

Note that most of the used memory goes to the convolution buffers from `nn`.

## Visualizing the memory reuse

We can analyse the sharing of the internal buffers by looking at the computation
graph of the network before and after the sharing.

For that, we have the `graphgen(net, input, opts)` function, which creates the
graph corresponding to the network `net`. The generated graph contains the storage
id of each `output`, and same colors means same storage.

Note that `net` is a `nn` model, and **not** a `nngraph` network. This allows us
to use `optnet.graphgen` to generate graph visualizations of `nn` networks without
having to use `nngraph`.

Let's have a look:

```lua
-- some handy models are defined in optnet.models
-- line alexnet, googlenet and resnet
models = require 'optnet.models'
modelname = 'googlenet'
net, input = models[modelname]()

generateGraph = require 'optnet.graphgen'

g = generateGraph(net, input)

graph.dot(g,modelname,modelname)
```

This generates the following graph:

![GoogleNet without memory optimization](doc/googlenet.gif)

Now what happens after we optimize the network ? Check the colors and the storage
ids.

```lua
models = require 'optnet.models'
modelname = 'googlenet'
net, input = models[modelname]()

opts = {inplace=true, reuseBuffers=true}

generateGraph = require 'optnet.graphgen'

optnet = require 'optnet'

optnet.optimizeMemory(net, input, opts)

g = generateGraph(net, input)

graph.dot(g,modelname..'_optimized',modelname..'_optimized')
```
![GoogleNet with memory optimization](doc/googlenet_optimized.gif)

## Counting the amount of saved memory

We can also provide a function to compute the amount of memory used by the network
in bytes, which allows us to check the amount of saved memory. It currently
counts only the `output` fields of each module, and not it's internal buffers.

Here is an example

```lua
optnet = require 'optnet'

models = require 'optnet.models'
modelname = 'googlenet'
net, input = models[modelname]()

opts = {countBuffers=true}

mem1 = optnet.countUsedMemory(net, input, opts)

optnet.optimizeMemory(net, input)

mem2 = optnet.countUsedMemory(net, input, opts)

optnet.removeOptimization(net)

mem3 = optnet.countUsedMemory(net, input, opts)

print('Before optimization        : '.. mem1/1024/1024 .. ' MBytes')
print('After optimization         : '.. mem2/1024/1024 .. ' MBytes')
print('After removing optimization: '.. mem3/1024/1024 .. ' MBytes')

```
