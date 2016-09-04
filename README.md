#OptNet - reducing memory usage in torch neural networks

Memory optimizations for torch neural networks.

Heavily inspired from the `Optimizer` from https://github.com/facebook/fb-caffe-exts

## Installing
Simply do
```
luarocks install optnet
```

## How does it work ?

It goes over the network and verify which buffers can be reused.
It supports both inference (evaluation) mode and training mode.

### Inference mode

Here is a list of currently tested modules. Numbers are for CPU version, with batch size of 1, for **double** type, in the format
**(total memory used, memory used for the outputs, memory used for the internal buffers, memory used for the parameters and grad parameters)**:

| Network | before optimization | after optimization | Relative save |
| ------- | :--------: | :-------: | :------: |
|alexnet | (973MB, 6MB, 43MB, 924MB) | (472MB, 1.5MB, 9MB, 462MB) | (51%, 75%, 80%, 50%) |
|vgg-A | (2311MB, 69MB, 215MB, 2027MB) | (1106MB, 31MB, 61MB, 1014MB) | (52%, 55%, 72%, 50%) |
|googlenet | (505MB, 69MB, 145MB, 292MB) | (193MB, 31MB, 16MB, 146MB) | (62%, 54%, 89%, 50%) |
|resnet 110 (cifar)| (113MB, 16MB, 71MB, 26MB) | (15MB, 0.5MB, 1.3MB, 13MB) | (87%, 97%, 98%, 50%) |

Note that for most of the models, for a batch size of 1 most of the memory is spent with the `weights` and `gradWeights` of the network (and the latter can be safely freed during inference).
More interestingly, the the output size is *linearly* dependent on the batch size, which means that the total savings are much more significant for bigger batch sizes.

In a more realistic setup where we use `cudnn` and batch size of 128, the gains are
way more significant, specially for very deep networks like resnet. The memory usage is shown in the following table (for **float** type), following **(total memory used, memory used for the outputs, memory used for the parameters and grad parameters)** as `cudnn` almost don't use internal buffers:

| Network | before optimization | after optimization | Relative save |
| ------- | :--------: | :-------: | :------: |
|alexnet | (859MB, 397MB, 462MB) | (328MB, 97MB, 231MB) | (62%, 75%, 50%) |
|vgg-A | (5340MB, 4386MB, 1014MB) | (2467MB, 1960MB, 507MB) | (54%, 55%, 50%) |
|googlenet | (4536MB, 4390MB, 146MB) | (2066MB, 1993MB, 73MB) | (54%, 55%, 50%) |
|resnet 110 (cifar)| (1049MB, 1036MB, 13MB) | (39MB, 32MB, 7MB) | (96%, 97%, 50%) |

### Training mode

We currently support a basic algorithm for training mode.
Using `cudnn` with batch size of 64, we currently obtain the following savings, in the format **(total memory used, memory used for the outputs, memory used for the gradInputs, memory used for the parameters and gradParameters)**:

| Network | before optimization | after optimization | Relative save |
| ------- | :--------: | :-------: | :------: |
|alexnet | (963MB, 195MB, 303MB, 462MB) | (816MB, 195MB, 156MB, 462MB) | (15%, 0%, 48%, 0%) |
|vgg-A | (5433MB, 2191MB, 2228MB, 1014MB) | (4228MB, 2191MB, 1023MB, 1014MB) | (22%, 0%, 54%, 0%) |
|googlenet | (6092MB, 2195MB, 3346MB, 146MB) | (4844MB, 2195MB, 2098MB, 146MB) | (20%, 0%, 37%, 0%) |
|resnet 110 (cifar)| (664MB, 259MB, 392MB, 13MB) | (428MB, 259MB, 156MB, 13MB) | (36%, 0%, 60%, 0%) |

Note that the relative save of the `gradInput` stays constant for different batch sizes, meaning that the total relative savings will be more important for bigger batch sizes (as the parameters doesn't depend on the batch size).

We can setup the optimizations for training mode by using `mode='training'` as follows

```lua
models = require 'optnet.models'
modelname = 'googlenet'
net, input = models[modelname]()

opts = {inplace=true, mode='training'}

optnet = require 'optnet'

optnet.optimizeMemory(net, input, opts)
```

### Optional parameters

Here is a list of options that are currently supported, and should be passed in the `opts` table as a third argument:
* `inplace`: uses in place modules when available (boolean)
* `mode`: selects between `training` and `inference` optimization algorithm (string)
* `reuseBuffers`: shares internal buffers between same modules (like unfolded images for convolution) (boolean)
* `removeGradParams`: remove `gradWeight` and `gradBias` in the networks, saving their sharings so that they can be exactly reconstructed. Only applies for `inference` mode (boolean)

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
-- like alexnet, googlenet, vgg and resnet
models = require 'optnet.models'
modelname = 'googlenet'
net, input = models[modelname]()

generateGraph = require 'optnet.graphgen'

-- visual properties of the generated graph
-- follows graphviz attributes
graphOpts = {
displayProps =  {shape='ellipse',fontsize=14, style='solid'},
nodeData = function(oldData, tensor)
  return oldData .. '\n' .. 'Size: '.. tensor:numel()
end
}

g = generateGraph(net, input, graphOpts)

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

graphOpts = {
displayProps =  {shape='ellipse',fontsize=14, style='solid'},
nodeData = function(oldData, tensor)
  return oldData .. '\n' .. 'Size: '.. tensor:numel()
end
}

g = generateGraph(net, input, graphOpts)

graph.dot(g,modelname..'_optimized',modelname..'_optimized')
```
![GoogleNet with memory optimization](doc/googlenet_optimized.gif)

## Counting the amount of saved memory

We can also provide a function to compute the amount of memory used by the network
in bytes, which allows us to check the amount of saved memory.
It decomposes the total amount of memory in four fields:
* total memory used by the outputs of each module
* total memory used by the gradInputs of each module
* total memory used by the internal buffers of each module
* total memory used by the weights and gradWeights of each module.

Here is an example

```lua
optnet = require 'optnet'

models = require 'optnet.models'
modelname = 'googlenet'
net, input = models[modelname]()

mem1 = optnet.countUsedMemory(net)

optnet.optimizeMemory(net, input)

mem2 = optnet.countUsedMemory(net)

optnet.removeOptimization(net)

mem3 = optnet.countUsedMemory(net)

print('Before optimization        : '.. mem1.total_size/1024/1024 .. ' MBytes')
print('After optimization         : '.. mem2.total_size/1024/1024 .. ' MBytes')
print('After removing optimization: '.. mem3.total_size/1024/1024 .. ' MBytes')

```
