# Hypergradients with Lua/Torch

## Objectives & Background

[hypergrad](https://github.com/HIPS/hypergrad) uses [HIPS/autograd](https://github.com/HIPS/autograd) to calculate the differentiations. Unfortunately, [HIPS/autograd](https://github.com/HIPS/autograd) does not support GPUs, and will not support it anytime soon. 

We are rewriting [hypergrad](https://github.com/HIPS/hypergrad) using Lua/Torch, using [torch-autograd](https://github.com/twitter/torch-autograd)

## Current Status
Can tune learning rates and L2 norms

## How to run

- `drmad_mnist.lua` is for tuning L2 penalties on MNIST. 
- `cuda_drmad_mnist.lua` is for tuning L2 penalties on MNIST with CUDA. 
- `lr_drmad_mnist.lua` is for tuning learning rates and L2 penalties on MNIST.  

## TODO
1. Experiments on CIFAR-10 and ImageNet
2. ~~Support for learning rates~~
3. ~~Support GPU~~
4. Integration with `hypero`
5. Refactoring

## Rally ([Net2Net](https://github.com/soumith/net2net.torch)) for ImageNet Dataset
ImageNet dataset usually needs ~450,000 iterations. DrMAD may not approxiate this long trajectory well. 

One approach would be to repeatedly initialize the weights using Net2Net, from small subsets to larget subsets and finally to the full dataset. 
