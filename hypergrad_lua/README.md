# Hypergradients with Lua/Torch

## Objectives & Background

[hypergrad](https://github.com/HIPS/hypergrad) uses [HIPS/autograd](https://github.com/HIPS/autograd) to calculate the differentiations. Unfortunately, [HIPS/autograd](https://github.com/HIPS/autograd) does not support GPUs, and will not support it anytime soon. 

We are to rewrite [hypergrad](https://github.com/HIPS/hypergrad) using Lua/Torch, using [torch-autograd](https://github.com/twitter/torch-autograd)

## TODO
1. Understand what is `automatic differentiation`. Read: [Automatic differentiation in machine learning: a survey](http://arxiv.org/abs/1502.05767)
2. Read: [Gradient-based Hyperparameter Optimization through Reversible Learning](http://arxiv.org/abs/1502.03492)
3. Converting the code of [hypergrad](https://github.com/HIPS/hypergrad) into Lua/Torch

## Lua/Torch resources & tutorials
* [Torch cheatsheet](https://github.com/torch/torch7/wiki/Cheatsheet)
* [torch-autograd](https://github.com/twitter/torch-autograd)