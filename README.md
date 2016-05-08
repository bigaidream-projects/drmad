# DrMAD: Distilling Reverse-Mode Automatic Differentiation for Optimizing Hyperparameters of Deep Neural Networks

![](https://github.com/bigaidream-projects/drmad/blob/master/shortcut.jpg)

[![Gitter](https://badges.gitter.im/Join Chat.svg)](https://gitter.im/bigaidream/drmad?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)
[![License](http://img.shields.io/badge/license-MIT-brightgreen.svg?style=flat)](LICENSE.md)
[![ZenHub] (https://raw.githubusercontent.com/ZenHubIO/support/master/zenhub-badge.png)](https://zenhub.io)

> Source code for http://arxiv.org/abs/1601.00917
> Accepted by IJCAI-2016

## Abstract

The performance of deep neural networks is well-known to be sensitive to the setting of their hyperparameters. Recent advances in reverse-mode automatic differentiation allow for optimizing hyperparameters with gradients. The standard way of computing these gradients involves a forward and backward pass of computations. However, the backward pass usually needs to consume unaffordable memory to store all the intermediate variables to exactly reverse the forward training procedure. In this work we propose a simple but effective method, DrMAD, to distill the knowledge of the forward pass into a shortcut path, through which we approximately reverse the training trajectory. Experiments on several image benchmark datasets show that DrMAD is at least 45 times faster and consumes 100 times less memory compared to state-of-the-art methods for optimizing hyperparameters with minimal compromise to its effectiveness. To the best of our knowledge, DrMAD is the first research attempt to make it practical to automatically tune thousands of hyperparameters of deep neural networks.

## Citation
```
@article{drmad2016,
  title={DrMAD: Distilling Reverse-Mode Automatic Differentiation for Optimizing Hyperparameters of Deep Neural Networks},
  author={Fu, Jie and Luo, Hongyin and Feng, Jiashi and Low, Kian Hsiang and Chua, Tat-Seng},
  journal={arXiv preprint arXiv:1601.00917},
  year={2016}
}

```

## GPU Version (Lua/Torch)

I'm working with [Nicholas Leonard](https://github.com/nicholas-leonard) from Element Research (NYC) to provide a toolbox for efficiently tuning Lua/Torch based deep learning systems. It is of course under the MIT license. The code has been moved to a new repo and under active development:

**Check it out at** https://github.com/nicholas-leonard/drmad

## CPU Version (Python)

The CPU code is used in the original paper. The code is mainly modified from [Gradient-based Optimization of Hyperparameters through Reversible Learning](https://github.com/HIPS/hypergrad/). 

### How to run these experiments (following the instruction of hypergrad)

To reproduce our experiments, use the code in [/cpu_py/experiments](https://github.com/bigaidream-projects/drmad/tree/master/cpu_py/experiments) folder, e.g. [./exp1/safe/safe.py](https://github.com/bigaidream-projects/drmad/blob/master/cpu_py/experiments/exp1/safe/safe.py). 

> We strongly recommend that you take a look at the code of [autograd](https://github.com/HIPS/autograd) first. 

You'll need to install [autograd](https://github.com/HIPS/autograd), our automatic differentiation package.
However, autograd (aka funkyYak) has changed a lot since we wrote the hypergrad code, and it would take a little bit of work to make them compatible again.

However, the hypergrad code should work with the version of FunkyYak as of Feb 2, at this revision:
https://github.com/HIPS/autograd/tree/be470d5b8d6c84bfa74074b238d43755f6f2c55c

So if you clone autograd, then type
git checkout be470d5b8d6c84bfa74074b238d43755f6f2c55c,
you should be at the same version we used to run the experiments.

That version also predates the setup.py file, so to get your code to use the old version, you'll either have to copy setup.py into the old revision and reinstall, or add FunkyYak to your PYTHONPATH.


## Acknowledgements
Jie Fu would like to thank Microsoft Azure for Research for providing the computational resources. This work is also supported by NUS-Tsinghua Extreme Search (NExT) project through the National Research Foundation, Singapore.