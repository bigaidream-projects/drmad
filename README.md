# DrMAD

![](https://github.com/bigaidream-projects/drmad/blob/master/docs/shortcut.jpg)

[![License](http://img.shields.io/badge/license-MIT-brightgreen.svg?style=flat)](LICENSE)

> Source code for the paper: [Distilling Reverse-Mode Automatic Differentiation for Optimizing Hyperparameters of Deep Neural Networks](http://arxiv.org/abs/1601.00917).

## What's DrMAD?

DrMAD is a hyperparameter tuning method based on automatic differentiation, which is a most [criminally underused tool](https://justindomke.wordpress.com/2009/02/17/automatic-differentiation-the-most-criminally-underused-tool-in-the-potential-machine-learning-toolbox/) for machine learning.

DrMAD can tune thousands of continuous hyperparameters (e.g. L1 norms for every single neuron) for deep models on GPUs.

### AD vs Bayesian optimization (BO)

BO, as a global optimization approach, can hardly support tuning more than 20 hyperparameters, because it treats the learning algorithm being tuned as a black-box and can only get feedback signals after convergence. Also, the learning rate tuned by BO is fixed for all iterations.

AD is different from symbolic differentiation (used by Mathematica), and numerical differentiation. AD, as a local optimization method based on gradient information, can make use of the feedback signals after every iteration and can tune thounsands of hyperparameters by using (hyper-)gradients with respect to hyperparameters. Checkout this [paper](https://arxiv.org/abs/1502.05767) if you want to understand AD techniques deeply.

Therefore, AD can tune hundreds of thousands of constant (e.g. L1 norm for every neuron) or dynamic (e.g. learning rates for every neuron at every iteration) hyperparameters.

The standard way of computing these (hyper-)gradients involves a forward and backward pass of computations. However, the backward pass usually needs to consume unaffordable memory (e.g. TBs of RAM for MNIST dataset) to store all the intermediate variables to `exactly` reverse the forward training procedure.

### Hypergradient with an approximate backward pass

![](https://github.com/bigaidream-projects/drmad/blob/master/docs/fig.jpg)


We propose a simple but effective method, DrMAD, to distill the knowledge of the forward pass into a shortcut path, through which we `approximately` reverse the training trajectory. When run on CPUs, DrMAD is at least 45 times faster and consumes 100 times less memory compared to state-of-the-art methods for optimizing hyperparameters with almost no compromise to its effectiveness. 

## CPU code for reproducing

For reproducing the original result in the paper, please refer to [CPU version](https://github.com/bigaidream-projects/drmad/tree/master/cpu_ver)

> In the original paper, we set the momentum to a small value (0.1). Now we found that setting this value to 0.9 or even 0.95 will give much better performance. 

## GPU code

We've implemented [GPU version with Theano](https://github.com/bigaidream-projects/drmad/tree/master/gpu_ver). In addition to the experiments in the original paper, we did experiments with convolutional networks in CIFAR-10. DrMAD also works with a small CNN on CIFAR-10. 

---

## Citation
```
@article{drmad2016,
  title={DrMAD: Distilling Reverse-Mode Automatic Differentiation for Optimizing Hyperparameters of Deep Neural Networks},
  author={Fu, Jie and Luo, Hongyin and Feng, Jiashi and Low, Kian Hsiang and Chua, Tat-Seng},
  journal={arXiv preprint arXiv:1601.00917},
  year={2016}
}

```

## Contact

If you have any problems or suggestions, please contact: jie.fu A_T u.nus.edu~~cation~~
