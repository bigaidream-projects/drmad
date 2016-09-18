
# CPU Version (Python)

We no longer update CPU version and keep it here mainly for reference.

The CPU code is used in the original paper. The detailed how-to-run instructions can be found [here](https://github.com/bigaidream-projects/drmad/tree/master/cpu_ver/README.md).



---

### How to reproduce our experiments

#### Dependencies


The code is mainly modified from [Gradient-based Optimization of Hyperparameters through Reversible Learning](https://github.com/HIPS/hypergrad/).

> We strongly recommend that you take a look at the code of [autograd](https://github.com/HIPS/autograd) first.

You'll need to install [autograd](https://github.com/HIPS/autograd), an automatic differentiation package.
However, autograd (aka funkyYak) has changed a lot since they wrote the hypergrad code, and it would take a little bit of work to make them compatible again.

However, the hypergrad code should work with the version of FunkyYak as of Feb 2, at this revision:
https://github.com/HIPS/autograd/tree/be470d5b8d6c84bfa74074b238d43755f6f2c55c

So if you clone autograd, then type
git checkout be470d5b8d6c84bfa74074b238d43755f6f2c55c,
you should be at the same version we used to run the experiments.

That version also predates the setup.py file, so to get your code to use the old version, you'll either have to copy setup.py into the old revision and reinstall, or add FunkyYak to your PYTHONPATH.

#### How to run

Use the code in [/cpu_ver/experiments](https://github.com/bigaidream-projects/drmad/tree/master/cpu_ver/experiments) folder, e.g. [./exp1/safe/safe.py](https://github.com/bigaidream-projects/drmad/blob/master/cpu_ver/experiments/exp1/safe/safe.py).
