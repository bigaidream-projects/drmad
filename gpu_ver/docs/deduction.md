## reference

- Hypergrad in AD (HIPS's Autograd): https://github.com/HIPS/hypergrad
 
- T1T2 in Theano: https://github.com/jelennal/t1t2

- DrMAD in AD (HIPS's Autograd): https://github.com/bigaidream-projects/drmad


## Introduction

This doc is written using https://github.com/Moeditor/Moeditor, please download it to better render the LaTeX script.

### Jargon

Here, we have a suitable definition on the jargon.

#### Parameters

`elementary parameters` are all the the trainable parameters (usually hyper-parameters are not trainable), such as `weights` and `bias`. So, the parameters of a model is divided into two fractions: one is `elementary parameters` (denoted as $\theta$), the other is `hyper-parameters` (denoted as $\lambda$).

#### Loss function

Define two loss function, one is $\mathcal{l}(\theta)$ (a.k.a. `validation loss`), which do not involve the regularization terms. And the other is $\mathcal{L}(\theta, \lambda) = \mathcal{l}(\theta) + \Omega(\theta, \lambda)$ (a.k.a. `training loss`), which involve the penalty terms $\Omega(\theta, \lambda)$.

> In code, we use `loss_*_L1` to denote the specific $\mathcal{L}_1(\mathcal{D}|\theta, \lambda)$ loss evaluated at $\mathcal{D}$ = a iteration data `*`.

#### Partition Datasets

For convenience, we divide the dataset into three parts: 
- `Elementray training set(ESet)` or `(1)`, 
- `Hyper training set (HSet)` or `(2)`, and
- `Validation set (VSet)` or `(3)`.

Conventionally, model selection is based on `VSet`. Since we use hyper-gradient to optimize hyper-parameters, we prefer to use a seprate `HSet` to obtain the `validation loss` for hyper-parameter optimization. And the `VSet` is only for `validation` (i.e. how many `meta-runs` or `episodes` are the best?). 

:exclamation: **Note that**, hyper-parameters themselves are **not** involved during the training on `HSet`.

> We denote $\mathcal{L}_1(\theta, \lambda)$ as the `training` loss on the first part of dataset (i.e. elementary training set). $\mathcal{l}_2(\theta)$ for the `validation` loss on the second part of dataset (i.e. hyper training set).

> In coding, we use `grad_x_L1` to denote the function $\nabla_x\mathcal{L}_1$. And `d**_L1` to denote the result of he function $\nabla_x\mathcal{L}_1 (x)$ evaluated at $x = $ `**`, where `**` can be any symbol.

#### Training

 One `meta-training` or `episode` is a complete training till the network converge. Usually, it will take serval episodes to obtain the best setting of hyper-parameters. Note that,  episode `t`'s result (i.e. hyper-parameters that have been updated by hyper-gradients) will be used as the default hyper-parameters in episode`t+1`.
 
 We prefer to use `elementary forward` and `elementary backward` to indicate forward and backward on `ESet`, which only affects the `elementary parameters`.
 
 And we use `meta forward` and `meta backword` to indicate forward and backward in the two seperate phases of one `episode`.




### Training Description

We denote `elementary parameters` as $\theta_i$, the subscript indicate $\theta_i$ used in the $i$ iteration in one episode. Assumed there's $T$ iteration of a episode, s.t. $\mathbf{\theta}_{\text{initial}} =  \mathbf{\theta}_{1}$ and $\theta_{\text{final}} = \theta_{T+1}$.

The whole training roughly includes 3 phases:

1. meta-forward: Training on `ESet`, only `elementary parameters` are updated. 
	- we only perserve $\theta_{\text{initial}}$ and $\theta_{\text{final}}$ and do not cache the parameters from iteration $2$ to $T$.
	- **DISSCUSS**:question:: is it enough for us to record the loss on `ESet` and then hack the `theano.function`, i.e. `grads = grad(stored_loss, approximated_params)`
2. Calculate the `validation loss` on `HSet`, denoted as $C_2$, and we compute the gradient w.r.t. elementary params only once in one episode, i.e. $\nabla_\theta C_2$. 
	- **DISSCUSS**:question:: do we need to shuffle the `ESet` and `HSet` in different `episode` as `T1T2` does?
3. meta-backward: Reversed-mode training. (NEED to rewrite...)
	- Calculate $\hat{\theta}_i$. DrMAD's core part: we use a linear approximation elementary parameters $\hat{\theta}_i$ to calcuate the gradients, where
	 $\hat{\theta}_i = \frac{i}{T+1} \cdot \theta_{\text{initial}} + (1 - \frac{i}{T+1}) \cdot \theta_{\text{final}}$.
	- calculate the error $\hat{\lambda}_i = (\nabla_{\lambda} C_2) |_{\theta=\hat{\theta}_i}$ in the $i$ iteration, i.e. $\hat{\lambda}_i = (\nabla_{\lambda} C_2)|_{\theta=\hat{\theta}_i} = \nabla_\theta C_2 (\nabla_\lambda \mathbf{\theta}(\theta_i)) \triangleq G(\theta_i) \approx  G(\hat{\theta}_i) $ 
	- $\tilde{\lambda}_t = \tilde{\lambda}_{t+1} + \hat{\lambda}_t$
	- iterate it for $T$ times, and we got the hyper-parameters we want: $\tilde{\lambda}_1$. (Note that, the subscript of $\lambda$ is using differently in the paper T1T2.)


#### training setting

training script:

```sh
THEANO_FLAGS=mode=Mode,device=gpu0,floatX=float32 python main.py
```

use `mode=Mode` for debug, and `mode=FAST_RUN` for later training.


TODOs:

MNIST + L2 norms + MLPs using DrMAD

