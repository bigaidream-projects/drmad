### A deduction of the paper 

> Paper: Gradient-based Hyperparameter Optimization through Reversible Learning
> arxiv: http://arxiv.org/abs/1502.03492

Please use Moeeditor to preview this file.

There's some typos in the pseudo code of the original paper. 

>Note: here, we use $\theta$ to indicate the elementary parameters rather than $w$, and since we use $\gamma$ to denote the hyper parameters, so we use $m$ to denote decays. 

>$T$: the number of iterations

#### Algorithm 1 (meta-forward)

**input**: intial $\theta_1$, decays $m$, learning rates $\alpha$, training loss $L(\theta, t; \gamma)$.

**initialize**: $v_1 = 0$

**for** t = 1 **to** $T$ **do**

&nbsp; &nbsp; &nbsp;&nbsp; $g_t = \nabla_\theta L(\theta, t; \gamma) $

&nbsp; &nbsp; &nbsp;&nbsp; $v_{t+1} = m_t v_t - (1-m_t) g_t$

&nbsp; &nbsp; &nbsp;&nbsp; $\theta_{t+1} = \theta_t + \alpha_t v_t$


**end for**

**output**: trained parameters $\theta_{T+1}$

> Note: decay $m$ and learning rate $\alpha$ can be different across different iterations, but in practice we usually set it fixed (i.e. $m_1 = m_2 = \cdots = m_T$, and $\alpha_1 = \alpha_2 = \cdots = \alpha_T$).



***

> Note: all the symbols in the algorithm 1 & 2 are symbols of variable in the sense of pragramming language, not math. That is to say, symbols like $dv, dm$ are variables (in Python), not notations for math, though there's a relation between them.


#### Alogorithm 2 (meta-backward)
**input**: $\theta_{T+1}$, $v_{T+1}$ decays $m$, learning rates $\alpha$, training loss $L(\theta, t; \gamma)$, loss $f(\theta)$.

**initialize**: $dv = 0, d\gamma = 0, d\alpha = 0, dm = 0$ (note,  $d\alpha, dm$ might be a matrix of size $T \times \#\text{parameters}$, or $T \times 1$)

**initialize**: $d\theta = \nabla_\theta f(\theta_{T+1})$

**for** t = $T$ **downto** 1 **do**

&nbsp; &nbsp; &nbsp;&nbsp; $d\alpha_t = d\theta^{\text{T}} v_{t}$ 
> &nbsp; &nbsp; &nbsp;&nbsp;&nbsp; &nbsp; &nbsp;&nbsp;
> [HINT: chain's rule, $\frac{df(\theta_{T+1})}{d\alpha_t} = (\nabla_\theta f(\theta_{T+1}) )^{\text{T}}\cdot \frac{d\theta_{T+1}}{d\alpha_t}$]

&nbsp; &nbsp; &nbsp;&nbsp; $\theta_{t} = \theta_{t+1} - \alpha_t v_t$

&nbsp; &nbsp; &nbsp;&nbsp; $g_t = \nabla_\theta L(\theta, t; \gamma) $

&nbsp; &nbsp; &nbsp;&nbsp; $v_{t} = [v_{t+1} + (1-m_t)g_t]/m_t$

&nbsp; &nbsp; &nbsp;&nbsp; $dv = dv + \alpha_t d\theta$

> &nbsp; &nbsp; &nbsp;&nbsp;&nbsp; &nbsp; &nbsp;&nbsp;
> [HINT: this $dv$ is actually store $\frac{df}{dv_t}$, this is same for $dw, d\alpha, dm$, in each run, they are overridden]

&nbsp; &nbsp; &nbsp;&nbsp; $dm_t = dv^{\text{T}}(v_t + g_t)$

&nbsp; &nbsp; &nbsp;&nbsp; $d\theta = d\theta - (1 - m_t) dv \nabla_\theta\nabla_\theta L(\theta, t; \gamma)$

&nbsp; &nbsp; &nbsp;&nbsp; $d\gamma = d\gamma - (1 - m_t) dv \nabla_\gamma\nabla_\theta L(\theta, t; \gamma)$

&nbsp; &nbsp; &nbsp;&nbsp; $dv = m_t \cdot dv$

**end for**

**output**: gradient of $f(\theta_{T+1})$ w.r.t. $w_1$, $v_1$, $m$, $\alpha$, and w.r.t. the hyper params $\gamma$

