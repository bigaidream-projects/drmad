# DrMAD in Theano

> WARNING: this version is not stable and we are still investigating. 


DrMAD-Theano uses `Lasagne` to build a simple MLP. 

Run:

`THEANO_FLAGS=mode=FAST_RUN,device=gpu0,floatX=float32 python simple_mlp.py`


### Structure

- `simple_mlp.py` includes three phases:
    - Phase 1: Algorithm 1. 
    - Phase 2: obtain the validation loss on validation set. 
    (Since there're multiple iterations, we have to output the gradients
     and take the average of `grads` across different iterations.)
    - Phase 3: Algorithm 2.
        - We use Lop() to obtain the hessian-vector products in line 6-7
         of Algo. 2., which is defined in `hypergrad.py`.
- `args.py` configuration for DrMAD
- `layers.py` provides class `DenseLayerWithReg()` to build up a simple MLP.
- `models.py` provides class `MLP()`.
- `updates.py` provides update rules for different theano functions.

### References
[hypergrad](https://github.com/HIPS/hypergrad)
[T1T2](https://github.com/jelennal/t1t2)
