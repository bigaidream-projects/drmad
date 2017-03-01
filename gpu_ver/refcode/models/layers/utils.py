import theano.tensor as T
import numpy as np


# Numerically stable log mean exp. Will keep dims.
def LogSumExp(x, axis=None):
    x_max = T.max(x, axis=axis, keepdims=True)
    return T.log(T.sum(T.exp(x - x_max), axis=axis, keepdims=True)) + x_max


# Numerically stable log mean exp. Will keep dims.
def LogMeanExp(x, axis=None):
    x_max = T.max(x, axis=axis, keepdims=True)
    return T.log(T.mean(T.exp(x - x_max), axis=axis, keepdims=True)) + x_max


# Numerically stable log mean exp. Will keep dims.
def LogMeanExpNp(x, axis=None):
    x_max = np.amax(x, axis=axis, keepdims=True)
    return np.log(np.mean(np.exp(x - x_max), axis=axis, keepdims=True)) + x_max


# Numerically stable log mean exp. Will keep dims.
def LogSumExpNp(x, axis=None):
    x_max = np.amax(x, axis=axis, keepdims=True)
    return np.log(np.sum(np.exp(x - x_max), axis=axis, keepdims=True)) + x_max


# Numerically stable log mean exp. Will keep dims.
def LogSumExp2Np(x, axis=None):
    x_max = np.amax(x, axis=axis, keepdims=True)
    return np.log2(np.sum(2 ** (x - x_max), axis=axis, keepdims=True)) + x_max


# Numerically stable log mean exp. Will keep dims.
def LogMeanExp2Np(x, axis=None):
    x_max = np.amax(x, axis=axis, keepdims=True)
    return np.log2(np.mean(2 ** (x - x_max), axis=axis, keepdims=True)) + x_max


# Numerically stable log mean exp with weighted mean. Will keep dims.
def LogMeanExp2WeightedNp(x, weights, axis=None):
    x_max = np.amax(x, axis=axis, keepdims=True)
    return np.log2(np.sum(2 ** (x - x_max) * weights, axis=axis,
                          keepdims=True)) + x_max
