import numpy as np
import theano.tensor as T

from refcode.models.layers.shared import t1_shared, t2_shared
from models.layers.activations import activation
from models.layers.noise import noise_conditions, dropout_conditions, noiseup, dropout
from models.layers.batchnorm import bn_shared, bn_layer

eps = np.float32(1e-8)
zero = np.float32(0.)
one = np.float32(1.)


def nan_o_meter(x):
    n_nans = T.sum(T.isnan(x))
    n_infs = T.sum(T.isinf(x))
    print '# Nans ', n_nans, '# Infs', n_infs
    return


class mlp_layer(object):
    def __init__(self, rng, rstream, index, x,
                 params, globalParams, useRglrz, bnPhase,
                 W=None, b=None, a=None, rglrzParam=None, normParam=None, normWindow=None):

        '''
            Class defining a fully connected layer.
        '''
        if params.model == 'convnet':
            nonLin = 'softmax'
            nIn = 10
            nOut = 10
        else:
            nonLin = params.activation[index]
            nIn = params.nHidden[index]
            nOut = params.nHidden[index + 1]

        '''
            Initializing shared variables.
        '''

        # defining shared T1 params
        if W is None:
            W, b, a = t1_shared(params, rng, index, nIn, nOut, nOut)
        self.W = W;
        self.b = b;
        self.a = a
        if params.batchNorm and (not params.aFix) and (nonLin != 'softmax'):
            self.paramsT1 = [W, b, a]
        else:
            self.paramsT1 = [W, b]

        # defining shared T2 params
        self.paramsT2 = []
        if rglrzParam is None:
            rglrzParam = t2_shared(params, globalParams, index, nIn, nOut)

        self.rglrzParam = rglrzParam
        self.paramsT2 = []
        if params.useT2:
            for rglrz in params.rglrzTrain:
                if (rglrz not in params.rglrzPerNetwork) and (rglrz not in params.rglrzPerNetwork1):
                    if rglrz != 'addNoise' or nonLin != 'softmax':
                        self.paramsT2 += [rglrzParam[rglrz]]  # if trained, put param here

        # defining shared BN params
        if normParam is None and params.batchNorm and nonLin != 'softmax':
            if normParam is None:
                normParam, paramsBN = bn_shared(params, nOut, index)
            self.normParam = normParam
            self.paramsBN = paramsBN

        # noise
        if (index == 0 and 'inputNoise' in rglrzParam.keys()):
            noiz = self.rglrzParam['inputNoise']
        elif 'addNoise' in rglrzParam.keys():
            noiz = self.rglrzParam['addNoise']
        if ('dropOut' in rglrzParam.keys()):
            drop = self.rglrzParam['dropOut']
        elif 'dropOutB' in rglrzParam.keys():
            drop = self.rglrzParam['dropOutB']

        '''
            Input transformations: convolution, BN, noise, nonlinearity
        '''

        # add normal noise to x
        self.x = x
        if noise_conditions(params, index, 'type0'):
            x = noiseup(x, useRglrz, noiz, params.noiseT1, params, index, rstream)
        if dropout_conditions(params, index, 'type0'):
            x = dropout(x, useRglrz, drop, params, nIn, rstream)

        # affine transform
        xLin = T.dot(x, self.W)

        # batchnorm transform
        if params.batchNorm and nonLin != 'softmax':
            xLin, updateBN = bn_layer(xLin, self.a, self.b, self.normParam, params, bnPhase)
            self.updateBN = updateBN
        else:
            xLin += self.b

        # noise before nonlinearity
        if noise_conditions(params, index, 'type1'):
            xLin = noiseup(xLin, useRglrz, noiz, params.noiseT1, params, index, rstream)
            # nonlinearity
        self.output = activation(xLin, nonLin)


