"""
this code is modified from:
    https://github.com/jelennal/t1t2/blob/master/models/convnet.py
"""

import numpy as np
import theano
import theano.tensor as T

from models.layers.conv_layers import conv_layer, pool_layer, average_layer
from refcode.training.monitor import stat_monitor

zero = theano.shared(value=0., borrow=True)


class convnet(object):
    def __init__(self, rng, rstream, x, wantOut, params, useRegular, bnPhase, globalParams=None):
        """
        Constructing the convolutional model.

        Arguments:
            rng, rstream         :: random streams
            x1, x2       :: x batches from T1 and T2 set
            wantOut1, wantOut2   :: corresponding labels
            params               :: all model parameters
            graph                :: theano variable determining how are BN params computed
            globalParams         :: T2 params when one-per-network

        """

        # defining shared variables shared across layers
        if globalParams is None:
            globalParams = {}
            for rglrz in params.rglrzPerNetwork:
                tempParam = np.asarray(params.rglrzInitial[rglrz][0], dtype=theano.config.floatX)
                globalParams[rglrz] = theano.shared(value=tempParam, name='%s_%d' % (rglrz, 0), borrow=True)
            for rglrz in params.rglrzPerNetwork1:
                tempParam = np.asarray(params.rglrzInitial[rglrz][0], dtype=theano.config.floatX)
                globalParams[rglrz + str(0)] = theano.shared(value=tempParam, name='%s_%d' % (rglrz, 0), borrow=True)
                tempParam = np.asarray(params.rglrzInitial[rglrz][1], dtype=theano.config.floatX)
                globalParams[rglrz] = theano.shared(value=tempParam, name='%s_%d' % (rglrz, 1), borrow=True)

        # initializations of counters, lists and dictionaries
        i = 0
        h = []
        penalty = 0.
        trackT2Params = {}
        for param in params.rglrz:
            trackT2Params[param] = []
        paramsT1, paramsT2, paramsBN, updateBN = [[], [], [], []]
        netStats = {}
        for key in params.activTrack:
            netStats[key] = []

        '''
            Constructing layers.
        '''
        for layer in params.convLayers:

            # construct layer
            print 'layer ', str(i), ':', layer.type, layer.filter, layer.maps, ' filters'
            if layer.type == 'conv':
                h.append(conv_layer(rng=rng, rstream=rstream, index=i, x=x,
                                    params=params, globalParams=globalParams, useRglrz=useRegular, bnPhase=bnPhase,
                                    filterShape=layer.filter, inFilters=layer.maps[0], outFilters=layer.maps[1],
                                    stride=layer.stride))
            elif layer.type == 'pool':
                h.append(pool_layer(rstream=rstream, index=i, x=x,
                                    params=params, useRglrz=useRegular, bnPhase=bnPhase,
                                    poolShape=layer.filter, inFilters=layer.maps[0], outFilters=layer.maps[1],
                                    stride=layer.stride))
            elif layer.type in ['average', 'average+softmax']:
                h.append(average_layer(rstream=rstream, index=i, x=x,
                                       params=params, useRglrz=useRegular, bnPhase=bnPhase,
                                       poolShape=layer.filter, inFilters=layer.maps[0], outFilters=layer.maps[1],
                                       stride=layer.stride))
            # elif layer.type == 'softmax':
            #                h.append(mlp_layer(rng=rng, rstream=rstream, index=i, splitPoint=splitPoint, x=x,
            #                                   params=params, globalParams=globalParams, graph=graph))

            # collect penalty term
            if layer.type in ['conv', 'softmax'] and ('L2' in params.rglrz):
                if 'L2' in params.rglrzPerMap:
                    tempW = h[-1].rglrzParam['L2'].dimshuffle(0, 'x', 'x', 'x') * T.sqr(h[-1].W)
                else:
                    tempW = h[-1].rglrzParam['L2'] * T.sqr(h[-1].W)
                penalty += T.sum(tempW)

                # collect T1 params
            if layer.type in ['conv', 'softmax']:
                paramsT1 += h[i].paramsT1
            elif params.batchNorm and params.convLayers[i].bn:
                paramsT1 += h[i].paramsT1

            # collect T2 params
            if params.useT2:
                paramsT2 += h[i].paramsT2

            # collect T2 for tracking
            for param in params.rglrz:
                if param == 'xNoise':
                    if i == 0 and layer.noise:
                        trackT2Params[param] += [h[-1].rglrzParam[param]]
                    else:
                        trackT2Params[param] += [zero]
                if param == 'addNoise':
                    if layer.noise:
                        trackT2Params[param] += [h[-1].rglrzParam[param]]
                    else:
                        trackT2Params[param] += [zero]
                if param in ['L1', 'L2', 'Lmax']:
                    if layer.type in ['conv', 'softmax']:
                        trackT2Params[param] += [h[-1].rglrzParam[param]]
                    else:
                        trackT2Params[param] += [zero]

                        # collect BN params&updates
            if params.batchNorm and params.convLayers[i].bn:
                paramsBN += h[-1].paramsBN
                updateBN += h[-1].updateBN

            x = h[-1].output
            i += 1

            # pack variables for output
        for rglrz in globalParams.keys():
            if rglrz in params.rglrzTrain:
                paramsT2 += [globalParams[rglrz]]
        self.paramsT1 = paramsT1
        self.paramsT2 = paramsT2

        self.paramsBN = paramsBN
        self.updateBN = updateBN

        # fix tracking of stats
        if params.trackStats:
            self.netStats = stat_monitor(layers=h, params=params)
        else:
            self.netStats = T.constant(0.)
        self.trackT2Params = trackT2Params
        for param in params.rglrz:
            print len(trackT2Params[param])
        print '# t1 params: ', len(paramsT1), '# t2 params: ', len(paramsT2)

        # output and predicted labels
        self.h = h
        self.y = h[-1].output
        self.guessLabel = T.argmax(self.y, axis=1)
        self.penalty = penalty if penalty != 0. else T.constant(0.)
        self.guessLabel = T.argmax(self.y, axis=1)

        # cost functions
        def stable(x, stabilize=True):
            if stabilize:
                x = T.where(T.isnan(x), 1000., x)
                x = T.where(T.isinf(x), 1000., x)
            return x

        if params.cost == 'categorical_crossentropy':
            def costFun1(y, label):
                return stable(-T.log(y[T.arange(label.shape[0]), label]),
                              stabilize=True)
        else:
            raise NotImplementedError
        if params.cost_T2 in ['categorical_crossentropy', 'sigmoidal', 'hingeLoss']:
            def costFun2(y, label):
                return stable(-T.log(y[T.arange(label.shape[0]), label]),
                              stabilize=True)
        else:
            raise NotImplementedError

        def costFunT1(*args, **kwargs):
            return T.mean(costFun1(*args, **kwargs))

        def costFunT2(*args, **kwargs):
            return T.mean(costFun2(*args, **kwargs))

        #        self.y1_avg = self.y1
        #        self.guessLabel1_avg = self.guessLabel1


        #        self.trainCost = useRglrz*costFunT1(self.y, wantOut) + (1-useRglrz)*costFunT2(self.y, wantOut)
        self.trainCost = costFunT1(self.y, wantOut)
        self.clasRate = T.mean(T.cast(T.neq(self.guessLabel, wantOut), 'float32'))