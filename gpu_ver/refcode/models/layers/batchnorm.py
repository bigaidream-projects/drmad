import numpy as np
import theano
import theano.tensor as T

import theano.tensor.nnet.bn as bn

eps = np.float32(1e-6)
zero = np.float32(0.)
one = np.float32(1.)

def bn_shared(params, outFilters, index):    

    ''' Setup BN shared variables.    

    '''
    normParam = {}       
    template = np.ones((outFilters,), dtype=theano.config.floatX)
    normParam['mean'] = theano.shared(value=0.*template, name='mean_%d' % (index), borrow=True)
    normParam['var'] = theano.shared(value=1.*template, name='var_%d' % (index), borrow=True)                                
    normParam['mean_batch'] = theano.shared(value=0.*template, name='mean_batch_%d' % (index), borrow=True) # need for exact 
    normParam['var_batch'] = theano.shared(value=1.*template, name='var_batch_%d' % (index), borrow=True) # need for exact                               
    normParam['iter'] = theano.shared(np.float32(1.), name='iter')                 

    paramsBN = [normParam['mean'], normParam['var'], normParam['mean_batch'], normParam['var_batch'], normParam['iter']]
    return normParam, paramsBN


def bn_layer(x, a, b, normParam, params, phase):

    ''' Apply BN.    

    # phase = 0 : BN eval with m1v1, BN ups weighter average 
    # phase = 1 : BN eval with m2v2, no BN ups

    '''
        
    minAlpha = params.movingAvMin
    iterStep = params.movingAvStep                  
    # compute mean & variance    
    if params.model == 'convnet':
        mean1 = T.mean(x, axis = (0, 2, 3))
        var1 = T.var(x, axis = (0, 2, 3))
    else:
        mean1 = T.mean(x, axis = 0)
        var1 = T.var(x, axis = 0)

    # moving average as a proxi for validation model 
    alpha = (1.-phase)*T.maximum(minAlpha, 1./normParam['iter'])                     
    mean2 = (1.-alpha)*normParam['mean'] + alpha*mean1 
    var2 = (1.-alpha)*normParam['var'] + alpha*var1   

    mean = (1.-phase)*mean2 + phase*mean1 
    var = (1.-phase)*var1 + phase*var1
    std = T.sqrt(var+eps)

    # apply transformation: 
    if params.model == 'convnet':
        x = bn.batch_normalization(x, a.dimshuffle('x', 0, 'x', 'x'), b.dimshuffle('x', 0, 'x', 'x'), 
                                mean.dimshuffle('x', 0, 'x', 'x'), std.dimshuffle('x', 0, 'x', 'x'), mode='high_mem')
    else:    
        x = bn.batch_normalization(x, a, b, mean, std) 
    updateBN = [mean2, var2, mean1, var1, normParam['iter']+iterStep]  
    return x, updateBN


def update_bn(model, params, evaluateBN, t1Data, t1Label):
    
    ''' Computation of exact batch normalization parameters for the trained model (referred to test-BN).

    Implemented are three ways to compute the BN parameters: 
        'lazy'      test-BN are approximated by a running average during training
        'default'   test-BN are computed by averaging over activations of params.m samples from training set 
        'proper'    test-BN of k-th layer are computed as in 'default', 
                    however the activations are recomputed by rerunning with test-BN params on all previous layers

    If the setting is 'lazy', this function will not be called, since running average test-BN 
    are computed automatically during training.                

    '''
    
    oldBN, newBN = [{}, {}]
    nSamples1 = t1Data.shape[0]

    batchSizeBN = nSamples1/params.m    
    trainPermBN = range(0, nSamples1)
    np.random.shuffle(trainPermBN)


    # list of layers which utilize BN    
    if params.model == 'convnet':
        allLayers = params.convLayers
        loopOver = filter(lambda i: allLayers[i].bn, range(len(allLayers)))
        print loopOver
    else:
        loopOver = range(params.nLayers-1)
        
    # extract old test-BN parameters, reset new
    oldBN['mean'] = map(lambda i: model.h[i].normParam['mean'].get_value(), loopOver)
    oldBN['var'] = map(lambda i: model.h[i].normParam['var'].get_value(), loopOver)                    
    newBN['mean'] = map(lambda i: 0.*oldBN['mean'][i], range(len(loopOver)))
    newBN['var'] = map(lambda i: 0.*oldBN['var'][i], range(len(loopOver)))

    # CASE: 'proper' 
    if params.testBN == 'proper':
        
        # loop over layers, loop over examples
        for i in len(range(loopOver)):
            layer = loopOver[i]
            for k in range(0, params.m):
                sampleIndexBN = trainPermBN[(k * batchSizeBN):((k + 1) * (batchSizeBN))]
                evaluateBN(t1Data[sampleIndexBN], 0, 1)                         

                newBN['mean'][i] = model.h[layer].normParam['mean_batch'].get_value() + newBN['mean'][i]
                newBN['var'][i] = model.h[layer].normParam['var_batch'].get_value() + newBN['var'][i]                   

            np.random.shuffle(trainPermBN)
            biasCorr = batchSizeBN / (batchSizeBN-1)                     
            # compute mean, adjust for biases
            newBN['mean'][i] /= params.m
            newBN['var'][i] *= biasCorr/params.m
            model.h[layer].normParam['mean'].set_value(newBN['mean'][i])
            model.h[layer].normParam['var'].set_value(newBN['var'][i])
        

    # CASE: 'default'                       
    elif params.testBN == 'default': 
        
        # loop over examples
        for k in range(0, params.m):
            sampleIndexBN = trainPermBN[(k * batchSizeBN):((k + 1) * (batchSizeBN))]
            evaluateBN(t1Data[sampleIndexBN], 0, 0)
                        
            newBN['mean'] = map(lambda (i, j): model.h[i].normParam['mean_batch'].get_value() + newBN['mean'][j], zip(loopOver, range(len(loopOver))))
            newBN['var'] = map(lambda (i, j): model.h[i].normParam['var_batch'].get_value() + newBN['var'][j], zip(loopOver, range(len(loopOver))))                    
                    
        # compute mean, adjust for biases
        biasCorr = batchSizeBN / (batchSizeBN-1)             
        newBN['var'] = map(lambda i: newBN['var'][i]*biasCorr/params.m, range(len(loopOver)))        
        newBN['mean'] = map(lambda i: newBN['mean'][i]/params.m, range(len(loopOver)))

        # updating test-BN parameters, update shared
        map(lambda (i,j): model.h[i].normParam['mean'].set_value(newBN['mean'][j]), zip(loopOver, range(len(loopOver))))
        map(lambda (i,j): model.h[i].normParam['var'].set_value(newBN['var'][j]), zip(loopOver, range(len(loopOver))))

    # printing an example of previous and updated versions of test-BN
    print 'BN samples: '
    print 'mean low', oldBN['mean'][1][0], newBN['mean'][1][0]
    print 'var low', oldBN['var'][1][0], newBN['var'][1][0]
    print 'mean up', oldBN['mean'][-1][0], newBN['mean'][-1][0]
    print 'var up', oldBN['var'][-1][0], newBN['var'][-1][0]
    
    return model 
    

