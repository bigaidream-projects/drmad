import numpy as np
import theano

from models.layers.activations import weight_multiplier

# nonnlinearity used by convnet
convNonLin = 'relu'


def t1_shared(params, rng, index, nIn, nOut, outFilters=7, filterShape=0, defineW=True):

    ''' Initializing T1 shared variables.
    
    '''
    
    if defineW:    

        if params.model == 'convnet':
            if params.convLayers[index].type == 'softmax':
                tempW = np.asarray(rng.randn(nIn, nOut),dtype=theano.config.floatX)
            else:    
                tempW = np.asarray(rng.randn(*filterShape),dtype=theano.config.floatX)
            if params.convLayers[index].type == 'softmax' or params.convLayers[index+1].type == 'average+softmax': 
                tempW *= 1e-5
            else: 
                tempW *= weight_multiplier(nIn, nOut, convNonLin)
        else:
            tempW = np.asarray(rng.randn(nIn, nOut),dtype=theano.config.floatX)
            tempW *= weight_multiplier(nIn, nOut, params.activation[index])
    else:
        tempW = 0.        

    W = theano.shared(value=tempW, name='W_%d' % index, borrow=True)

    tempB = np.zeros((outFilters,), dtype=theano.config.floatX)  #np.asarray(rng.uniform(low=-1.0, high=1.0, size=(nOut,)), dtype=theano.config.floatX)
    b = theano.shared(value=tempB, name='b_%d' % index, borrow=True)

    tempA = np.ones((outFilters,), dtype=theano.config.floatX)  #np.asarray(rng.uniform(low=-1.0, high=1.0, size=(nOut,)), dtype=theano.config.floatX)
    a = theano.shared(value=tempA, name='a_%d' % index, borrow=True)

    return W, b, a


def t2_shared(params, globalParams, index, inFilters, outFilters, filterShape=0):

    ''' Initializing T2 shared variables.
    
    '''

    rglrzParam = {}
    
    for rglrz in params.rglrz:

        if rglrz in params.rglrzPerUnit:
            tempValue = params.rglrzInitial[rglrz][index]   
            if params.model == 'convnet':                              
                tempParam = tempValue * np.ones(*filterShape, dtype=theano.config.floatX) # differentiated by name as well
            else:   
                tempParam = tempValue * np.ones(outFilters, dtype=theano.config.floatX)
            rglrzParam[rglrz] = theano.shared(value=tempParam, name='%s_%d' % (rglrz, index), borrow=True)

        if rglrz in params.rglrzPerMap:
            tempValue = params.rglrzInitial[rglrz][index]                                 
            if (rglrz == 'addNoise' or 'rglrz' == 'inputNoise') and params.noiseWhere == 'type0':
                tempParam = tempValue * np.ones((inFilters, ), dtype=theano.config.floatX) 
            else:
                tempParam = tempValue * np.ones((outFilters, ), dtype=theano.config.floatX) 
            rglrzParam[rglrz] = theano.shared(value=tempParam, name='%s_%d' % (rglrz, index), borrow=True)

        elif rglrz in params.rglrzPerNetwork1:
            if index == 0: # and rglrz == 'addNoise'  
                rglrzParam[rglrz] = globalParams[rglrz+str(0)]
            else: rglrzParam[rglrz] = globalParams[rglrz]

        elif rglrz in params.rglrzPerNetwork:
            if (rglrz == 'inputNoise' and index == 0) or (rglrz != 'inputNoise'):
                rglrzParam[rglrz] = globalParams[rglrz]
        else:
            tempValue = params.rglrzInitial[rglrz][index]                                 
            tempParam = np.asarray(tempValue, dtype=theano.config.floatX)
            rglrzParam[rglrz] = theano.shared(value=tempParam, name='%s_%d' % (rglrz, index), borrow=True)
            
    return rglrzParam        

    
