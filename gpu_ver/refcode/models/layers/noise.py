import theano
import theano.tensor as T


def noise_conditions(params, index, noiseWhere):
    doNoise = (('addNoise' in params.rglrz) or (index == 0 and 'inputNoise' in params.rglrz)) and params.noiseWhere == noiseWhere
    if params.model == 'convnet':
        doNoise = doNoise and params.convLayers[index].noise 
    return doNoise           

def dropout_conditions(params, index, noiseWhere):
    doDrop = (('dropOut' in params.rglrz) or ('dropOutB' in params.rglrz))
    if params.model == 'convnet':
        doDrop = doDrop and params.convLayers[index].noise 
    return doDrop           

        
def noiseup(x, useRglrz, noiz, noiseType, params, index, rstream):    
    ''' 
        Additive and multiplicative Gaussian Noise    
    '''

    if 'inputNoise' in params.rglrzPerMap or 'addNoise' in params.rglrzPerMap:
        noiz = noiz.dimshuffle('x', 0, 'x', 'x') 

    if noiseType == 'multi1':                
        x = x * (1. + useRglrz*noiz*rstream.normal(x.shape, dtype=theano.config.floatX))
    elif noiseType == 'multi0':                
        x = x * ((1.-useRglrz)*1.+useRglrz*noiz)*rstream.normal(x.shape, dtype=theano.config.floatX)
    else:
        x = x + useRglrz*noiz*rstream.normal(x.shape, dtype=theano.config.floatX)
    return x

def dropout(x, useRglrz, drop, params, nIn, rstream):
    ''' 
        Dropout noise.     
    '''    
    doDrop = useRglrz
    scaleLayer = 1 - useRglrz

    if 'dropOut' in params.rglrz:
        mask = rstream.binomial(n=1, p=(1.-drop), size=x.shape, dtype=theano.config.floatX)
        x = x * (1*(1-doDrop)+mask*doDrop)
        x = x * (1-drop*scaleLayer)

    elif 'dropOutB' in params.rglrz:
        mask = rstream.binomial(n=1, p=(1.-drop), size=(nIn,), dtype=theano.config.floatX)
        x = x * (1*(1-doDrop)+mask*doDrop)
        x = x * (1-drop*scaleLayer)
    return x


