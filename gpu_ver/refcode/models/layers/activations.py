import numpy as np
import theano.tensor as T

eps = np.float32(1e-8)
zero = np.float32(0.)
one = np.float32(1.)
leaky_slope = np.float32(1e-2)
convNonLin = 'relu'

def relu(x):
    output = T.maximum(0., x)
    return output

def leaky_relu(x):
    output = T.maximum(0., x) + leaky_slope*T.minimum(0, x)
    return output

    
def activation(x, key):
    ''' 
        Defining various activation functions.    
    '''
    identity = lambda x: x    
    activFun = {'lin':  identity,
                'relu': relu,
                'leaky_relu':leaky_relu,
                'tanh': T.tanh, 
                'sig':  T.nnet.sigmoid, 
                'softmax': T.nnet.softmax,
                }[key]
    
    return activFun(x)   


def weight_multiplier(nIn, nOut, key):    
    ''' 
        Initial range of values for weights, given diffrent activation functions.    
    '''    
    weightMultiplier = {'lin':  np.sqrt(1./(nIn+nOut)), 
                        'relu': np.sqrt(1./(nIn+nOut))*np.sqrt(12), 
                        'elu':  np.sqrt(1./(nIn+nOut))*np.sqrt(12),
                        'leaky_relu': np.sqrt(1./(nIn+nOut))*np.sqrt(12),     
                        'tanh': np.sqrt(1./(nIn+nOut))*np.sqrt(6.),
                        'sig':  np.sqrt(1./(nIn+nOut))*np.sqrt(6.)/4, 
                        'softmax': 1e-5}[key]

    return weightMultiplier       
    
    
    