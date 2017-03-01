import numpy as np
import theano
import theano.tensor as T

'''
    Functions for monitoring various quantities during training. 
    
    grad_monitor - monitors gradient/update norms, angle between gradients/updates; 
    stat_monitor - monitors network statistics

    grad_extract - extract monitored gradient/update norms, angle between gradients/updates; 
    stat_extract - extract monitored network statistics

    t2_extract - extract monitored t2_parameters
    
'''

def t2_extract(model, params, trackNoise, trackPenal):
    # layers to be read from
    loopOver = range(params.nLayers)
    penalList = trackPenal.keys(); noiseList = trackNoise.keys()

    for param in params.rglrz:
        if param == 'inputNoise':
            tempParam = map(lambda i: np.mean(model.trackT2Params[param][i].get_value()), range(1))
            tempParam = np.append(tempParam, np.zeros(len(loopOver)-1))
        else:
            tempParam = map(lambda i: np.mean(model.trackT2Params[param][i].get_value()), loopOver)

        if param in penalList:
             trackPenal[param] = np.append(trackPenal[param], np.array([tempParam]), axis = 0)
        elif param in noiseList:
             trackNoise[param] = np.append(trackNoise[param], np.array([tempParam]), axis = 0)

    return trackNoise, trackPenal


def grad_monitor(param, grad, updates, params, opt, g_t=0., m=0., v=0., e=1e-10):
    
    zero = np.float32(0.); eps = 1e-10
    old_grad = theano.shared(np.float32(param.get_value()) * zero, name="old_grad_%s" % param.name)
    updates.append((old_grad, grad))
    sharedName, _ = param.name.split('_')
    
    # tracked gradient values when adaptive learning rate
    if opt == 'adam':
        old_g_t = m/(T.sqrt(v) + e) 
        all_grads = {
            'grad' : T.mean(T.sqrt(grad**2)),
#            'grad_rel' : T.mean(T.sqrt((grad/(param+1e-12))**2)),
            'grad_angle' : T.sum(grad*old_grad)/(T.sqrt(T.sum(grad**2))*T.sqrt(T.sum(old_grad**2))+eps) ,
#            'grad_max' : T.max(T.sqrt(grad**2)),
            'p_t' : T.mean(T.sqrt((g_t)**2)),
#            'p_t_rel' : T.mean(T.sqrt((g_t/(param+1e-12))**2)),
            'p_t_angle' : T.sum(g_t*old_g_t)/(T.sqrt(T.sum(g_t**2))*T.sqrt(T.sum(old_g_t**2)+eps)),
#            'p_t_max' : T.max(T.sqrt(grad**2))
            }
                
    # tracked gradient values when regular SGD (+momentum)    
    elif opt == None:
        all_grads = {
            'grad' : T.mean(T.sqrt(grad**2)),
#            'grad_rel' : T.mean(T.sqrt((grad/(param+1e-12))**2)),
            'grad_angle' : T.sum(grad*old_grad)/(T.sqrt(T.sum(grad**2))*T.sqrt(T.sum(old_grad**2))+eps) ,
#            'grad_max' : T.max(T.sqrt(grad**2))
            }

    # store tracked grads for output
    temp = []
    if params.listGrads == 'all':
        for grad_type in all_grads.keys():
            temp += [all_grads[grad_type]] 
    else:
        for grad_type in filter(lambda name: name in all_grads.keys(), params.listGrads):
            temp += [all_grads[grad_type]] 

    trackGrads = T.stacklists(temp)        
    return updates, trackGrads


def grad_extract(dataPack, params, sharedNames, trackGrads):

    assert len(dataPack) == len(sharedNames)
    first = True if trackGrads == {} else False     
    tempGrads = {}    
    
    for shared_name, pack in zip(sharedNames, dataPack):
        name, _ = shared_name.split('_') 
        if name in tempGrads.keys():                    
            tempGrads[name] +=  [pack]
        else:
            tempGrads[name] = [pack]

    for name in tempGrads.keys():
        if first:   
            trackGrads[name] = np.array([tempGrads[name]])
        else:
            trackGrads[name] = np.append(trackGrads[name], np.array([tempGrads[name]]), axis = 0)
    
    return trackGrads    


def stat_monitor(layers, params):
    
    i=0; eps = 1e-4; netStats =  {}
    temp = []
    for key in params.activTrack:
        netStats[key] =  []

    for layer in layers:
        
        # mean, std, max for cnn and mlp
        output = layer.output        
        if params.model == 'convnet' and params.convLayers[i].type in ['conv', 'pool']:    
            tempMean = T.mean(output, axis = (0, 2, 3))
            tempSTD = T.std(output, axis = 0) # std of each pixel in each output image 
            tempMax = T.max(output, axis = (0, 2, 3))
            tempSpars = T.mean(T.le(output, eps), axis = (0, 2, 3)) # as % units s.t. unit<0
            tempConst = T.mean(T.le(tempSTD, eps)) # as % units s.t. var(unit)<0               
        else:    
            tempMean = T.mean(output, axis = 0)
            tempSTD = T.std(output, axis = 0)
            tempMax = T.max(output, axis = 0)
            tempSpars = T.mean(T.le(output, eps), axis = 0) # as % units s.t. unit<0
            tempConst = T.mean(T.le(tempSTD, eps), axis = 0) # as % units s.t. var(unit)<0               
        
        if (key == 'rnoise' or key == 'rnstd') and 'addNoise' in params.rglrz and 'addNoise' not in params.rglrzPerMap:
             tempRNoise = layer.rglrzParam['addNoise']/T.mean(tempSTD, axis = (1, 2))
        else:
             tempRNoise = 0.*tempMean       

        # three cases of layers, with a different number of shared variables defined
        if params.model == 'convnet' and params.convLayers[i].type in ['pool', 'average', 'average+softmax']:
            if params.batchNorm and params.convLayers[i].bn:
                W = 0.*tempMean; b=layer.b; a=layer.a                
            else:    
                W=0.*tempMean; b=0.*tempMean; a=0.*tempMean  
        else:        
            W = layer.W; b=layer.b; a=layer.a

        
        i += 1
        stats = {'mean': T.mean(tempMean),
                 'std': T.mean(tempSTD),
                 'max': T.max(tempMax),
               'const': T.mean(tempConst), 
               'spars': T.mean(tempSpars),
               'wmean': T.mean(abs(W)),
                'wstd': T.std(W),
                'wmax': T.max(abs(W)),
              'rnoise': T.mean(tempRNoise),
               'rnstd': T.std(tempRNoise),
               'bias' : T.mean(b),
                'bstd': T.std(b),
                   'a': T.mean(a),
                'astd': T.std(a)}
               
        for key in filter(lambda name: name in stats.keys(), params.activTrack):
            netStats[key] += [stats[key]]
            
    for key in filter(lambda name: name in stats.keys(), params.activTrack):
        temp += [netStats[key]] 
    outStats = T.stacklists(temp)
    outStats.shape
                           
    return outStats
    
        
def stat_extract(modelStats, params, trackLayers):    
    stats = ['mean', 'std', 'max', 'const', 'spars', 'wmean', 'wstd', 'wmax',               
             'rnoise', 'rnstd', 'bias', 'bstd', 'a', 'astd']

    i = 0
    for param in filter(lambda name: name in stats, params.activTrack):
        tempParam = modelStats[i]
        trackLayers[param] = np.append(trackLayers[param], np.array([tempParam]), axis = 0)        
        i += 1
    return trackLayers            

    
        
#        for key in params.activTrack: allStats += [netStats[key]]
#        hStat = T.stacklists(allStats)
    