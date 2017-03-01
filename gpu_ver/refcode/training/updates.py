import numpy as np
import theano
import theano.tensor as T
from training.hypergrad import hypergrad
from training.monitor import grad_monitor

from refcode.training.adaptive import adam


def remove_nans(x):
    return T.switch(T.isnan(x) + T.isinf(x), 0, x)

def scale_norm(x, threshold=3.):
    norm = T.sqrt(T.sum(x*x))
    multiplier = T.switch(norm < threshold, 1, threshold / norm)
    return x * multiplier

def clip_grad(x, threshold=10.):
    x = T.clip(x, -threshold, threshold)
    return x


def separateLR(params, sharedName, globalLR1, globalLR2):    
    ''' 
        Get learning rate from the name of the shared variable.    
    '''
    sharedName, _ = sharedName.split('_')
    customizedLR = globalLR1
    if (sharedName in params.rglrzLR.keys()):   
        customizedLR = globalLR2*params.rglrzLR[sharedName]

    return customizedLR      


def update_fun(param, grad, dataset, history, opt, learnParams, params):
    """
        Computing the update from gradient. 
        Adaptive step sizes, learning rate, momentum etc. 
    """
    epsilon = np.asarray(0.0, dtype=theano.config.floatX)

    # specification of learning rate, (hyper)param specific
    globalLR1, globalLR2, momentParam1, momentParam2 = learnParams
    assert dataset in ['T1', 'T2']
    lr = globalLR1 if dataset == 'T1' else separateLR(params, param.name, globalLR1, globalLR2) 
 
    # update with sgd
    if opt is None:
        updates = []
        if params.trackGrads:
            updates, trackGrads = grad_monitor(param, grad, updates, params, opt)
            other = [grad]
        else:    
            trackGrads = []
            other = [grad]                          
        up = - lr * grad

    # update with adam    
    else:
        up, updates, trackGrads, other = opt.up(param, grad, params, lr, dataset)

    # dictionary param to grad (first time around)
    if params.useT2 and dataset == 'T1':
        history['grad'][param] = grad
        history['up'][param] = up

    # momentum
    if params.use_momentum:
        oldup = theano.shared(np.asarray(param.get_value() * 0., dtype='float32'),
                              broadcastable=param.broadcastable,
                              name='oldup_%s' % param.name)
        momentParam = momentParam1 if dataset == 'T1' else momentParam2
        up += momentParam * oldup
        updates += [(oldup, up)]

    # new parameter
    newparam = param + up

    # min value (assumption: all hyperparams >= 0)
    if dataset == 'T2':
        newparam = T.maximum(epsilon, newparam)

    updates += [(param, newparam)]
    adamGrad = [other]
    return updates, trackGrads, adamGrad


def updates(model, params, global_lr1, global_lr2, moment_param1, moment_param2):
    
    """
        Computing updates of T1 and T2 parameters.
    
    Inputs:
        mlp :: model
        params :: specification of the model and training
        global_lr1, global_lr2 :: global learning rates for ele and hyper
        momentParam1, momentParam2 :: momentum parameters for T1 and T2
        phase :: external parameter in case of ifelse (currently not in use)        
        
    Outputs:
        update_ele :: update of T1 parameters and related shared variables
        update_hyper :: update of T2 parameters and related shared variables
        upnormdiff, debugs :: variable tracked for debugging
                        
    """

    # cost_ele is used in training elementary params (theta)
    # cost_valid is used in hyper / validation set, hyper params is denoted in (gamma)
    cost_ele = model.trainCost + model.penalty
    cost_valid = model.trainCost

    # dC/dtheta
    dele_dtheta = T.grad(cost_ele, model.paramsT1)
    dvalid_dtheta_temp = T.grad(cost_valid, model.paramsT1)
        
    # optimizers
    optimizer1 = adam() if params.opt1 in ['adam'] else None
    optimizer2 = adam() if params.opt2 in ['adam'] else None
    update_ele = [] if optimizer1 is None else optimizer1.initial_updates()
    update_hyper = [] if optimizer2 is None else optimizer2.initial_updates()

    update_valid, dvalid_dtheta, dvalid_dgamma, temp_ups, track_ele, track_hyper = [], [], [], [], [], []
    history_ele = {'grad': dict(), 'up': dict()}
    history_hyper = {'grad': dict(), 'up': dict()}
    learn_params = [global_lr1, global_lr2, moment_param1, moment_param2]

    """
        Updating T1 params

    """
    for param, grad in zip(model.paramsT1, dele_dtheta):

            grad = scale_norm(remove_nans(grad), threshold=3.)                
            ups, track, _ = update_fun(param, grad, 'T1',
                                       history_ele, optimizer1, learn_params, params)
            update_ele += ups
            track_ele += [track]

    """
        Updating T2 params

    """
    if params.useT2:     

        """
            Save grads C2T1 for the T2 update:
        """
        for param, grad in zip(model.paramsT1, dvalid_dtheta_temp):

            grad = scale_norm(remove_nans(grad), threshold=3.)
            grad = clip_grad(grad, threshold=10.)
            save_grad = theano.shared(np.asarray(param.get_value() * 0., dtype='float32'),
                                     broadcastable=param.broadcastable,
                                     name='gradC2T1_%s' % param.name)
            update_valid += [(save_grad, grad)]
            dvalid_dtheta += [save_grad]

        """
            If gradient dC2/dT1 is also estimated with adam
        """
        if params.avC2grad in ['adam', 'momentum']:
            #dvalid_dtheta = T.grad(cost_valid, mlp.paramsT1)
            if params.avC2grad == 'adam': opt3 = adam()
            else: opt3 = None
            temp_ups = [] if opt3 is None else opt3.initial_updates()

            newC2 = []
            grad = scale_norm(remove_nans(grad), threshold=3.)
            grad = clip_grad(grad, threshold=10.)
            for param, grad in zip(model.paramsT1, dvalid_dtheta):
                temp_up, _, newGrad = update_fun(param, T.reshape(grad, param.shape), 'T1',
                                                history_hyper, opt3, learn_params, params)
                temp_ups += temp_up[:-1]
                newC2 += newGrad
            dvalid_dtheta = newC2

        paramsT2, dvalid_dgamma = hypergrad(model.paramsT1, model.paramsT2, dvalid_dtheta,
                                       model.trainCost, model.trainCost, model.penalty)

        for param, grad in zip(model.paramsT2, dvalid_dgamma):
            paramName, _ = param.name.split('_')
            if params.decayT2 > 0. and paramName not in ['L2', 'L1']:
                grad += params.decayT2*param 

            grad = scale_norm(remove_nans(grad), threshold=3.) 
            grad = clip_grad(grad, threshold=10.)                              
            temp_up, track, _ = update_fun(param, T.reshape(grad, param.shape),'T2',
                                          {}, optimizer2, learn_params, params)
            update_hyper += temp_up
            track_hyper += [track]
        print "Parameters ",
        print ", ".join([p.name for p in model.paramsT2]),
        print "are trained on hyper set"
                         
    # monitored variables for output                         
    if (not params.useT2) and params.trackGrads:
        debugs = track_ele
    elif params.trackGrads:
        debugs = track_ele + track_hyper
    else:
        debugs = []

    return update_ele, update_valid, update_hyper+temp_ups, debugs
    
    
