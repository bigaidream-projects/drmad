import numpy as np
import theano
import theano.tensor as T

from refcode.training.adaptive import adam

zero = np.float32(0.)
step = 0.001

def update_fun(param, grad, penaltyparam, dataset, history, opt, params, globalLR1, globalLR2, momentParam1, momentParam2):

        epsilon = np.asarray(0.0, dtype=theano.config.floatX)
        def separateLR(params, sharedName, globalLR1, globalLR2):
            sharedName = sharedName[:-2];customizedLR = globalLR2
            if (sharedName in params.rglrzLR.keys()) or (not params.adaptT2LR):
                customizedLR = globalLR2*params.rglrzLR[sharedName]
            return customizedLR      
 
        assert dataset in ['T1', 'T2']
        lr = globalLR1 if dataset == 'T1' else separateLR(params, param.name, globalLR1, globalLR2) 
 
        # Standard update
        if opt is None:
            updates = []
            if params.trackGrads:
                old_grad = theano.shared(np.asarray(param.get_value() * 0., dtype='float32'),
                                    broadcastable=param.broadcastable,
                                    name='oldgrad_%s' % param.name)
                updates += [(old_grad, grad)]
                grad_mean = T.mean(T.sqrt(grad**2))
                grad_rel = T.mean(T.sqrt((grad/(param+1e-12))**2))
                grad_angle = T.sum(grad*old_grad)/(T.sqrt(T.sum(grad**2))*T.sqrt(T.sum(old_grad**2))+1e-12) 
                check = T.stacklists([grad_mean, grad_rel, grad_angle])
                other = [grad]
            else:    
                check = grad
                other = [grad]
                          
            up = - lr * grad
        else:
            up, updates, check, other = opt.up(param, grad, params, lr=lr, dataset=dataset)

        # dictionary param to grad (first time around)
        if params.useT2 and dataset == 'T1':
            history['grad'][param] = grad
            history['up'][param] = up
        # add momentum to update
        if params.use_momentum:
            oldup = theano.shared(np.asarray(param.get_value() * 0., dtype='float32'),
                                  broadcastable=param.broadcastable,
                                  name='oldup_%s' % param.name)
            momentParam = momentParam1 if dataset == 'T1' else momentParam2
            up += momentParam * oldup
            updates += [(oldup, up)]

        # New parameter
        newparam = param + up
        # min value  |  NOTE assumption: all hyperparams can only be positive
        if dataset == 'T2':
            newparam = T.maximum(epsilon, newparam)

        updates += [(param, newparam)]
        paramUpPair = [(param, check)]
        adamGrad = [other]

        return updates, paramUpPair, adamGrad

# ---------------------------------------------------------------------------------------- FD MEMORY
class fd_memory(object):
    def __init__(self, params, mlp, updateC1T1=None, updateC2T1=None, gradC1T2=None):
    
        if updateC1T1 is None:
            updateC1T1 = []; updateC2T1 = []; gradC1T2 = []
            for param in mlp.paramsT1:
                    uC1 = theano.shared(np.float32(param.get_value()) * zero, name="old_uc1_%s" % param.name)
                    uC2 = theano.shared(np.float32(param.get_value()) * zero, name="old_uc2_%s" % param.name)
                    updateC1T1 += [uC1]; updateC2T1 += [uC2]                              
            for param in mlp.paramsT2:
                    gT2 = theano.shared(np.float32(param.get_value()) * zero, name="old_ut2_%s" % param.name)
                    gradC1T2 += [gT2]    

        self.updateC1T1 = updateC1T1
        self.updateC2T1 = updateC2T1
        self.gradC1T2 = gradC1T2

# ------------------------------------------------------------------------------------------ PHASE 1            
def fd1(mlp, fdm, params, globalLR1, globalLR2, momentParam1, momentParam2):
    
    # gradient of T1 ----------------------------------- GRADS
    cost1 = mlp.classError1 + mlp.penalty
    gradT1 = T.grad(cost1, mlp.paramsT1)
    gradT1reg = T.grad(cost1, mlp.paramsT2)
        
    # take opt from Adam?
    if params.opt1 in ['adam']: opt1 = adam()
    else: opt1 = None
    if params.opt2 in ['adam']: opt2 = adam()
    else: opt2 = None    

    updateT1 = [] if opt1 is None else opt1.initial_updates()
    updateT2 = [] if opt2 is None else opt2.initial_updates()

    onlyT1param = []
    history = {'grad': dict(), 'up': dict()}

    assert len(mlp.paramsT1) == len(gradT1) 
    assert len(mlp.paramsT1) == len(fdm.updateC1T1) 
    assert len(mlp.paramsT2) == len(gradT1reg) 
    assert len(mlp.paramsT2) == len(fdm.gradC1T2) 


    for param, grad, uC1 in zip(mlp.paramsT1, gradT1, fdm.updateC1T1):
                tempUp, tempPair, _ = update_fun(param, grad, mlp.penaltyMaxParams.get(param, None),
                                                'T1', history, opt1, params,
                                                globalLR1, globalLR2, momentParam1, momentParam2)
                updateT1 += tempUp
                onlyT1param += tempPair                

                newparam = tempUp[-1][-1] 
                just_up = newparam - param
                updateT1 += [(uC1, just_up)]                                 

     # save grad T2 of C1 as shared (2) in gradT1reg
    for param, grad, gT2 in zip(mlp.paramsT2, gradT1reg, fdm.gradC1T2):
                updateT2 += [(gT2, grad)]                           
                    
    
    debugs = [check for (_, check) in onlyT1param]                        
    return updateT1 + updateT2, debugs#, T2_grads


# ------------------------------------------------------------------------------------------ PHASE 2                
def fd2(mlp, fdm, params, globalLR1, globalLR2, momentParam1, momentParam2):

    cost2 = mlp.classError2
    gradC2 = T.grad(cost2, mlp.paramsT1)  

    tempUps = []      
    history = {'grad': dict(), 'up': dict()}


    if params.avC2grad in ['adam', 'momentum']:
                if params.avC2grad == 'adam': opt3 = adam()
                else: opt3 = None
                tempUps = [] if opt3 is None else opt3.initial_updates()
        
                newC2 = []
                for param, grad in zip(mlp.paramsT1, gradC2):            
                    tempUp, _, newGrad = update_fun(param, T.reshape(grad, param.shape), None, 
                                                   'T1', history, opt3, params,
                                                   globalLR1, globalLR2, momentParam1, momentParam2)
                    newC2 += newGrad
                    tempUps += tempUp[:-1]
                gradC2 = newC2


    updateT1 = []; updateT2 = []
    # save grad W of C2 as shared (3), update W - (1) + (3)
    for param, grad, uC1, uC2 in zip(mlp.paramsT1, gradC2, fdm.updateC1T1, fdm.updateC2T1):                               
                updateT1 += [(uC2, - step*globalLR1*grad)]                
                updateT1 += [(param, param - uC1 - step*globalLR1*grad)]    



    return updateT1 + updateT2 + tempUps
    

# ------------------------------------------------------------------------------------------ PHASE 3            
def fd3(mlp, fdm, params, globalLR1, globalLR2, momentParam1, momentParam2):

    cost1 = mlp.classError1 + mlp.penalty
    gradT1reg = T.grad(cost1, mlp.paramsT2)        

    updateT1 = []; updateT2 = []; onlyT2param = []    
    # take opt from Adam?
    if params.opt2 in ['adam']: opt2 = adam()
    else: opt2 = None    

    # update W - (1) + (3)            
    for param, uC1, uC2 in zip(mlp.paramsT1, fdm.updateC1T1, fdm.updateC2T1):                               
        updateT1 += [(param, param + uC1 - uC2)]

    # compute grad T2 of C1,  update T2 - [(4) - (2) ] / lr1
    for param, grad, gT2 in zip(mlp.paramsT2, gradT1reg, fdm.gradC1T2):   
        if params.T2onlySGN:
           grad_proxi = T.sgn((grad - gT2)/step*globalLR1)
        else:
           grad_proxi = (grad - gT2)/step*globalLR1
            
        tempUp, tempPair, _ = update_fun(param, T.reshape(grad_proxi, param.shape), None,
                              'T2', {}, opt2, params,
                              globalLR1, globalLR2, momentParam1, momentParam2)
        updateT2 += tempUp
        onlyT2param += tempPair        
     
     
    debugs = [check for (_, check) in onlyT2param]  
    return updateT1 + updateT2, debugs
    

    