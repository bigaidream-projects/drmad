import theano.tensor as T
import theano
# TODO: more robust implementation: address problems with Lop 

# hypers influencing only penalty term (cause Theano)
penalty_list = ['L1', 'L2', 'Lmax', 'LmaxSlope', 'LmaxCutoff', 'LmaxHard']
# hypers influencing only NLL (cause Theano)
noise_list = ['addNoise', 'inputNoise']


def hypergrad(params_ele, params_hyper,
              dvalid_dtheta,
              loss_ele, loss_hyper, loss_ele_penalty=0.):
    
    """ Function defining the hypergradients: gradients of validation cost
        with respect to various hyperparameters.     
    
        The function is separating penalty hyperparameters 
        (which is assumed to depend only on w) from noise and other hyperparameters,
        due to otherwise dependancy errors in the Lop operator.
        
        Inputs: 
        
        paramsT1, paramsT2 :: T1 and T2 parameters
        c1, c2 :: cross-entropy on training and validation set
        p1, p2 :: penalty terms on training and validation set (p2 assumed 0)
        
    """
    # initializations
    reg_penalty, reg_noise, grad_penalty, grad_noise, w, dvalid_dw = [], [], [], [], [], []

    # separate different types of parameters
    for regular in params_hyper:
        reg_type, _ = regular.name.split('_')
        if reg_type in penalty_list:
            reg_penalty += [regular]
        elif reg_type in noise_list:
            reg_noise += [regular]
        else:
            print 'Hypergrad not implemented for ', reg_type

    # separate weight parameters and gradients
    for (param, grad) in zip(params_ele, dvalid_dtheta):
        paramType, _ = param.name.split('_')
        if paramType == 'W':
            w += [param]
            dvalid_dw += [grad]
                     
    # hyper-gradients        
    if reg_penalty:
        dpenalty_dw = T.grad(loss_ele_penalty, w)
        dpenalty_dw = [-grad for grad in dpenalty_dw]
        grad_penalty = T.Lop(dpenalty_dw, reg_penalty, dvalid_dw)
    if reg_noise:
        dele_dtheta = T.grad(loss_ele, params_ele)
        dele_dtheta = [-grad for grad in dele_dtheta]
        grad_noise = T.Lop(dele_dtheta, reg_noise, dvalid_dtheta)

    # outputs     
    params_hyper = reg_penalty + reg_noise
    dvalid_dgamma = grad_penalty + grad_noise

    return params_hyper, dvalid_dgamma


def L_hvp_meta(params_ele, params_hyper, pseudo_params_ele, vec, batchx, batchy):
    """
    :param params_ele: elementary params
    :param params_hyper: hyper params
    :param pseudo_params_ele: the psed
    :param vec: a vector multiple to the hessian, could be learning rate vec or momentum vec
    :param batchx: data x of this iteration
    :param batchy: data y of this iteration
    :return: gradient w.r.t. hyper params
    """

    reg_params_penalty, reg_params_noise, grad_penalty, grad_noise, w, dvalid_dw = [], [], [], [], [], []

    # forward to obtain loss & gradients
    loss_ele, loss_ele_penalty = L_hvp_meta_unsafe(batchx, batchy, 1, 0)

    # separate different types of parameters
    for regular in params_hyper:
        reg_type, _ = regular.name.split('_')
        if reg_type in penalty_list:
            reg_params_penalty += [regular]
        elif reg_type in noise_list:
            reg_params_noise += [regular]
        else:
            print 'Hypergrad not implemented for ', reg_type

    # VJ = T.Lop(y, W, v), to calc v * dy/dW
    if reg_params_penalty:
        dpenalty_dw = T.grad(loss_ele_penalty, w)
        dpenalty_dw = [-grad for grad in dpenalty_dw] # dpenalty_dw might be calc through `meta_backward_ele()`,
                                                      # as you like, discuss it later
        grad_penalty = T.Lop(dpenalty_dw, reg_params_penalty, vec)

    # if reg_params_noise:
    #     dele_dtheta = T.grad(loss_ele, params_ele)
    #     dele_dtheta = [-grad for grad in dele_dtheta]
    #     grad_noise = T.Lop(dele_dtheta, reg_params_noise, dL_dtheta)

    # outputs
    params_hyper = reg_params_penalty + reg_params_noise
    dvalid_dgamma = grad_penalty + grad_noise

    return dvalid_dgamma


def L_hvp_meta_unsafe(params_ele, params_hyper, pseudo_params_ele, batchx, batchy, x, y, loss):
    """
    :param params_ele: elementary params
    :param params_hyper: hyper params
    :param pseudo_params_ele: the psed, a dictionary whose keys are elements in params_ele
    :param batchx: data x of this iteration
    :param batchy: data y of this iteration
    :param x: variable x of the model
    :param y: variable y of the model
    :param loss: symbol of loss function expression
    :return: gradient w.r.t. hyper params at pseudo_params_ele

    Attention please! In order to save the memory, the value of params_ele
    would be replaced by the values of pseudo_params_ele. SAVE ve the values of
    weights before calling me!
    """

    reg_params_penalty, reg_params_noise, grad_penalty, grad_noise, w, dvalid_dw = [], [], [], [], [], []
    # replace the params
    for param in params_ele:
        param.set_value(pseudo_params_ele[param])

    # separate different types of parameters
    for regular in params_hyper:
        reg_type, _ = regular.name.split('_')
        if reg_type in penalty_list:
            reg_params_penalty += [regular]
        elif reg_type in noise_list:
            reg_params_noise += [regular]
        else:
            print 'Hypergrad not implemented for ', reg_type
    # get gradient w.r.t. hyper params
    if reg_params_penalty:
        dloss_dpenalty = T.grad(loss, penalty_list)

    # forward & backward to obtain gradients
    meta_fwbw_ele = theano.function([x, y], dloss_dpenalty)
    grad_penalty = meta_fwbw_ele(batchx, batchy)

    # if reg_params_noise:
    #     dele_dtheta = T.grad(loss_ele, params_ele)
    #     dele_dtheta = [-grad for grad in dele_dtheta]
    #     grad_noise = T.Lop(dele_dtheta, reg_params_noise, dL_dtheta)

    # outputs
    params_hyper = reg_params_penalty + reg_params_noise
    dvalid_dgamma = grad_penalty + grad_noise

    return dloss_dpenalty, dvalid_dgamma