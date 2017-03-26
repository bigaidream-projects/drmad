"""
    This module is to collect and calculate updates for different theano function.

"""
import numpy as np
import theano
import theano.tensor as T
from lasagne.updates import sgd, momentum
from hypergrad import hypergrad
from pprint import pprint



def get_or_compute_grads(loss_or_grads, params):
    """Helper function returning a list of gradients
    Parameters
    ----------
    loss_or_grads : symbolic expression or list of expressions
        A scalar loss expression, or a list of gradient expressions
    params : list of shared variables
        The variables to return the gradients for
    Returns
    -------
    list of expressions
        If `loss_or_grads` is a list, it is assumed to be a list of
        gradients and returned as is, unless it does not match the length
        of `params`, in which case a `ValueError` is raised.
        Otherwise, `loss_or_grads` is assumed to be a cost expression and
        the function returns `theano.grad(loss_or_grads, params)`.
    Raises
    ------
    ValueError
        If `loss_or_grads` is a list of a different length than `params`, or if
        any element of `params` is not a shared variable (while we could still
        compute its gradient, we can never update it and want to fail early).
    """
    if any(not isinstance(p, theano.compile.SharedVariable) for p in params):
        raise ValueError("params must contain shared variables only. If it "
                         "contains arbitrary parameter expressions, then "
                         "lasagne.utils.collect_shared_vars() may help you.")
    if isinstance(loss_or_grads, list):
        if not len(loss_or_grads) == len(params):
            raise ValueError("Got %d gradient expressions for %d parameters" %
                             (len(loss_or_grads), len(params)))
        return loss_or_grads
    else:
        return theano.grad(loss_or_grads, params)

from collections import OrderedDict


def custom_mom(loss_or_grads, params, learning_rate, momentum=0.9):

    grads = get_or_compute_grads(loss_or_grads, params)
    updates = OrderedDict()

    for param, grad in zip(params, grads):
        value = param.get_value(borrow=True)
        velocity = theano.shared(np.zeros(value.shape, dtype=value.dtype),
                                 broadcastable=param.broadcastable)
        updates[velocity] = momentum * velocity - (1. - momentum) * grad
        updates[param] = param + learning_rate * velocity

    return updates


def update(params_theta, params_lambda, params_weight, loss, penalty, lossWithPenalty,
           lr_ele, lr_hyper, mom):
    # (1) phase 1
    # it's a simple MLP, we can use lasagne's routine to calc the updates
    update_ele = custom_mom(lossWithPenalty, params_theta, lr_ele, momentum=mom)

    dloss_dweight = T.grad(loss, params_weight)
    print "type 1,", type(dloss_dweight[0])

    try_1 = T.grad(penalty, params_weight)
    print "type 2", type(try_1[0])
    try_2 = T.grad(lossWithPenalty, params_weight)
    print "type 3", type(try_2[0])
    try_3 = [-grad for grad in dloss_dweight]
    print "type 4", type(try_3[0])

    # (2) calc updates for Phase 2
    # in this phase, no params is updated. Only dl_dtheta will be saved.
    update_valid = []
    grad_dloss_dweight = []
    for param, grad in zip(params_weight, dloss_dweight):
        save_grad = theano.shared(np.asarray(param.get_value() * 0., dtype='float32'),
                                  broadcastable=param.broadcastable,
                                  name='dl_dtheta_%s' % param.name)
        update_valid += [(save_grad, grad)]
        grad_dloss_dweight += [save_grad]

    return update_ele, update_valid, grad_dloss_dweight, dloss_dweight


def updates_hyper(params_lambda, params_weight, lossWithPenalty, grad_l_theta, share_var_dloss_dweight):
    # (3) meta_backward
    update_hyper = []
    HVP_weight = []
    HVP_lambda = []
    grad_valid_weight = []
    dLoss_dweight = T.grad(lossWithPenalty, params_weight)
    print "where's the tensorvariable comes from?", type(dLoss_dweight[0]), type(params_weight)
    for grad in grad_l_theta:
        print(type(grad))
        save_grad = theano.shared(np.asarray(grad, dtype=theano.config.floatX),
                                  name='grad_L_theta_xxx')
        grad_valid_weight += [save_grad]

    # print 'wtf', type(grad_valid_weight[0])
    dLoss_dweight = [-grad for grad in dLoss_dweight]
    dLoss_dweight = [-grad for grad in dLoss_dweight]
    HVP_weight_temp, HVP_lambda_temp = hypergrad(params_lambda, params_weight,
                                          dLoss_dweight, share_var_dloss_dweight)
    for param, grad in zip(params_weight, HVP_weight_temp):
        save_grad = theano.shared(np.asarray(param.get_value() * 0., dtype=theano.config.floatX),
                                  name='HVP_weight')
        update_hyper += [(save_grad, grad)]
        HVP_weight += [save_grad]

    for param, grad in zip(params_lambda, HVP_lambda_temp):
        save_grad = theano.shared(np.asarray(param.get_value() * 0., dtype=theano.config.floatX),
                                  name='HVP_lambda')
        update_hyper += [(save_grad, grad)]
        HVP_lambda += [save_grad]
    output_hyper_list = HVP_weight_temp + HVP_lambda_temp
    return update_hyper, output_hyper_list, share_var_dloss_dweight