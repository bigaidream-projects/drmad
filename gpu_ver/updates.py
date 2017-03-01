"""
    This module is to collect and calculate updates for different theano function.

"""
import numpy as np
import theano
import theano.tensor as T
from lasagne.updates import sgd
from hypergrad import hypergrad
from pprint import pprint


def update(params_theta, params_lambda, params_weight, loss, penalty, lossWithPenalty,
           lr_ele, lr_hyper):
    # (1) phase 1
    # it's a simple MLP, we can use lasagne's routine to calc the updates
    update_ele = sgd(lossWithPenalty, params_theta, lr_ele)

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