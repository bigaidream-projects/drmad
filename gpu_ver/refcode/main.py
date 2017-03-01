"""
    This training script is just a practice on mnist using Theaao.

"""

import warnings
from time import time

warnings.filterwarnings("ignore", category=DeprecationWarning)
import numpy as np
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import theano.sandbox.rng_mrg
import theano
import argparse
from setup import setup
theano.config.exception_verbosity = 'high'
theano.config.floatX = 'float32'
theano.config.optimizer='fast_compile'
# theano.config.device='GPU0'

"""
    Params setting

"""
parser = argparse.ArgumentParser()
parser.add_argument('-v', '--verbose',
                    type=bool, default=True)
parser.add_argument('-m', '--model',
                    type=str, choices=['mlp', 'convnet'], default='mlp')
parser.add_argument('-d', '--dataset',
                    choices=['mnist', 'cifar10'], default='mnist')
parser.add_argument('-r', '--ratioValid', help='the ratio of valid set to train set',
                    type=float, default=0.2)
parser.add_argument('--bn', help='use BatchNorm or not',
                    type=bool, default=False)
parser.add_argument('--predata', help='to load from preprocess data or not',
                    type=bool, default=False)

"""
    Setup

"""
from preprocess.read_preprocess import read_preprocess
from refcode.models.mlp import mlp
from refcode.models.convnet import convnet
from models.layers.batchnorm import update_bn
from refcode.training import lr_schedule
from refcode.training import updates


def run_exp(replace_params):
    params = setup(replace_params)

    """
        load data

    """
    if params.predata is False:
        X_elementary, Y_elementary, X_hyper, Y_hyper, X_valid, Y_valid, X_test, Y_test = read_preprocess(params=params)
        np.savez(params.predataName, X_elementary=X_elementary, Y_elementary=Y_elementary, X_hyper=X_hyper,
             Y_hyper=Y_hyper, X_v=X_valid, Y_v=Y_valid, X_test=X_test, Y_test=Y_test)
    else:
        tmpload = np.load(params.predataName)
        X_elementary, Y_elementary, X_hyper, Y_hyper, X_valid, Y_valid, X_test, Y_test = \
            tmpload['X_elementary'], tmpload['Y_elementary'], tmpload['X_hyper'], tmpload['Y_hyper'],\
            tmpload['X_v'], tmpload['Y_v'], tmpload['X_test'], tmpload['Y_test']

    """
        Build Theano functions

    """
    # random numbers
    rng = np.random.RandomState(params.seed)
    rstream = RandomStreams(rng.randint(params.seed + 1) + 1)

    # inputs
    useRegular = T.fscalar('useRegular')
    bnPhase = T.fscalar('bnPhase')
    if params.model == 'convnet':
        x = T.ftensor4('x')
    elif params.model == 'mlp':
        x = T.matrix('x')
    else:
        raise AttributeError
    y = T.ivector('y')
    global_lr1 = T.fscalar('global_lr1')
    global_lr2 = T.fscalar('global_lr2')
    moment_ele = T.fscalar('moment_ele')
    moment2 = T.fscalar('moment2')

    # network
    if params.model == 'convnet':
        model = convnet(rng=rng, rstream=rstream, x=x, wantOut=y,
                        params=params, useRegular=useRegular, bnPhase=bnPhase)
    else:
        model = mlp(rng=rng, rstream=rstream, x=x, y=y,
                    setting=params)

    update_ele, update_valid, update_hyper, debug = updates(model=model, params=params,
                                                            global_lr1=global_lr1, global_lr2=global_lr2,
                                                            moment_param1=moment_ele, moment_param2=moment2)
    updateBN = []
    if params.batchNorm:
        for param, up in zip(model.paramsBN, model.updateBN):
            updateBN += [(param, up)]

    func_elementary = theano.function(
        inputs=[x, y, global_lr1, moment_ele, useRegular, bnPhase],
        outputs=[model.trainCost, model.guessLabel] + debug,
        updates=update_ele + updateBN,
        # mode=theano.compile.MonitorMode(post_func=detect_nan),
        on_unused_input='ignore',
        allow_input_downcast=True)

    # this function should only be execute once
    func_hyper_valid = theano.function(
        inputs=[x, y, useRegular, bnPhase],
        outputs=[model.trainCost, model.guessLabel] + debug,
        updates=update_valid,
        # mode=theano.compile.MonitorMode(post_func=detect_nan),
        on_unused_input='ignore',
        allow_input_downcast=True)

    func_hyper = theano.function(
        inputs=[x, y, global_lr1, moment_ele, global_lr2, moment2, useRegular, bnPhase],
        outputs=[model.trainCost, model.guessLabel] + debug,
        updates=update_hyper,
        # mode=theano.compile.MonitorMode(post_func=detect_nan),
        on_unused_input='ignore',
        allow_input_downcast=True)

    func_eval = theano.function(
        inputs=[x, y, useRegular, bnPhase],
        outputs=[model.trainCost, model.guessLabel, model.netStats],
        on_unused_input='ignore',
        allow_input_downcast=True)

    evaluateBN = theano.function(
        inputs=[x, useRegular, bnPhase],
        updates=updateBN,
        on_unused_input='ignore',
        # mode=theano.compile.MonitorMode(post_func=detect_nan),
        allow_input_downcast=True)

    """
        Initialization

    """

    # initialize
    loopOver = range(params.nLayers)   # layers to be read from
    n_ele, n_valid, n_test = X_elementary.shape[0], X_valid.shape[0], X_test.shape[0]
    n_batch_ele = n_ele / params.batchSize1
    # permutations
    test_perm, ele_perm = range(0, n_test), range(0, n_ele)
    np.random.shuffle(test_perm)

    # tracking
    # (1) best results
    bestVal = 1.
    bestValTst = 1.
    # (2) errors
    temp_err_ele, tempError2, temp_cost_ele, tempCost2 = [], [], [], []
    t1Error, t2Error, validError, testError = [], [], [], []
    t1Cost, t2Cost, penaltyCost, validCost, testCost = [], [], [], [], []
    # (3) activation statistics (per layer)
    trackTemplate = np.empty((0, params.nLayers), dtype=object)
    trackLayers = {}
    for stat in params.activTrack: trackLayers[stat] = trackTemplate
    # (4) penalty, noise, activation parametrization (per layer)
    penalList = ['L1', 'L2', 'Lmax', 'LmaxCutoff', 'LmaxSlope', 'LmaxHard']
    noiseList = ['addNoise', 'inputNoise', 'dropOut', 'dropOutB']
    sharedNames = [p.name for p in model.paramsT1] + [p.name for p in model.paramsT2]
    print sharedNames

    trackPenal = {}
    trackPenalSTD = {}
    trackNoise = {}
    trackNoiseSTD = {}
    trackGrads = {}
    track1stFeatures = []

    trackRglrzTemplate = np.empty((0, len(loopOver)), dtype=object)
    for param in params.rglrz:
        if param in penalList:
            trackPenal[param] = trackRglrzTemplate
            trackPenalSTD[param] = trackRglrzTemplate
        if param in noiseList:
            trackNoise[param] = trackRglrzTemplate
            trackNoiseSTD[param] = trackRglrzTemplate
    # (5) other
    trackLR1, trackLR2 = [], []

    params.halfLife = params.halfLife * 10000. / (params.maxEpoch * n_batch_ele)
    print('number of iters total', params.maxEpoch * n_batch_ele)
    print('number of iters within epoch', n_batch_ele)

    """
        Phase 1: meta-forward

    """
    last_iter = params.maxEpoch * n_batch_ele - 1

    t_start = time()

    # save the model parameters into theta_initial
    theta_initial = {}
    for w in model.paramsT1:
        theta_initial[w] = w.get_value()

    for i in range(0, params.maxEpoch * n_batch_ele):

        curr_epoch = i / n_batch_ele
        curr_batch = i % n_batch_ele

        """
            Learning rate and momentum schedules.

        """
        t = 1. * i / (params.maxEpoch * n_batch_ele)
        lr_ele = np.asarray(params.learnRate1 *
                         lr_schedule(fun=params.learnFun1, var=t, halfLife=params.halfLife, start=0),
                         theano.config.floatX)
        moment_ele = np.asarray(params.momentum1[1] - (params.momentum1[1] - (params.momentum1[0])) *
                             lr_schedule(fun=params.momentFun, var=t, halfLife=params.halfLife, start=0),
                             theano.config.floatX)

        if curr_batch == 0:
            np.random.shuffle(ele_perm)

        """
            Update

        """
        sample_idx_ele = ele_perm[(curr_batch * params.batchSize1):((curr_batch + 1) * params.batchSize1)]
        batch_x, batch_y = X_elementary[sample_idx_ele], Y_elementary[sample_idx_ele]
        res = func_elementary(batch_x, batch_y, lr_ele, moment_ele, 1, 0)
        (cost_ele, pred_ele, debugs) = (res[0], res[1], res[2:])
        temp_err_ele += [1. * sum(batch_y != pred_ele) / params.batchSize1]
        temp_cost_ele += [cost_ele]

        if np.isnan(cost_ele):
            print 'NANS', cost_ele

        """
            Evaluate

        """
        if params.verbose or (curr_epoch == n_batch_ele - 1):
            if (params.batchNorm and (curr_epoch > 1)) \
                    and ((curr_epoch % params.evaluateTestInterval) == 0 or i == last_iter) \
                    and params.testBN != 'lazy':
                model = update_bn(model, params, evaluateBN, X_elementary, Y_elementary)

            if params.model == 'mlp':
                n_eval = 5000
            else:
                n_eval = 1000

            eval_error = 0.
            temp_idx = test_perm[:n_eval]
            batch_x, batch_y = X_test[temp_idx], Y_test[temp_idx]
            eval_loss, y_test, stats = func_eval(batch_x, batch_y, 0, 1)
            eval_error = 1. * sum(y_test != batch_y) / n_eval

            print("Epoch %d, batch %d, time = %ds, eval_err = %.2f, eval_loss = %.4f" %
                  (curr_epoch, curr_batch, time() - t_start, eval_error * 100, eval_loss))

    # save the model parameters after T1 into theta_final
    theta_final = {}
    for w in model.paramsT1:
        theta_final[w] = w.get_value()
    """
        Phase 2: Validation on Hyper set

    """
    n_hyper = X_hyper.shape[0]
    n_batch_hyper = n_hyper / params.batchSize2
    hyper_perm = range(0, n_hyper)
    np.random.shuffle(hyper_perm)

    err_valid = 0.
    cost_valid = 0.
    t_start = time()
    for i in range(0, n_batch_hyper):

        sample_idx = hyper_perm[(i * params.batchSize2):((i + 1) * params.batchSize2)]
        batch_x, batch_y = X_elementary[sample_idx], Y_elementary[sample_idx]
        res = func_hyper_valid(batch_x, batch_y, 0, 1)
        (valid_cost, pred_hyper, debugs) = (res[0], res[1], res[2:])
        err_valid += 1. * sum(batch_y != pred_hyper) / params.batchSize2
        cost_valid += valid_cost

    err_valid /= n_batch_hyper
    cost_valid /= n_batch_hyper

    print("Valid on Hyper Set: time = %ds, valid_err = %.2f, valid_loss = %.4f" %
          (time() - t_start, err_valid * 100, cost_valid))

    """
        Phase 3: meta-backward (TODO)

    """
    # if params.useT2:
    #     curr_batch_hyper = 0
    #     n_hyper = X_hyper.shape[0]
    #     train2Perm = range(0, n_hyper)
    #     n_batch_hyper = n_hyper / params.batchSize2
    #     if curr_batch_hyper == n_batch_hyper - 1:
    #         np.random.shuffle(train2Perm)
    #         curr_batch_hyper = 0
    #
    #     for i in range(params.maxEpoch * n_batch_ele - 1, 0, -1): # TODO: reverse-mode
    #         # make batches
    #         sample_idx_ele = ele_perm[(curr_batch * params.batchSize1):
    #         ((curr_batch + 1) * (params.batchSize1))]
    #         sampleIndex2 = train2Perm[(curr_batch_hyper * params.batchSize2):
    #         ((curr_batch_hyper + 1) * (params.batchSize2))]
    #
    #         if (i % params.T1perT2 == 0) and (i >= params.triggerT2):
    #
    #             res = func_hyper_valid(X_hyper[sampleIndex2], Y_hyper[sampleIndex2], lr_ele, moment_ele, 0, 1)
    #             (c2, y2, debugs) = (res[0], res[1], res[2:])
    #
    #             res = func_hyper(X_elementary[sample_idx_ele], Y_elementary[sample_idx_ele], lr_ele, moment_ele, lr_hyper, moment2, 1, 0)
    #             (cost_ele, pred_ele, debugs) = (res[0], res[1], res[2:])
    #
    #             tempError2 += [1. * sum(Y_hyper[sampleIndex2] != y2) / params.batchSize2]
    #             tempCost2 += [c2]
    #             curr_batch_hyper += 1
    #             if np.isnan(cost_ele): print 'NANS in part 2!'
    #             if np.isnan(c2): print 'NANS in part 1!'
    #
    #         else:
    #             res = func_elementary(X_elementary[sample_idx_ele], Y_elementary[sample_idx_ele], lr_ele, moment_ele, 1, 0)
    #             (cost_ele, pred_ele, debugs) = (res[0], res[1], res[2:])
    #
    #         temp_err_ele += [1. * sum(Y_elementary[sample_idx_ele] != pred_ele) / params.batchSize1]
    #         temp_cost_ele += [cost_ele]
    #         if np.isnan(cost_ele): print 'NANS!'
    #
    #         # refactor it
    #         t2Error += [np.mean(tempError2)]
    #         t2Cost += [np.mean(tempCost2)]


if __name__ == '__main__':
    args = parser.parse_args()
    print(args)
    print('Finish parse!')
    run_exp(replace_params=args)
