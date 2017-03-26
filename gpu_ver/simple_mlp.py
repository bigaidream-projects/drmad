"""
    This training script is just a practice on mnist using Theano.

"""
from time import time
import theano
import theano.tensor as T
import numpy as np
from preprocess.read_preprocess import read_preprocess
from args import setup
from models import MLP, ConvNet
from updates import update, updates_hyper
import time as TIME

theano.config.exception_verbosity = 'high'
theano.config.floatX = 'float32'
theano.config.optimizer='fast_compile'


def run_exp(args, update_lambda, fix_weight):

    if args.predata is False:
        X_elementary, Y_elementary, X_hyper, Y_hyper, X_valid, Y_valid, X_test, Y_test = read_preprocess(params=args)
        np.savez(args.processedDataName, X_elementary=X_elementary, Y_elementary=Y_elementary, X_hyper=X_hyper,
             Y_hyper=Y_hyper, X_v=X_valid, Y_v=Y_valid, X_test=X_test, Y_test=Y_test)
    else:
        tmpload = np.load(args.processedDataName)
        X_elementary, Y_elementary, X_hyper, Y_hyper, X_valid, Y_valid, X_test, Y_test = \
            tmpload['X_elementary'], tmpload['Y_elementary'], tmpload['X_hyper'], tmpload['Y_hyper'],\
            tmpload['X_v'], tmpload['Y_v'], tmpload['X_test'], tmpload['Y_test']

    """
        Build Theano functions

    """

    if args.model == 'convnet':
        x = T.ftensor4('x')
    elif args.model == 'mlp':
        x = T.matrix('x')
    else:
        raise AttributeError
    y = T.matrix('y')
    lr_ele = T.fscalar('lr_ele')

    lr_ele_true = np.array(args.lrEle, theano.config.floatX)
    mom = 0.95
    lr_hyper = T.fscalar('lr_hyper')
    grad_valid_weight = T.tensor4('grad_valid_weight')

    if args.model == 'mlp':
        model = MLP(x=x, y=y, args=args)
    elif args.model == 'convnet':
        model = ConvNet(x=x, y=y, args=args)

        if args.dataset == 'mnist':
            nc = 1
            nPlane = 28
        else:
            nc = 3
            nPlane = 32
        X_elementary = X_elementary.reshape(-1, nc, nPlane, nPlane)
        X_hyper = X_hyper.reshape(-1, nc, nPlane, nPlane)
        X_valid = X_valid.reshape(-1, nc, nPlane, nPlane)
        X_test = X_test.reshape(-1, nc, nPlane, nPlane)
    else:
        raise AttributeError

    update_ele, update_valid, output_valid_list, share_var_dloss_dweight = update(model.params_theta, model.params_lambda, model.params_weight,
                                      model.loss, model.penalty, model.lossWithPenalty,
                                      lr_ele, lr_hyper, mom)

    if update_lambda:
        for up, origin in zip(update_lambda, model.params_lambda):
            origin.set_value(np.array(up))
            boo = origin.get_value()
            # print 'update', type(up), type(boo), boo[1]
            # TIME.sleep(20)

    if fix_weight:
        for fix, origin in zip(fix_weight, model.params_weight):
            origin.set_value(np.array(fix))
    else:
        fix_weight = []
        for origin in model.params_weight:
            fix_weight.append(origin.get_value())

    # Phase 1
    func_elementary = theano.function(
        inputs=[x, y, lr_ele],
        outputs=[model.lossWithPenalty, model.prediction],
        updates=update_ele,
        on_unused_input='ignore',
        allow_input_downcast=True)

    func_eval = theano.function(
        inputs=[x, y],
        outputs=[model.loss, model.prediction],
        on_unused_input='ignore',
        allow_input_downcast=True)

    # Phase 2
    # actually, in the backward phase
    func_hyper_valid = theano.function(
        inputs=[x, y],
        outputs=[model.loss, model.prediction] + output_valid_list,
        updates=update_valid,
        on_unused_input='ignore',
        allow_input_downcast=True)


    """
         Phase 1: meta-forward

    """
    X_mix = np.concatenate((X_valid, X_test), axis=0)
    Y_mix = np.concatenate((Y_valid, Y_test), axis=0)
    print X_valid.shape, X_mix.shape
    X_valid, Y_valid = X_mix[:len(X_mix) / 2], Y_mix[:len(X_mix) / 2]
    X_test, Y_test = X_mix[len(X_mix) / 2:], Y_mix[len(X_mix) / 2:]
    n_ele, n_valid, n_test = X_elementary.shape[0], X_valid.shape[0], X_test.shape[0]
    # TODO: remove this override
    n_ele = 20000
    X_elementary, Y_elementary = X_elementary[:n_ele], Y_elementary[:n_ele]

    print "# of ele, valid, test: ", n_ele, n_valid, n_test
    n_batch_ele = n_ele / args.batchSizeEle
    test_perm, ele_perm = range(0, n_test), range(0, n_ele)
    last_iter = args.maxEpoch * n_batch_ele - 1
    temp_err_ele = []
    temp_cost_ele = []
    eval_loss = 0.
    t_start = time()

    iter_index_cache = []

    # save the model parameters into theta_initial
    theta_initial = []
    for i, w in enumerate(model.params_theta):
        theta_initial.append(w.get_value())

    for i in range(0, args.maxEpoch * n_batch_ele):
        curr_epoch = i / n_batch_ele
        curr_batch = i % n_batch_ele

        """
            Learning rate and momentum schedules.

        """
        t = 1. * i / (args.maxEpoch * n_batch_ele)

        """
            Update

        """
        sample_idx_ele = ele_perm[(curr_batch * args.batchSizeEle):((curr_batch + 1) * args.batchSizeEle)]
        iter_index_cache.append(sample_idx_ele)
        batch_x, batch_y = X_elementary[sample_idx_ele], Y_elementary[sample_idx_ele]
        if i == 399:
            print "399!!!!!!!!!!!", batch_y
        tmp_y = np.zeros((args.batchSizeEle, 10))
        for idx, element in enumerate(batch_y):
            tmp_y[idx][element] = 1
        batch_y = tmp_y
        res = func_elementary(batch_x, batch_y, lr_ele_true)
        (cost_ele, pred_ele, debugs) = (res[0], res[1], res[2:])
        # print("Epoch %d, batch %d, time = %ds, train_loss = %.4f" %
        #       (curr_epoch, curr_batch, time() - t_start, cost_ele))

        # temp_err_ele += [1. * sum(batch_y != pred_ele) / args.batchSizeEle]
        temp_cost_ele += [cost_ele]
        eval_error = 0.

        # if np.isnan(cost_ele):
        #     print 'NANS', cost_ele

        """
            Evaluate

        """
        if args.verbose or (curr_batch == n_batch_ele - 1):

            if args.model == 'mlp':
                n_eval = n_test
            else:
                n_eval = 1000

            temp_idx = test_perm[:n_eval]
            batch_x, batch_y = X_test[temp_idx], Y_test[temp_idx]
            tmp_y = np.zeros((n_eval, 10))
            for idx, element in enumerate(batch_y):
                tmp_y[idx][element] = 1
            batch_y = tmp_y
            eval_loss, y_test = func_eval(batch_x, batch_y)

            wrong = 0
            for e1, e2 in zip(y_test, Y_test[temp_idx]):
                if e1 != e2:
                    wrong += 1
            # eval_error = 1. * sum(int(Y_test[temp_idx] != batch_y)) / n_eval
            eval_error = 100. * wrong / n_eval
            print "test sample", n_eval
            print("Valid on Test Set: Epoch %d, batch %d, time = %ds, eval_loss = %.4f, eval_error = %.4f" %
                  (curr_epoch, curr_batch + 1, time() - t_start, eval_loss, eval_error))

    # save the model parameters after T1 into theta_final
    theta_final = []
    for i, w in enumerate(model.params_theta):
        theta_final.append(w.get_value())

    """
        Phase 2: Validation on Hyper set

    """
    n_hyper = X_hyper.shape[0]
    n_batch_hyper = n_hyper / args.batchSizeHyper
    hyper_perm = range(0, n_hyper)
    # np.random.shuffle(hyper_perm)

    err_valid = 0.
    cost_valid = 0.
    t_start = time()
    grad_l_theta = []
    for i in range(0, n_batch_hyper):
        sample_idx = hyper_perm[(i * args.batchSizeHyper):((i + 1) * args.batchSizeHyper)]
        batch_x, batch_y = X_elementary[sample_idx], Y_elementary[sample_idx]
        # TODO: refactor, too slow
        tmp_y = np.zeros((args.batchSizeEle, 10))
        for idx, element in enumerate(batch_y):
            tmp_y[idx][element] = 1
        batch_y = tmp_y
        res = func_hyper_valid(batch_x, batch_y)
        valid_cost, pred_hyper, grad_temp = res[0], res[1], res[2:]
        err_tmp = 0.
        # err_tmp = 1. * sum(batch_y != pred_hyper) / args.batchSizeHyper
        err_valid += err_tmp
        # print "err_temp", err_tmp
        cost_valid += valid_cost

        # accumulate gradient and then take the average
        if i == 0:
            for grad in grad_temp:
                grad_l_theta.append(np.asarray(grad))
        else:
            for k, grad in enumerate(grad_temp):
                grad_l_theta[k] += grad

    err_valid /= n_batch_hyper
    cost_valid /= n_batch_hyper

    # get average grad of all iterations on validation set

    for i, grad in enumerate(grad_l_theta):
        print grad.shape
        grad_l_theta[i] = grad / (np.array(n_hyper * 1., dtype=theano.config.floatX))


    print("Valid on Hyper Set: time = %ds, valid_err = %.2f, valid_loss = %.4f" %
          (time() - t_start, err_valid * 100, cost_valid))

    """
        Phase 3: meta-backward

    """

    # updates for phase 3

    update_hyper, output_hyper_list, phase_3_input = updates_hyper(model.params_lambda, model.params_weight,
                                                    model.lossWithPenalty, grad_l_theta, output_valid_list)

    # Phase 3
    # dloss_dpenalty = T.grad(model.loss, model.params_lambda)
    func_hyper = theano.function(
        inputs=[x, y],
        outputs=output_hyper_list + output_valid_list,
        updates=update_hyper,
        on_unused_input='ignore',
        allow_input_downcast=True)

    # init for pseudo params
    pseudo_params = []
    for i, v in enumerate(model.params_theta):
        pseudo_params.append(v.get_value())

    def replace_pseudo_params(ratio):
        for i, param in enumerate(model.params_theta):
            pseudo_params[i] = (1 - ratio) * theta_initial[i] + ratio * theta_final[i]
            param.set_value(pseudo_params[i])

    n_backward = len(iter_index_cache)
    print "n_backward", n_backward

    rho = np.linspace(0.001, 0.999, n_backward)

    # initialization
    up_lambda, up_v = [], []
    for param in model.params_lambda:
        temp_param = np.zeros_like(param.get_value() * 0., dtype=theano.config.floatX)
        up_lambda += [temp_param]

    for param in model.params_weight:
        temp_v = np.zeros_like(param.get_value() * 0., dtype=theano.config.floatX)
        up_v += [temp_v]

    # time.sleep(20)
    up_theta = grad_l_theta

    iter_index_cache = iter_index_cache[:n_backward]

    for iteration in range(n_backward)[::-1]:
        replace_pseudo_params(rho[iteration])         # line 4
        curr_epoch = iteration / n_batch_ele
        curr_batch = iteration % n_batch_ele
        if iteration % 40 == 0:
            print "Phase 3, ep{} iter{}, total{}".format(curr_epoch, curr_batch, iteration)
        sample_idx_ele = iter_index_cache[iteration]
        # sample_idx_ele = ele_perm[(curr_batch * args.batchSizeEle):((curr_batch + 1) * args.batchSizeEle)]
        batch_x, batch_y = X_elementary[sample_idx_ele], Y_elementary[sample_idx_ele]
        if curr_batch == 399:
            print "399!!!!!!!!!!!", batch_y
        tmp_y = np.zeros((args.batchSizeEle, 10))
        for idx, element in enumerate(batch_y):
            tmp_y[idx][element] = 1
        batch_y = tmp_y

        if args.model == 'mlp':
            for p3, p1, input_p in zip(up_v, up_theta, phase_3_input):
                # print p3.shape, p1.shape
                p3 += lr_ele_true * p1
                input_p.set_value(p3)
                tmp = input_p.get_value()
                # print 'set up_v to obtain hypergrad', tmp[1][1]
                # TIME.sleep(2)
        else:
            for p3, p1, input_p in zip(up_v, up_theta, phase_3_input):
                p3 += lr_ele_true * p1
                input_p.set_value(p3)

        # hessian vector product
        HVP_value = func_hyper(batch_x, batch_y)
        HVP_weight_value = HVP_value[:4]
        HVP_lambda_value = HVP_value[4:8]
        debug_orz = HVP_value[8:]

        # return
        cnt = 0
        for p1, p2, p3, hvp1, hvp2 in zip(up_theta, up_lambda, up_v, HVP_weight_value, HVP_lambda_value):
            # this code is to monitor the up_lambda
            if cnt == 3:
                tmp2 = np.array(hvp2)
                tmp1 = np.array(hvp1)
                if iteration % 40 == 0:
                    print "up_lambda", p2[3][0]
            else:
                cnt += 1
            p1 -= (1. - mom) * np.array(hvp1)
            p2 -= (1. - mom) * np.array(hvp2)
            p3 *= mom

        # print up_lambda[2][0][0]

    return model.params_lambda, up_lambda, fix_weight, eval_loss, eval_error


def update_lambda_every_meta(ori, up, hyper_lr, updatemode):
    tmp = []
    for x, y in zip(ori, up):
        if updatemode == 'unit':
            new_y = np.mean(y, axis=1, keepdims=True)
            tmp.append(x.get_value() - np.sign(new_y) * np.array(float(hyper_lr) * 1., dtype=theano.config.floatX))
            print "metaupdate", x.get_value()[0][1], tmp[-1][0][1]
        else:
            raise AttributeError
    return tmp


if __name__ == '__main__':
    args = setup()
    print 'all argument: ', args
    temp_lambda = None
    loss_change = []
    tmp_weights = None
    for i in range(args.metaEpoch):
        origin_lambda, temp_lambda, tmp_weights, eval_loss, eval_err = run_exp(args, temp_lambda, tmp_weights)
        temp_lambda = update_lambda_every_meta(origin_lambda, temp_lambda, args.lrHyper, 'unit')

        loss_change.append((float(eval_loss), eval_err))
        print("---------------------------------------------------------------------------------------------------")
        print("Training Result: ")
        for k, v in enumerate(loss_change):
            print k, v
        print("---------------------------------------------------------------------------------------------------")
