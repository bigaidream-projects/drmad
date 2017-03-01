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


def run_exp(args, update_lambda):

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
    lr_hyper = T.fscalar('lr_hyper')
    grad_valid_weight = T.tensor4('grad_valid_weight')

    if args.model == 'mlp':
        model = MLP(x=x, y=y, args=args)
    elif args.model == 'convnet':
        model = ConvNet(x=x, y=y, args=args)
        X_elementary = X_elementary.reshape(-1, 1, 28, 28)
        X_hyper = X_hyper.reshape(-1, 1, 28, 28)
        X_valid = X_valid.reshape(-1, 1, 28, 28)
        X_test = X_test.reshape(-1, 1, 28, 28)
    else:
        raise AttributeError

    update_ele, update_valid, output_valid_list, share_var_dloss_dweight = update(model.params_theta, model.params_lambda, model.params_weight,
                                      model.loss, model.penalty, model.lossWithPenalty,
                                      lr_ele, lr_hyper)

    if update_lambda:
        for up, origin in zip(update_lambda, model.params_lambda):
            print origin[0]
            origin += up * args.lrHyper
            print origin[0]

    print(update_ele)
    print(update_valid)

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
    n_ele, n_valid, n_test = X_elementary.shape[0], X_valid.shape[0], X_test.shape[0]
    n_batch_ele = n_ele / args.batchSizeEle
    test_perm, ele_perm = range(0, n_test), range(0, n_ele)
    last_iter = args.maxEpoch * n_batch_ele - 1
    temp_err_ele = []
    temp_cost_ele = []

    t_start = time()

    # save the model parameters into theta_initial
    theta_initial = {}
    for w in model.params_theta:
        theta_initial[w] = w.get_value()

    for i in range(0, args.maxEpoch * n_batch_ele):
        curr_epoch = i / n_batch_ele
        curr_batch = i % n_batch_ele

        """
            Learning rate and momentum schedules.

        """
        t = 1. * i / (args.maxEpoch * n_batch_ele)
        lr_ele = 0.02
        moment_ele = 0.5
        # if curr_batch == 0:
        #     np.random.shuffle(ele_perm)

        """
            Update

        """
        sample_idx_ele = ele_perm[(curr_batch * args.batchSizeEle):((curr_batch + 1) * args.batchSizeEle)]
        batch_x, batch_y = X_elementary[sample_idx_ele], Y_elementary[sample_idx_ele]
        tmp_y = np.zeros((args.batchSizeEle, 10))
        for idx, element in enumerate(batch_y):
            tmp_y[idx][element] = 1
        batch_y = tmp_y
        res = func_elementary(batch_x, batch_y, lr_ele)
        (cost_ele, pred_ele, debugs) = (res[0], res[1], res[2:])
        # temp_err_ele += [1. * sum(batch_y != pred_ele) / args.batchSizeEle]
        temp_cost_ele += [cost_ele]
        eval_error = 0.

        if np.isnan(cost_ele):
            print 'NANS', cost_ele

        """
            Evaluate

        """
        if args.verbose or (curr_epoch == n_batch_ele - 1):

            if args.model == 'mlp':
                n_eval = 5000
            else:
                n_eval = 1000

            eval_error = 0.
            temp_idx = test_perm[:n_eval]
            batch_x, batch_y = X_test[temp_idx], Y_test[temp_idx]
            tmp_y = np.zeros((n_eval, 10))
            for idx, element in enumerate(batch_y):
                tmp_y[idx][element] = 1
            batch_y = tmp_y
            eval_loss, y_test = func_eval(batch_x, batch_y)
            # eval_error = 1. * sum(y_test != batch_y) / n_eval

            print("Epoch %d, batch %d, time = %ds, eval_err = %.2f, eval_loss = %.4f" %
                  (curr_epoch, curr_batch, time() - t_start, eval_error * 100, eval_loss))

    # save the model parameters after T1 into theta_final
    theta_final = {}
    for w in model.params_theta:
        theta_final[w] = w.get_value()

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
    for grad in grad_l_theta:
        print grad.shape
        grad = grad / (np.array(n_hyper * 1., dtype=theano.config.floatX))

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
    pseudo_params = {}
    for w in model.params_theta:
        pseudo_params[w] = w.get_value()

    def replace_pseudo_params(ratio):
        for param in model.params_theta:
            pseudo_params[param] = (1 - ratio) * theta_initial[param] + ratio * theta_final[param]
            param.set_value(pseudo_params[param])

    for xxx in phase_3_input:
        print(xxx)
        params = xxx.get_value()
        print params
        print params.shape

    rho = np.linspace(0.001, 0.999, args.maxEpoch * n_batch_ele)

    # initialization
    up_lambda = []
    for param in model.params_lambda:
        temp_param = np.zeros_like(param.get_value() * 0., dtype=theano.config.floatX)
        up_lambda += [temp_param]
        print temp_param.shape

    # time.sleep(20)
    up_theta = grad_l_theta

    for iteration in range(args.maxEpoch * n_batch_ele)[::-1]:
        replace_pseudo_params(rho[iteration])         # line 4
        curr_epoch = iteration / n_batch_ele
        curr_batch = iteration % n_batch_ele
        print "Phase 3, ep{} iter{}".format(curr_epoch, curr_batch)
        sample_idx_ele = ele_perm[(curr_batch * args.batchSizeEle):((curr_batch + 1) * args.batchSizeEle)]
        batch_x, batch_y = X_elementary[sample_idx_ele], Y_elementary[sample_idx_ele]
        tmp_y = np.zeros((args.batchSizeEle, 10))
        for idx, element in enumerate(batch_y):
            tmp_y[idx][element] = 1
        batch_y = tmp_y
        
        if args.model == 'mlp':
            for p1, input_p in zip(up_theta, phase_3_input):
                # print p1[1]
                input_p.set_value(p1)
                tmp = input_p.get_value()
                print tmp[1][1]
        else:
            pass
            # TODO: I don't know what it should print. The shape of tmp is different so I blocked it.

        # hessian vector product
        HVP_value = func_hyper(batch_x, batch_y)
        HVP_weight_value = HVP_value[:4]
        HVP_lambda_value = HVP_value[4:8]
        debug_orz = HVP_value[8:]

        # return
        cnt = 0
        for p1, p2, hvp1, hvp2 in zip(up_theta, up_lambda, HVP_weight_value, HVP_lambda_value):
            if cnt == 3:
                tmp2 = np.array(hvp2)
                tmp1 = np.array(hvp1)
                print "hvp_weight", tmp1[3][0]
                print "hvp_lambda", tmp2[3][0] * (1 - lr_ele)
                print "up_weight", p1[3][0]
                print "up_lambda", p2[3][0]
                # print up_theta[3][0]
                # TIME.sleep(5)
            else:
                cnt += 1
            p1 -= lr_ele * np.array(hvp1)
            p2 -= lr_ele * np.array(hvp2)
            # print

    print up_lambda[0]

    return model.params_lambda, up_lambda


if __name__ == '__main__':
    args = setup()
    print 'all argument: ', args
    temp_lambda = None
    for i in range(args.metaEpoch):
        _, temp_lambda = run_exp(args, temp_lambda)
