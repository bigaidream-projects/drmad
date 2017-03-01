import theano.tensor as T
import theano
# TODO: more robust implementation: address problems with Lop
from theano.compat import OrderedDict, izip

# hypers influencing only penalty term (cause Theano)
penalty_list = ['L1', 'L2', 'Lmax', 'LmaxSlope', 'LmaxCutoff', 'LmaxHard']
# hypers influencing only NLL (cause Theano)
noise_list = ['addNoise', 'inputNoise']


def format_as(use_list, use_tuple, outputs):
    """
    Formats the outputs according to the flags `use_list` and `use_tuple`.
    If `use_list` is True, `outputs` is returned as a list (if `outputs`
    is not a list or a tuple then it is converted in a one element list).
    If `use_tuple` is True, `outputs` is returned as a tuple (if `outputs`
    is not a list or a tuple then it is converted into a one element tuple).
    Otherwise (if both flags are false), `outputs` is returned.
    """
    assert not (use_list and use_tuple), \
        "Both flags cannot be simultaneously True"
    if (use_list or use_tuple) and not isinstance(outputs, (list, tuple)):
        if use_list:
            return [outputs]
        else:
            return (outputs,)
    elif not (use_list or use_tuple) and isinstance(outputs, (list, tuple)):
        assert len(outputs) == 1, \
            "Wrong arguments. Expected a one element list"
        return outputs[0]
    elif use_list or use_tuple:
        if use_list:
            return list(outputs)
        else:
            return tuple(outputs)
    else:
        return outputs


def Lop2(f, wrt, eval_points):
    res = []
    for t1, t2, t3 in zip(f, wrt, eval_points):
        res.append(T.grad(T.sum(t1 * t3), t2))
    return res


def Lop(f, wrt, eval_points, consider_constant=None,
        disconnected_inputs='raise'):
    """

        This Lop() has the same functionality of Theano.Lop()
    """
    if type(eval_points) not in (list, tuple):
        eval_points = [eval_points]

    using_list = isinstance(wrt, list)
    using_tuple = isinstance(wrt, tuple)

    if not isinstance(f, (list, tuple)):
        f = [f]

    # make copies of f and grads so we don't modify the client's copy
    f = list(f)
    grads = list(eval_points)

    # var_grads = []
    # for grad in grads:
    #     var_grad = theano.shared(grad, name="let me see", allow_downcast=True)
    #     var_grads += [var_grad]

    if not isinstance(wrt, (list, tuple)):
        wrt = [wrt]

    assert len(f) == len(grads)
    known = dict(izip(f, grads))

    # print "I know nothing.", known, "\n\n\n", f, "\n\n\n", grads, type(grads[0])

    ret = T.grad(cost=None, known_grads=known,
               consider_constant=consider_constant, wrt=wrt,
               disconnected_inputs=disconnected_inputs)

    # print "return value", ret[0], type(ret[0])

    return format_as(using_list, using_tuple, ret)

def hypergrad(params_lambda, params_weight,
             dL_dweight, grad_valid_weight):


    # theano.gradient.Lop(f, wrt, eval_points, consider_constant=None,
    #                     disconnected_inputs='raise')
    #     Computes the L operation on `f` wrt to `wrt` evaluated at points given
    # in `eval_points`. Mathematically this stands for the jacobian of `f` wrt
    # to `wrt` left muliplied by the eval points.

    hypergrad_penalty_weight = Lop2(dL_dweight, params_weight, grad_valid_weight)
    hypergrad_penalty_lambda = Lop2(dL_dweight, params_lambda, grad_valid_weight)

    return hypergrad_penalty_weight, hypergrad_penalty_lambda
