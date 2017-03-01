import lasagne.layers as ll
import theano.tensor as T

from layers import DenseLayerWithReg


class mlp(object):
    def __init__(self, rng, rstream, x, y, setting): # add cost

        """

        Constructing the mlp model.
        
        Arguments: 
            rng, rstream         - random streams

        """
        self.paramsEle = []
        self.paramsHyper = []
        self.layers = [ll.InputLayer((None, 3, 28, 28))]
        self.layers.append(ll.ReshapeLayer(self.layers[-1], (None, 3*28*28)))
        penalty = 0.
        for num in [1000, 1000, 1000, 10]: # TODO: refactor it later
            self.layers.append(DenseLayerWithReg(setting, self.layers[-1], num_units=num))
            self.paramsEle += self.layers[-1].W
            self.paramsEle += self.layers[-1].b
            if setting.regL2 is not None:
                tempL2 = self.layers[-1].L2 * T.sqr(self.layers[-1].W)
                penalty += T.sum(tempL2)
                self.paramsHyper += self.layers[-1].L2

        self.y = self.layers[-1].output
        self.prediction = T.argmax(self.y, axis=1)
        self.penalty = penalty if penalty != 0. else T.constant(0.)

        def stable(x, stabilize=True):
            if stabilize:
                x = T.where(T.isnan(x), 1000., x)
                x = T.where(T.isinf(x), 1000., x)
            return x

        if setting.cost == 'categorical_crossentropy':
            def costFun1(y, label):
                return stable(-T.log(y[T.arange(label.shape[0]), label]),
                              stabilize=True)
        else:
            raise NotImplementedError


        def costFunT1(*args, **kwargs):
            return T.mean(costFun1(*args, **kwargs))

        # cost function
        self.trainCost = costFunT1(self.y, y)
        self.classError = T.mean(T.cast(T.neq(self.guessLabel, y), 'float32'))
