import numpy as np
import theano.tensor as T

from lasagne.layers import Layer
from lasagne import nonlinearities, init
eps = np.float32(1e-8)
zero = np.float32(0.)
one = np.float32(1.)


def nan_o_meter(x):
    n_nans = T.sum(T.isnan(x))
    n_infs = T.sum(T.isinf(x))
    print '# Nans ', n_nans, '# Infs', n_infs 
    return
    

def relu(x):
    return T.maximum(x, 0)


class DenseLayerWithReg(Layer):
    """

        A DenseLayer with Reg
        
    """
    def __init__(self, args, incoming, num_units, W=init.GlorotUniform(),
                 b=init.Constant(0.), nonlinearity=nonlinearities.rectify,
                 num_leading_axes=1, **kwargs):
        super(DenseLayerWithReg, self).__init__(incoming, **kwargs)
        self.nonlinearity = (nonlinearities.identity if nonlinearity is None
                             else nonlinearity)

        self.num_units = num_units

        if num_leading_axes >= len(self.input_shape):
            raise ValueError(
                    "Got num_leading_axes=%d for a %d-dimensional input, "
                    "leaving no trailing axes for the dot product." %
                    (num_leading_axes, len(self.input_shape)))
        elif num_leading_axes < -len(self.input_shape):
            raise ValueError(
                    "Got num_leading_axes=%d for a %d-dimensional input, "
                    "requesting more trailing axes than there are input "
                    "dimensions." % (num_leading_axes, len(self.input_shape)))
        self.num_leading_axes = num_leading_axes

        if any(s is None for s in self.input_shape[num_leading_axes:]):
            raise ValueError(
                    "A DenseLayer requires a fixed input shape (except for "
                    "the leading axes). Got %r for num_leading_axes=%d." %
                    (self.input_shape, self.num_leading_axes))
        num_inputs = int(np.prod(self.input_shape[num_leading_axes:]))

        self.W = self.add_param(W, (num_inputs, num_units), name="W")
        if b is None:
            self.b = None
        else:
            self.b = self.add_param(b, (num_units,), name="b",
                                    regularizable=False)

        if args.regL1 is True:
            self.L1 = self.add_param(init.Constant(args.regInit['L1']),
                                     (num_inputs, num_units), name="L1")
        if args.regL2 is True:
            self.L2 = self.add_param(init.Constant(args.regInit['L2']),
                                     (num_inputs, num_units), name="L2")

    def get_output_shape_for(self, input_shape):
        return input_shape[:self.num_leading_axes] + (self.num_units,)

    def get_output_for(self, input, **kwargs):
        num_leading_axes = self.num_leading_axes
        if num_leading_axes < 0:
            num_leading_axes += input.ndim
        if input.ndim > num_leading_axes + 1:
            # flatten trailing axes (into (n+1)-tensor for num_leading_axes=n)
            input = input.flatten(num_leading_axes + 1)

        activation = T.dot(input, self.W)
        if self.b is not None:
            activation = activation + self.b
        return self.nonlinearity(activation)


def conv_output_length(input_length, filter_size, stride, pad=0):
    """Helper function to compute the output size of a convolution operation
    From Lasagne

    """
    if input_length is None:
        return None
    if pad == 'valid':
        output_length = input_length - filter_size + 1
    elif pad == 'full':
        output_length = input_length + filter_size - 1
    elif pad == 'same':
        output_length = input_length
    elif isinstance(pad, int):
        output_length = input_length + 2 * pad - filter_size + 1
    else:
        raise ValueError('Invalid pad: {0}'.format(pad))

    # This is the integer arithmetic equivalent to
    # np.ceil(output_length / stride)
    output_length = (output_length + stride - 1) // stride

    return output_length

def _as_tuple(x, N, t=None):
    try:
        X = tuple(x)
    except TypeError:
        X = (x,) * N

    if (t is not None) and not all(isinstance(v, t) for v in X):
        raise TypeError("expected a single value or an iterable "
                        "of {0}, got {1} instead".format(t.__name__, x))

    if len(X) != N:
        raise ValueError("expected a single value or an iterable "
                         "with length {0}, got {1} instead".format(N, x))

    return X


class Conv2DLayerWithReg(Layer):
    """
    Conv2DLayer(args, incoming, num_filters, filter_size,
    stride=(1, 1), pad=0, untie_biases=False,
    W=lasagne.init.GlorotUniform(), b=lasagne.init.Constant(0.),
    nonlinearity=lasagne.nonlinearities.rectify,
    convolution=theano.tensor.nnet.conv2d, **kwargs)

    2D convolutional layer from Lasagne with DrMAD regularizer term

    """
    def __init__(self, args, incoming, num_filters, filter_size, stride=(1, 1),
                 pad=0, untie_biases=False,
                 W=init.GlorotUniform(), b=init.Constant(0.),
                 nonlinearity=nonlinearities.rectify,
                 convolution=T.nnet.conv2d, **kwargs):
        super(Conv2DLayerWithReg, self).__init__(incoming, **kwargs)
        
        if nonlinearity is None:
            self.nonlinearity = nonlinearities.identity
        else:
            self.nonlinearity = nonlinearity

        self.num_filters = num_filters
        self.filter_size = _as_tuple(filter_size, 2)
        self.stride = _as_tuple(stride, 2)
        self.untie_biases = untie_biases
        self.convolution = convolution

        if pad == 'valid':
            self.pad = (0, 0)
        elif pad in ('full', 'same'):
            self.pad = pad
        else:
            self.pad = _as_tuple(pad, 2, int)

        self.W = self.add_param(W, self.get_W_shape(), name="W")
        if b is None:
            self.b = None
        else:
            if self.untie_biases:
                biases_shape = (num_filters, self.output_shape[2], self.
                                output_shape[3])
            else:
                biases_shape = (num_filters,)
            self.b = self.add_param(b, biases_shape, name="b",
                                    regularizable=False)
         
        if args.regL1 is True:
            self.L1 = self.add_param(init.Constant(args.regInit['L1']),
                                     self.get_W_shape() , name="L1")
        if args.regL2 is True:
            self.L2 = self.add_param(init.Constant(args.regInit['L2']),
                                     self.get_W_shape() , name="L2")


    def get_W_shape(self):
        """Get the shape of the weight matrix `W`.

        Returns
        -------
        tuple of int
            The shape of the weight matrix.
        """
        num_input_channels = self.input_shape[1]
        return (self.num_filters, num_input_channels, self.filter_size[0],
                self.filter_size[1])

    def get_output_shape_for(self, input_shape):
        pad = self.pad if isinstance(self.pad, tuple) else (self.pad,) * 2

        output_rows = conv_output_length(input_shape[2],
                                         self.filter_size[0],
                                         self.stride[0],
                                         pad[0])

        output_columns = conv_output_length(input_shape[3],
                                            self.filter_size[1],
                                            self.stride[1],
                                            pad[1])

        return (input_shape[0], self.num_filters, output_rows, output_columns)

    def get_output_for(self, input, input_shape=None, **kwargs):
        # The optional input_shape argument is for when get_output_for is
        # called directly with a different shape than self.input_shape.
        if input_shape is None:
            input_shape = self.input_shape

        if self.stride == (1, 1) and self.pad == 'same':
            # simulate same convolution by cropping a full convolution
            conved = self.convolution(input, self.W, subsample=self.stride,
                                      image_shape=input_shape,
                                      filter_shape=self.get_W_shape(),
                                      border_mode='full')
            shift_x = (self.filter_size[0] - 1) // 2
            shift_y = (self.filter_size[1] - 1) // 2
            conved = conved[:, :, shift_x:input.shape[2] + shift_x,
                            shift_y:input.shape[3] + shift_y]
        else:
            # no padding needed, or explicit padding of input needed
            if self.pad == 'full':
                border_mode = 'full'
                pad = [(0, 0), (0, 0)]
            elif self.pad == 'same':
                border_mode = 'valid'
                pad = [(self.filter_size[0] // 2,
                        (self.filter_size[0] - 1) // 2),
                       (self.filter_size[1] // 2,
                        (self.filter_size[1] - 1) // 2)]
            else:
                border_mode = 'valid'
                pad = [(self.pad[0], self.pad[0]), (self.pad[1], self.pad[1])]
            if pad != [(0, 0), (0, 0)]:
                input = padding.pad(input, pad, batch_ndim=2)
                input_shape = (input_shape[0], input_shape[1],
                               None if input_shape[2] is None else
                               input_shape[2] + pad[0][0] + pad[0][1],
                               None if input_shape[3] is None else
                               input_shape[3] + pad[1][0] + pad[1][1])
            conved = self.convolution(input, self.W, subsample=self.stride,
                                      image_shape=input_shape,
                                      filter_shape=self.get_W_shape(),
                                      border_mode=border_mode)

        if self.b is None:
            activation = conved
        elif self.untie_biases:
            activation = conved + self.b.dimshuffle('x', 0, 1, 2)
        else:
            activation = conved + self.b.dimshuffle('x', 0, 'x', 'x')

        return self.nonlinearity(activation)


