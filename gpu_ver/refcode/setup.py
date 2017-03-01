import numpy as np

def cnn_setup(params):

    """
    Defining convolutional architecture.

    Arguments: Parameters defining the entire model.

    """
    class cnn_layer():
        def __init__(self, layerType, dimFilters, nFilters, stride, border='valid', doNoise = True, doBN = True):
            self.type = layerType
            self.filter = dimFilters
            self.maps = nFilters
            self.stride = stride
            self.border = border
            self.noise = doNoise
            self.bn = doBN

    cl1 = cnn_layer('conv', (3, 3), (3, 96),  (1, 1), 'valid')
    cl2 = cnn_layer('conv', (3, 3), (96, 96), (1, 1), 'full')
    cl2alt = cnn_layer('conv', (3, 3), (96, 96), (1, 1), 'full')

    cl3 = cnn_layer('pool', (2, 2), (96, 96), (2, 2), 'dummy', 1, 1) # dummy is not used
    cl3alt = cnn_layer('conv', (3, 3), (96, 96), (2, 2), 'valid') # ? stride = 2?


    cl4 = cnn_layer('conv', (3, 3), (96, 192),  (1, 1), 'valid')
    cl5 = cnn_layer('conv', (3, 3), (192, 192), (1, 1), 'full')
    cl6 = cnn_layer('conv', (3, 3), (192, 192), (1, 1), 'valid')

    cl7 = cnn_layer('pool', (2, 2), (192, 192), (2, 2), 'dummy', 1, 1)
    cl7alt = cnn_layer('conv', (3, 3), (192, 192), (2, 2),'valid') # ? stride = 2?


    cl8  = cnn_layer('conv', (3, 3), (192, 192), (1, 1), 'valid')
    cl9  = cnn_layer('conv', (1, 1), (192, 192), (1, 1), 'valid')
    cl10 = cnn_layer('conv', (1, 1), (192, 10),  (1, 1), 'valid', 0, 0)

    cl11 = cnn_layer('average+softmax', (6, 6), (10, 10), (6, 6), 'dummy', 0, 0)
    cl11alt = cnn_layer('average', (6, 6), (10, 10), (6, 6), 0, 0, 0)
    cl12alt = cnn_layer('softmax', (6, 6), (10, 10), (1, 1), 0, 0, 0)

    if params.dataset == 'mnist':
        cl1 = cnn_layer('conv', (3, 3), (1, 96), (1, 1))
    if params.cnnType == 'all_conv':
        cnn_layers = [cl1, cl2, cl3alt, cl4, cl5, cl7alt, cl8, cl9, cl10, cl11]
    elif params.cnnType == 'ladder_baseline':
        cnn_layers = [cl1, cl2, cl2alt, cl3, cl4, cl5, cl6, cl7, cl8, cl9, cl10, cl11]
    else:
        cnn_layers = [cl1, cl2, cl3, cl4, cl5, cl7, cl8, cl9, cl10, cl11]

    return cnn_layers


def setup(replace_params=None):

    ''' Defining entire neural network model and methods for training.

    Arguments:  Dictionary of the form {'paramName': paramValue}.
                E.g. replace_params = {'useT2': False, 'learnRate1': 0.1}
    '''

    ones = np.ones(20)

    class Params():
        def __init__(self, opt):
            print(opt)
            # general setting
            self.verbose = opt.verbose
            self.model = opt.model
            self.dataset = opt.dataset
            self.batchSize1 = 128
            self.batchSize2 = 128
            self.maxEpoch = 1
            self.seed = 1234
            self.cnnType = 'ladder_baseline'
            self.cnnNonlin = 'leaky_relu'
            # meta-backward
            self.meta_bw = True
            self.useVal = 0
            self.saveName = 'result.pkl'
            # PREPROCESSING
            self.predata = opt.predata
            self.predataName = self.dataset + '_preprocess.npz'
            self.ratioHyper = 0.2                                               # elementary set : hyper set
            self.ratioValid = opt.ratioValid                                   # how much of T2 goes to validatio set
            self.preProcess = 'global_contrast_norm'                           # what input preprocessing? 'None'/'m0'/'m0s1'/'minMax'/'pca'/'global_contrast_norm'/'zca'/'global_contrast_norm+zca'
            self.preContrast = 'None'                                          # nonlinear transform over input? 'None'/'tanh'/'arcsinh'/'sigmoid'
            # ARCHITECTURE
            self.nHidden = [784, 1000, 1000, 1000, 10]                         # how many hidden units in each layer?
            self.activation = ['relu','relu','relu','softmax']               # what nonlinearities in each layer?
            self.nLayers = len(self.nHidden)-1                                 # how many layers are there?
            # BATCH NORMALIZATION
            self.batchNorm = False                                              # use batch normalization?
            self.aFix = True                                                   # fix scalling parameter?
            self.movingAvMin = 0.10                                            # moving average paramerer? [0.05-0.20]
            self.movingAvStep = 1                                              # moving average step size?
            self.evaluateTestInterval = 25                                     # how often compute the "exact" BN parameters? i.e. replacing moving average with the estimate from the whole training data
            self.m = 550                                                       # when computing "exact" BN parameters, average over how many samples from training set?
            self.testBN = 'default'                                            # when computing "exact" BN parameters, how? 'default'/'proper'/'lazy'
            self.poolBNafter = False
            # REGULARIZATION
            self.rglrzTrain = ['L2']                               # which rglrz are trained? (which are available? see: rglrzInitial)
            self.rglrz = ['L2']                                    # which rglrz are used?
            self.rglrzPerUnit = []                                             # which rglrz are defined per hidden unit? (default: defined one per layer)
            self.rglrzPerMap = []                                              # which rglrz are defined per map? (for convnets)
            self.rglrzPerNetwork = []                                          # which rglrz are defined per network?
            self.rglrzPerNetwork1 = []                                         # which rglrz are defined per network? BUT have a separate param for the first layer
            self.rglrzInitial = {'L1': 0.*ones,                                # initial values of rglrz
                                 'L2': 0.001*ones,
                         'LmaxCutoff': 0.*ones,                                # soft cutoff param1
                          'LmaxSlope': 0.*ones,                                # soft cutoff param2
                           'LmaxHard': 2.*ones,                                # hard cutoff aka maxnorm
                          'addNoise' : 0.3*ones,
                        'inputNoise' : [0.],                                   # only input noise (if trained, need be PerNetwork)
                            'dropOut': [0.2]+20*[0.5],
                           'dropOutB': [0.2]+20*[0.5]}                   # shared dropout pattern within batch
            self.rglrzLR = {'L1': 0.0001,                                      # scaling factor for learning rate: corresponds to hyperparameters (expected) order of magnitude
                            'L2': 0.001,
                    'LmaxCutoff': 0.1,
                     'LmaxSlope': 0.0001,
                     'addNoise' : 1.,
                   'inputNoise' : 1.}
            # REGULARIZATION: noise specific
            self.noiseupSoftmax = False                                        # is there noise in the softmax layer?
            self.noiseWhere = 'type1'                                          # where is noise added at input? 'type0' - after non-linearity, 'type1' - before non-linearity
            self.noiseT1 = 'None'                                              # type of gaussian noise? 'None'/'multi0'/'multi1'/'fake_drop' --> (x+n)/x*n/x*(n+1)/x*s(n)
            # TRAINING: COST
            self.cost = 'categorical_crossentropy'                             # cost for T1? 'L2'/'categorical_crossentropy'
            self.cost_T2 = 'categorical_crossentropy' # TODO                   # cost for T2? 'L2'/'crossEntropy'                       TODO: 'sigmoidal'/'hingeLoss'
            self.penalize_T2 = False                                           # apply penalty for T2?
            self.cost2Type = 'default'                                         # type of T1T2 cost 'default'/'C2-C1'
            # TRAINING: T2 FD or exact
            self.finiteDiff = False                                            # use finite difference for T2?
            self.FDStyle = '3'                                                 # type of finite difference implementation  '2'/'3'
            self.checkROP = False  # TODO                                      # check ROP operator efficiency
            self.T2gradDIY = False  # TODO                                     # use your own ROP operator
            self.T2onlySGN = False                                             # consider only the sign for T2 update, not the amount
            # TRAINING: OPTIMIZATION
            self.learnRate1 = 0.002                                            # T1 max step size
            self.learnRate2 = 0.001                                            # T2 max step size
            self.learnFun1 = 'olin'                                             # learning rate schedule for T1? (see LRFunctions for options)
            self.learnFun2 = 'None'                                            # learning rate schedule for T2?
            self.opt1 = 'adam'                                                 # optimizer for T1? 'adam'/None (None is SGD)
            self.opt2 = 'adam'                                                 # optimizer for T2? 'adam'/None (None is SGD)
            self.use_momentum = False                                          # applies both to T1 and T2, set the terms to 0 for either if want to disable for one
            self.momentum1 = [0.5, 0.9]                                        # T1 max and min momentum values
            self.momentum2 = [0.5, 0.9]                                        # T2 max and min momentum values
            self.momentFun = 'exp'                                             # momentum decay function
            self.halfLife = 1                                                  # decay function parameter, set to be at halfLife*10,000 updates later
            self.triggerT2 = 0.                                                # when to start training with T2
            self.hessian_T2 = False                                            # apply penalty for T2?
            self.avC2grad = 'None'                                             # taking averaging of C2grad and how? 'None'/'adam'/'momentum'
            self.decayT2 = 0.                                                  # decay factor for T2 params
            self.MM = 1  # TODO?                                               # for stochastic net: how many parallel samples do we take? IMPORTANCE: could be used to train discrete hyperparameters as well
            # TRAINING: OTHER
            # TRACKING, PRINTING
            self.trackPerEpoch = 1                                             # how often within epoch track error?
            self.printInterval = 10                                             # how often print error?
            self.printBest = 40000                                             # each updates print best value?
            self.activTrack = ['mean', 'std', 'max',                           # what network statistics are you following?
                               'const', 'spars',
                               'wmean', 'wstd', 'wmax',
                               'rnoise', 'rnstd',
                               'bias', 'a']
            self.forVideo = ['a', 'b', 'h']                                    # takes a sample of say 100-200 of those from each layer
            self.showGrads = False                                             # do you show gradient values?
#            self.listGrads = ['grad', 'grad_rel', 'grad_angle', 'grad_max', 'p_t', 'p_t_rel', 'p_t_angle', 'p_t_max']
            self.listGrads = ['grad', 'grad_angle', 'p_t', 'p_t_angle']        # which gradient measures to track?
            self.trackGrads = False                                            # monitor gradients during training?
            self.trackStats = False                                            # monitor layer and weight statistics during training?
            self.track4Vid = False                                             # TODO: monitor for creating animation
            self.track1stFeatures = False                                      # TODO: monitor 1st layer features


    # replace default parameters
    params = Params(replace_params)
    # for key, val in replace_params.iteritems():
    #     assert hasattr(params, key), 'Setting %s does not exist' % key
    #     setattr(params, key, val)

    # additional parameters if convolutional network
    if params.model == 'convnet':
        params.convLayers = cnn_setup(params)
        params.nLayers = len(params.convLayers)
    else:
        assert len(params.nHidden) == len(params.activation)+1
        params.nLayers = len(params.nHidden)-1

    # change dimensions for cifar-10 and svhn
    if params.dataset in ['cifar10', 'svhn']:
        params.nHidden[0] = 3*1024

    if (not params.meta_bw) or (params.rglrz == []):
        params.rglrzTrain = []
        params.meta_bw = False

    return params
