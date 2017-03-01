import cPickle
import gzip
import os

from sklearn import preprocessing
import numpy as np
from numpy.random import RandomState    
import scipy


class ContrastNorm(object):
    def __init__(self, scale=55, epsilon=1e-6):
        self.scale = np.float64(scale)
        self.epsilon = np.float64(epsilon)

    def apply(self, data, copy=False):
        if copy:
            data = np.copy(data)
        data_shape = data.shape
        if len(data.shape) > 2:
            data = data.reshape(data.shape[0], np.product(data.shape[1:]))
        assert len(data.shape) == 2, 'Contrast norm on flattened data'
#        assert np.min(data) >= 0.
#        assert np.max(data) <= 1.
        data -= data.mean(axis=1)[:, np.newaxis]
        norms = np.sqrt(np.sum(data ** 2, axis=1)) / self.scale
        norms[norms < self.epsilon] = self.epsilon
        data /= norms[:, np.newaxis]
        if data_shape != data.shape:
            data = data.reshape(data_shape)
        return data


class ZCA(object):
    def __init__(self, n_components=None, data=None, filter_bias=0.1):
        self.filter_bias = np.float64(filter_bias)
        self.P = None
        self.P_inv = None
        self.n_components = 0
        self.is_fit = False
        if n_components and data is not None:
            self.fit(n_components, data)
    def fit(self, n_components, data):
        if len(data.shape) == 2:
            self.reshape = None
        else:
            assert n_components == np.product(data.shape[1:]), \
                'ZCA whitening components should be %d for convolutional data'\
                % np.product(data.shape[1:])
            self.reshape = data.shape[1:]
        data = self._flatten_data(data)
        assert len(data.shape) == 2
        n, m = data.shape
        self.mean = np.mean(data, axis=0)
        bias_filter = self.filter_bias * np.identity(m, 'float64')
        cov = np.cov(data, rowvar=0, bias=1) + bias_filter
        eigs, eigv = np.linalg.eig(cov.astype(np.float64))
        assert not np.isnan(eigs).any()
        assert not np.isnan(eigv).any()
        print 'eigenvals larger than bias', np.sum(eigs > 0.1)/3072.
        print 'min eigenval: ', eigs.min(), 'max eigenval: ', eigs.max()
        assert eigs.min() > 0
        if self.n_components:
            eigs = eigs[-self.n_components:]
            eigv = eigv[:, -self.n_components:]
        sqrt_eigs = np.sqrt(eigs)
        self.P = np.dot(eigv * (1.0 / sqrt_eigs), eigv.T)
        assert not np.isnan(self.P).any()
        self.P_inv = np.dot(eigv * sqrt_eigs, eigv.T)
        self.P = np.float32(self.P)
        self.P_inv = np.float32(self.P_inv)
        self.is_fit = True
    def apply(self, data, remove_mean=True):
        data = self._flatten_data(data)
        d = data - self.mean if remove_mean else data
        return self._reshape_data(np.dot(d, self.P))
    def inv(self, data, add_mean=True):
        d = np.dot(self._flatten_data(data), self.P_inv)
        d += self.mean if add_mean else 0.
        return self._reshape_data(d)
    def _flatten_data(self, data):
        if self.reshape is None:
            return data
        assert data.shape[1:] == self.reshape
        return data.reshape(data.shape[0], np.product(data.shape[1:]))
    def _reshape_data(self, data):
        assert len(data.shape) == 2
        if self.reshape is None:
            return data
        return np.reshape(data, (data.shape[0],) + self.reshape)
        
        
def store(item, name):
    """
        Pickle item under name.

    """
    import pickle
    file = open(name+'.pkl','wb')
    pickle.dump(item, file)
    file.close()
    return


def permute(data, label, params):
    """
        Permute data.
    
    """
    rndSeed = RandomState(params.seed)
    permute = rndSeed.permutation(data.shape[0])
    data = data[permute]
    label = label[permute]

    return (data, label)
    

def read(params):

    """
        Read data from 'datasets/...'
    
    """
    if params.dataset == 'mnist':
        
       filename = 'datasets/mnist.pkl.gz' 
       if not os.path.exists(filename):
           raise Exception("Dataset not found!")
    
       data = cPickle.load(gzip.open(filename))
       t1Data, t1Label = data[0][0], np.int32(data[0][1])
       vData, vLabel = data[1][0], np.int32(data[1][1])
       testD, testL = data[2][0], np.int32(data[2][1])
    
    elif params.dataset == 'not_mnist':
        
       filename = 'datasets/not_mnist.pkl.gz' 
       if not os.path.exists(filename):
           raise Exception("Dataset not found!")
    
       data = cPickle.load(gzip.open(filename))
       t1Data, t1Label = data[0][0], np.int32(data[0][1])
       testD, testL = data[1][0], np.int32(data[1][1])
       del data
                      
       split = 400000
       t1Data, t1Label = permute(t1Data, t1Label, params)                
       vData, vLabel = t1Data[split:], t1Label[split:]
       t1Data, t1Label = t1Data[:split], t1Label[:split]

    elif params.dataset == 'svhn':
        
       f1 = 'datasets/svhn_train.pkl.gz' 
       f2 = 'datasets/svhn_test.pkl.gz' 
       if not os.path.exists(f1) or not os.path.exists(f2):
           raise Exception("Dataset not found!")
    
       [t1Data, t1Label] = cPickle.load(gzip.open(f1))
       [testD, testL] = cPickle.load(gzip.open(f2))
       t1Label = t1Label[:,0]; testL = testL[:,0]
                      
       split = 65000
       t1Data, t1Label = permute(t1Data, t1Label, params)                
       vData, vLabel = t1Data[split:], t1Label[split:]
       t1Data, t1Label = t1Data[:split], t1Label[:split]

    elif params.dataset == 'cifar10':
    
       folderName = 'datasets/cifar-10-batches-py/' # assumes unzipped
       if not os.path.exists(folderName):
           raise Exception("Dataset not found!")
    
       batchNames = ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4'] 
       t1Data, t1Label = np.empty((0,3072), dtype = float), np.empty((0), dtype = int)
    
       for item in batchNames: 
           fo = open(folderName + item, 'rb'); dict = cPickle.load(fo); fo.close()
           t1Data = np.append(t1Data, np.float32(dict['data']), axis = 0)
           t1Label = np.append(t1Label, np.int32(dict['labels']))
           
       fo = open(folderName + 'data_batch_5', 'rb'); dict = cPickle.load(fo); fo.close()
       vData = np.float32(dict['data']); vLabel = np.int32(dict['labels'])  
       fo = open(folderName + 'test_batch', 'rb'); dict = cPickle.load(fo); fo.close()
       testD = np.float32(dict['data']); testL = np.int32(dict['labels'])   

    else: 
        print 'Dataset '+params.dataset+' is not implemented.'
#    TODO
#    elif params.daaset == 'svhn':        
    return  t1Data, t1Label, vData, vLabel, testD, testL


def gcn(data, params):
    """
        Global contrast normalization of data.
        Each image has mean zero and var one across its own pixels.   
        
    """
    eps = 1e-6; lam = 0

    gcn_data = []    
    for temp in data:
        
        gcnMean = np.mean(temp, axis=1) 
        gcnStd = np.maximum(eps, np.sqrt(lam + np.var(temp, axis = 1)))       

        temp = temp - gcnMean[:, np.newaxis]
        gcn_data += [temp/gcnStd[:, np.newaxis]] 
    
    return gcn_data


def zca_white(data, params, eps=1e-5): # TODO: FIX doesn't seem to work
    
    ''' ZCA whitening of data.
            
    '''
    test = data[0] 
        
    m = np.mean(test, axis = 0)
    ctest = test -  m    
    covMatrix = np.dot(ctest.T, ctest) / 1.*test.shape[1]
    
    U,S,V = np.linalg.svd(covMatrix)    
    S = np.diag(S)
    ZCA = np.dot(np.dot(U, 1.0/np.sqrt(S + eps)), U.T)
 
    whiteData = []
    for item in data:
        whiteData += [np.dot(item - m, ZCA)] # whitened
    store(ZCA, params.dataset+'_test_zca')

    return whiteData  


def show_samples(samples, nShow):   
    
    """
        Show some input samples.

    """
    import math
    import matplotlib.pyplot as plt
    _, nFeatures, x, y = samples.shape
    nColumns = int(math.ceil(nShow/5.))
    
    for i in range(nShow):
        plt.subplot(5, nColumns, i+1)
        image = samples[i]
        image = np.rollaxis(image, 0, 3)*5.
        plt.imshow(image) 
#        plt.axis('off')


def read_preprocess(params):
    """
        Read data, form T1 and T2 sets, preprocess data.

    """

    if params.dataset == 'mnist':
        pcha = 1
        plen = 28
    else:
        pcha = 3
        plen = 32

    ratioHyper = params.ratioHyper
    ratioValid = params.ratioValid
    preProcess = params.preProcess
    preContrast = params.preContrast
    sigmoid = lambda x: 1./(1.+ np.exp(-x))
    
    # read data
    t1Data, t1Label, vData, vLabel, testD, testL = read(params)

    # permuting data    
    vData, vLabel = permute(vData, vLabel, params)
    t1Data, t1Label = permute(t1Data, t1Label, params)

    # form datasets T1 and T2 
    if params.meta_bw:
        nVSamples = vData.shape[0]
        # set up t2+validation
        if ratioHyper > 1.:
            tempIndex = int(round((ratioHyper - 1.)*nVSamples))
            tempData = t1Data[:tempIndex]
            tempLabel = t1Label[:tempIndex]
            vData = np.concatenate((vData, tempData))
            vLabel = np.concatenate((vLabel, tempLabel))
            t1Data = t1Data[tempIndex:]
            t1Label = t1Label[tempIndex:]
        elif ratioHyper < 1.:
            tempIndex = int(round((1.-ratioHyper)*nVSamples))
            tempData = vData[:tempIndex]
            tempLabel = vLabel[:tempIndex]
            t1Data = np.concatenate((t1Data, tempData))
            t1Label = np.concatenate((t1Label, tempLabel))
            vData = vData[tempIndex:]
            vLabel = vLabel[tempIndex:]
        # shuffle indices in t2+validation
        nVSamples = vData.shape[0]
        # set up t2 and validation
        if params.ratioValid > 0:
           tempIndex = int(round(nVSamples*(1.-ratioValid)))
           t2Data = vData[:tempIndex]
           t2Label = vLabel[:tempIndex]
           vData = vData[tempIndex:]
           vLabel = vLabel[tempIndex:]
        else:   
           tempIndex = int(round(nVSamples*(1.-ratioValid)))
           t2Data = vData
           t2Label = vLabel
           vData = vData[tempIndex:]
           vLabel = vLabel[tempIndex:]

    else:
        t2Data = []
        t2Label = [] 
        if not params.ratioValid > 0:
           t1Data = np.concatenate((vData, t1Data))
           t1Label = np.concatenate((vLabel, t1Label))            

    # global contrast normalization and ZCA    
    if preProcess in ['global_contrast_norm', 'global_contrast_norm+zca', 'zca']:
        
        if not params.meta_bw:
            t2Data = t1Data[:5, :]
        #data = [t1Data, t2Data, testD, vData]
        if params.dataset == 'convnet':
            t1Data = t1Data.reshape(-1, pcha, plen, plen)
            t2Data = t2Data.reshape(-1, pcha, plen, plen)
            testD = testD.reshape(-1, pcha, pcha, plen)
        t1Data.astype(dtype=np.float64); t2Data.astype(dtype=np.float64); testD.astype(dtype=np.float64)
       
        #print np.max(t1Data), np.max(t2Data), np.max(testD), ' shapes:', t1Data.shape, t2Data.shape, testD.shape
        #print np.var(t1Data), np.var(t2Data), np.var(testD) 
           
        if preProcess in ['global_contrast_norm', 'global_contrast_norm+zca']:
            gcn = ContrastNorm()
            t1Data = gcn.apply(t1Data/np.float64(255.))
            t2Data = gcn.apply(t2Data/np.float64(255.))
            testD = gcn.apply(testD/np.float64(255.))

            #print np.max(t1Data), np.max(t2Data), np.max(testD), ' shapes:', t1Data.shape, t2Data.shape, testD.shape
            #print np.var(t1Data), np.var(t2Data), np.var(testD) 

           
        if preProcess in ['zca', 'global_contrast_norm+zca']:                 
            white = ZCA(3072, t1Data.copy())
            t1Data = white.apply(t1Data)
            t2Data = white.apply(t2Data)
            testD = white.apply(testD)
            
            #print np.max(t1Data), np.max(t2Data), np.max(testD), ' shapes:', t1Data.shape, t2Data.shape, testD.shape
            #print np.var(t1Data), np.var(t2Data), np.var(testD), 
        
    # other kinds of preprocessing            
    else:         
        scaler = {
             'm0': preprocessing.StandardScaler(with_std = False).fit(t1Data),
             'm0s1': preprocessing.StandardScaler().fit(t1Data),
             'minMax': preprocessing.MinMaxScaler().fit(t1Data),
             'None': 1.
             }[preProcess]             
        if preProcess != 'None':
           t1Data = scaler.transform(t1Data)
           if params.meta_bw: t2Data = scaler.transform(t2Data)
           vData = scaler.transform(vData)
           testD = scaler.transform(testD)

    # contrast 
    contrastFun = {
         'tanh': np.tanh,
         'arcsinh': np.arcsinh,
         'sig': sigmoid,
         'None': 1.
         }[preContrast]
    if preContrast != 'None':
       t1Data = contrastFun(t1Data)
       if params.meta_bw: t2Data = contrastFun(t2Data)
       vData = contrastFun(vData)
       testD = contrastFun(testD)


    print '- size T1, valid, T2'
    print t1Data.shape, vData.shape
    if params.meta_bw: print t2Data.shape
        


    # reshape if convnet
    if params.model == 'convnet':
        if params.dataset in ['mnist', 'not_mnist']:
            t1Data = t1Data.reshape(-1, 1, 28, 28)
            vData  =  vData.reshape(-1, 1, 28, 28)
            testD  =  testD.reshape(-1, 1, 28, 28)
            if params.meta_bw: 
                t2Data = t2Data.reshape(-1, 1, 28, 28)    
            
        if params.dataset in ['cifar10', 'svhn']:
            t1Data = t1Data.reshape(-1, 3, 32, 32)
            vData  =  vData.reshape(-1, 3, 32, 32)
            testD  =  testD.reshape(-1, 3, 32, 32)
            if params.meta_bw: 
                t2Data = t2Data.reshape(-1, 3, 32, 32)
                
    # final shape            
    print 'Elementary Set data shape: ', t1Data.shape, t1Label.shape
    if np.sum(np.isinf(t1Data)) > 0 : print 'Nan in T1 data!!'
    if np.sum(np.isinf(t1Label)) > 0 : print 'Nan in T1 label!!'

    if params.meta_bw: 
        print 'Hyper Set data shape: ', t2Data.shape, t2Label.shape
        if np.sum(np.isinf(t2Data)) > 0 : print 'Nan in T2 data!!'
        if np.sum(np.isinf(t2Label)) > 0 : print 'Nan in T2 label!!'
            
#    show_samples(t1Data[:100]/255., 50)    
        
    return t1Data, t1Label, t2Data, t2Label, vData, vLabel, testD, testL

