import cPickle
import os
import tarfile
import urllib
import pickle


import numpy as np

datadir = os.path.expanduser('~/repos/drmad/data/cifar10')

def datapath(fname):
    return os.path.join(datadir, fname)

def cifar10(cifar10_datadir):
    datadir = cifar10_datadir

    if not os.path.isdir(datadir):
        os.mkdir(datadir)

    cifar_download_url = 'http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
    cifar_download_savepath = os.path.join(datadir, os.path.basename(cifar_download_url))

    if not os.path.exists(cifar_download_savepath):
        print('Downloading cifar-10 from {}'.format(cifar_download_url))
        urllib.urlretrieve(cifar_download_url, cifar_download_savepath)
        print('File saved to {}'.format(cifar_download_savepath))

    train_samples_count = 50000

    X_train = np.zeros((train_samples_count, 3, 32, 32), dtype="uint8")
    y_train = np.zeros((train_samples_count,), dtype="uint8")
    X_test = None
    y_test = None

    with tarfile.open(cifar_download_savepath, 'r:gz') as tar_fp:
        for tar_item in tar_fp.getmembers():
            fp = tar_fp.extractfile(tar_item)
            if fp is not None:
                if 'test_batch' not in tar_item.name and 'data_batch' not in tar_item.name:
                    continue

                pickle_data = cPickle.loads(fp.read())
                data = pickle_data['data']
                data = data.reshape(data.shape[0], 3, 32, 32)
                labels = pickle_data['labels']

                if 'data_batch' in tar_item.name:
                    batch_num = int(tar_item.name.strip()[-1])
                    X_train[(batch_num - 1) * 10000: batch_num * 10000, :, :, :] = data
                    y_train[(batch_num - 1) * 10000: batch_num * 10000] = labels
                elif 'test_batch' in tar_item.name:
                    X_test = data
                    y_test = np.asarray(labels)

    y_train = np.reshape(y_train, (len(y_train), 1))
    y_test = np.reshape(y_test, (len(y_test), 1))

    return X_train, y_train, X_test, y_test

def load_data(normalize=False):
    with open(datapath("cifar10_data.pkl")) as f:
        train_images, train_labels, test_images, test_labels = pickle.load(f)

    one_hot = lambda x, K : np.array(x[:,None] == np.arange(K)[None, :], dtype=int)
    partial_flatten = lambda x : np.reshape(x, (x.shape[0], np.prod(x.shape[1:])))
    train_images = partial_flatten(train_images) / 255.0
    test_images  = partial_flatten(test_images)  / 255.0
    train_labels = one_hot(train_labels, 10)
    test_labels = one_hot(test_labels, 10)
    N_data = train_images.shape[0]

    train_labels = train_labels.reshape(-1, 10)
    test_labels = test_labels.reshape(-1, 10)
    if normalize:
        train_mean = np.mean(train_images, axis=0)
        train_images = train_images - train_mean
        test_images = test_images - train_mean
    return train_images, train_labels, test_images, test_labels, N_data



def gz_to_pickle():
    data = cifar10(datadir);
    with open(datapath("cifar10_data.pkl"), "w") as f:
        pickle.dump(data, f, 1)

def load_data_as_dict(normalize=True):
    X, T = load_data(normalize)[:2]
    return {'X' : X, 'T' : T}


    return [{"X" : dat[0], "T" : dat[1]} for dat in datapairs]
if __name__ == '__main__':
    X_train, y_train, X_test, y_test = cifar10(datadir)
    gz_to_pickle();
    print('X_train: {}, y_train: {}', X_train.shape, y_train.shape)
    print('X_test: {}, y_test: {}', X_test.shape, y_test.shape)
