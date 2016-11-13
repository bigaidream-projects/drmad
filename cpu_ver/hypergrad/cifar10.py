import cPickle
import os
import tarfile
import urllib

import numpy as np


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

if __name__ == '__main__':
    cifar10_datadir = os.path.normpath(os.path.join(os.getcwd(), '..', 'data', 'cifar10'))
    X_train, y_train, X_test, y_test = cifar10(cifar10_datadir)
    print('X_train: {}, y_train: {}', X_train.shape, y_train.shape)
    print('X_test: {}, y_test: {}', X_test.shape, y_test.shape)
