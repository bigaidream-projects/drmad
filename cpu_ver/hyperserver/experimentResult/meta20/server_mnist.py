"""Runs for paper"""
import matplotlib as mpl
mpl.use('Agg')
import numpy as np
import pickle
from collections import defaultdict
from funkyyak import grad, kylist, getval

import hypergrad.mnist as mnist
from hypergrad.mnist import random_partition
from hypergrad.nn_utils import make_nn_funs, VectorParser
from hypergrad.optimizers import sgd_meta_only_mad as sgd
from hypergrad.util import RandomState, dictslice, dictmap
from hypergrad.odyssey import omap
import hyperParamServerDrmad.loaddataSubClass as loadData



# layer_sizes = [784, 50, 50, 50, 10]
# N_layers = len(layer_sizes) - 1
# batch_size = 50
# N_iters = 2000    # 5000
# N_train_Full = 10000
# N_train = 6000
# N_valid = 3000
# N_tests = 3000
#
# all_N_meta_iter = [0, 0, 20]
# alpha = 0.05  #0.1
# meta_alpha = 0.1
# beta = 0.1
# seed = 0
# N_thin = 500
# N_meta_thin = 1
# log_L2 = -4.0
# log_init_scale = -3.0


layer_sizes = [784, 50,50,50, 10]
N_layers = len(layer_sizes) - 1
batch_size = 50
N_iters = 8000
N_train_Full = 30000
N_train = 12000
N_valid = 5000
N_tests = 5000

all_N_meta_iter = [0, 0, 20]
alpha = 0.01
meta_alpha = 0.2
beta = 0.1
seed = 0
N_thin = 500
N_meta_thin = 1
log_L2 = -4.0
log_init_scale = -3.0

classNum = 10
SubclassNum = 10
clientNum = 5

def run():
    RS = RandomState((seed, "top_rs"))
    data = loadData.loadMnist()

    train_data_subclass = []

    train_data, tests_data = loadData.load_data_as_dict(data, classNum)
    train_data = random_partition(train_data, RS, [N_train_Full]).__getitem__(0)
    tests_data = random_partition(tests_data, RS, [ N_tests]).__getitem__(0)


    train_data_subclass= loadData.loadSubsetData(train_data,RS, N_train, clientNum)

    print "training samples {0}: testing samples: {1}".format(N_train,N_tests)

    w_parser, pred_fun, loss_fun, frac_err = make_nn_funs(layer_sizes)
    N_weights = w_parser.vect.size

    def transform_weights(z_vect, transform):
        return z_vect * np.exp(transform)

    def regularization(z_vect):
        return np.dot(z_vect, z_vect) * np.exp(log_L2)

    def constrain_reg(t_vect, name):
        all_t = w_parser.new_vect(t_vect)
        for i in range(N_layers):
            all_t[('biases', i)] = 0.0
        if name == 'universal':
            t_mean = np.mean([np.mean(all_t[('weights', i)])
                              for i in range(N_layers)])
            for i in range(N_layers):
                all_t[('weights', i)] = t_mean
        elif name == 'layers':
            for i in range(N_layers):
                all_t[('weights', i)] = np.mean(all_t[('weights', i)])
        elif name == 'units':
            for i in range(N_layers):
                all_t[('weights', i)] = np.mean(all_t[('weights', i)], axis=1, keepdims=True)
        else:
            raise Exception
        return all_t.vect

    def process_transform(t_vect):
        # Remove the redundancy due to sharing transformations within units
        all_t = w_parser.new_vect(t_vect)
        new_t = np.zeros((0,))
        for i in range(N_layers):
            layer = all_t[('weights', i)]
            assert np.all(layer[:, 0] == layer[:, 1])
            cur_t = log_L2 - 2 * layer[:, 0]
            new_t = np.concatenate((new_t, cur_t))
        return new_t

    def train_z(data, z_vect_0, transform):
        N_data = data['X'].shape[0]
        def primal_loss(z_vect, transform, i_primal, record_results=False):
            RS = RandomState((seed, i_primal, "primal"))
            idxs = RS.randint(N_data, size=batch_size)
            minibatch = dictslice(data, idxs)
            w_vect = transform_weights(z_vect, transform)
            loss = loss_fun(w_vect, **minibatch)
            reg = regularization(z_vect)
            if record_results and i_primal % N_thin == 0:
                print "Iter {0}: train: {1}".format(i_primal, getval(loss))
            return loss + reg
        return sgd(grad(primal_loss), transform, z_vect_0, alpha, beta, N_iters)

    all_transforms, all_tests_loss, all_tests_rates, all_avg_regs = [], [], [], []
    def train_reg(reg_0, constraint, N_meta_iter, i_top):
        def hyperloss(transform, i_hyper, cur_train_data, cur_valid_data):
            RS = RandomState((seed, i_top, i_hyper, "hyperloss"))
            z_vect_0 = RS.randn(N_weights) * np.exp(log_init_scale)
            z_vect_final = train_z(cur_train_data, z_vect_0, transform)
            w_vect_final = transform_weights(z_vect_final, transform)
            return loss_fun(w_vect_final, **cur_valid_data)
        hypergrad = grad(hyperloss)

        def error_rate(transform, i_hyper, cur_train_data, cur_valid_data):
            RS = RandomState((seed, i_top, i_hyper, "hyperloss"))
            z_vect_0 = RS.randn(N_weights) * np.exp(log_init_scale)
            z_vect_final = train_z(cur_train_data, z_vect_0, transform)
            w_vect_final = transform_weights(z_vect_final, transform)
            return frac_err(w_vect_final, **cur_valid_data)

        cur_reg = reg_0
        for i_hyper in range(N_meta_iter):
            if i_hyper % N_meta_thin == 0:
                test_rate = error_rate(cur_reg, i_hyper, train_data, tests_data)
                all_tests_rates.append(test_rate)
                all_transforms.append(cur_reg.copy())
                all_avg_regs.append(np.mean(cur_reg))
                print "Hyper iter {0}, error rate {1}".format(i_hyper, all_tests_rates[-1])
                print "Cur_transform", np.mean(cur_reg)
            tempConstrained_grad = np.zeros(N_weights)
            for client_i in range (0,clientNum):

                RS = RandomState((seed, i_top, i_hyper, "hyperloss"))
                cur_split = random_partition(train_data_subclass.__getitem__(client_i), RS, [N_train-N_valid, N_valid])
                print("calculate hypergradients")
                raw_grad = hypergrad(cur_reg, i_hyper, *cur_split)
                print("calculate hypergradients end ")

                constrained_grad = constrain_reg(raw_grad, constraint)

                tempConstrained_grad += constrained_grad/clientNum

            cur_reg -= np.sign(tempConstrained_grad) * meta_alpha

            print("calculate hypergradients end ")

        return cur_reg

    reg = np.zeros(N_weights)+0.2
    constraints = ['universal', 'layers', 'units']
    for i_top, (N_meta_iter, constraint) in enumerate(zip(all_N_meta_iter, constraints)):
        print "Top level iter {0}".format(i_top)
        reg = train_reg(reg, constraint, N_meta_iter, i_top)

    all_L2_regs = np.array(zip(*map(process_transform, all_transforms)))
    return all_L2_regs, all_tests_rates, all_avg_regs

def plot():
    import matplotlib.pyplot as plt
    mpl.rcParams['font.family'] = 'serif'
    mpl.rcParams['image.interpolation'] = 'none'
    with open('results.pkl') as f:
        all_L2_regs, all_tests_rates, all_avg_regs = pickle.load(f)

    fig = plt.figure(0)
    fig.clf()
    ax = fig.add_subplot(211)
    color_cycle = ['RoyalBlue', 'DarkOliveGreen', 'DarkOrange', 'MidnightBlue']
    colors = []
    for i, size in enumerate(layer_sizes[:-1]):
        colors += [color_cycle[i]] * size
    for c, L2_reg_curve in zip(colors, all_L2_regs):
        ax.plot(L2_reg_curve, color=c)
    ax.set_ylabel('Log L2 regularization')
    # ax.set_ylim([-3.0, -2.5])

    ax = fig.add_subplot(212)
    ax.plot(all_avg_regs)
    ax.set_ylabel('Average log L2 regularization')

    ax = fig.add_subplot(213)
    ax.plot(all_tests_rates)
    ax.set_ylabel('Test error (%)')
    ax.set_xlabel('Meta iterations')
    plt.savefig("reg_learning_curve.eps", format='eps', dpi=1000)

    initial_filter = np.array(all_L2_regs)[:layer_sizes[0], -1].reshape((28, 28))
    fig.clf()
    fig.set_size_inches((5, 5))
    ax = fig.add_subplot(111)
    ax.matshow(initial_filter, cmap = mpl.cm.binary)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.savefig('bottom_layer_filter.eps', format='eps', dpi=1000)

def experiment():
    # project_dir = os.environ['EXPERI_PROJECT_PATH']
    project_dir ="/home/jie/d2/bitbucket/hypergradient_bo"
    filedir = project_dir+"/hyperParamServerDrmad/experiment/1/17"
    genoutput("")
    result = run( )
    return result
import sys
class Logger(object):
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

def genoutput(path):
    sys.stdout = Logger(path+"outputServerMNIST.txt")

if __name__ == '__main__':
    import time
    t0 = time.time()
    results=experiment()
    with open('resultsServer.pkl', 'w') as f:
        pickle.dump(results, f, 1)
    t1 = time.time()

    total = t1 - t0
    print(total)