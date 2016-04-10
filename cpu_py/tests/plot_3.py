"""Gradient descent to optimize everything"""
"""Aiming for smooth curves by running for a long time with small steps."""
import numpy as np
import pickle

from hypergrad.nn_utils import VectorParser
from hypergrad.nn_utils import logit, inv_logit, d_logit
from hypergrad.optimizers import sgd3, rms_prop, adam, simple_sgd

# ----- Fixed params -----
N_epochs = 50
N_meta_thin = 1  # How often to save meta-curve info.
init_params = np.array([1.4, 3.9])

# ----- Superparameters - aka meta-meta params that control metalearning -----
meta_alpha = 0.010
meta_gamma = 0.9 # Setting this to zero makes things much more stable
N_meta_iter = 3
# ----- Initial values of learned hyper-parameters -----
init_log_alphas = -4.5
init_invlogit_betas = inv_logit(0.99)
init_V0 = 0.0


def make_toy_funs():
    parser = VectorParser()
    parser.add_shape('weights', 2)

    def rosenbrock(w):
        x = w[1:]
        y = w[:-1]
        return sum(100.0*(x-y**2.0)**2.0 + (1-y)**2.0 + 200.0*y)

    def loss(W_vect, X=0.0, T=0.0, L2_reg=0.0):
        return 800 * logit(rosenbrock(W_vect) / 500)

    return parser, loss


def run():
    N_iters = N_epochs
    parser, loss_fun = make_toy_funs()
    N_weight_types = len(parser.names)
    N_weights = parser.vect.size
    hyperparams = VectorParser()
    hyperparams['log_alphas']      = np.full(N_iters, init_log_alphas)
    hyperparams['invlogit_betas']  = np.full(N_iters, init_invlogit_betas)
    hyperparams['V0']  = np.full(N_weights, init_V0)

    all_learning_curves = []
    all_param_curves = []
    all_x = []
    def hyperloss_grad(hyperparam_vect, ii):
        learning_curve = []
        params_curve = []
        def callback(x, i):
            params_curve.append(x)
            learning_curve.append(loss_fun(x))

        def indexed_loss_fun(w, log_L2_reg, j):
            return loss_fun(w)

        cur_hyperparams = hyperparams.new_vect(hyperparam_vect)
        W0 = init_params
        V0 = cur_hyperparams['V0']
        alphas = np.exp(cur_hyperparams['log_alphas'])
        betas = logit(cur_hyperparams['invlogit_betas'])
        log_L2_reg = 0.0
        results = sgd3(indexed_loss_fun, loss_fun, W0, V0,
                       alphas, betas, log_L2_reg, callback=callback)
        hypergrads = hyperparams.copy()
        hypergrads['V0']              = results['dMd_v'] * 0
        hypergrads['log_alphas']      = results['dMd_alphas'] * alphas
        hypergrads['invlogit_betas']  = (results['dMd_betas'] *
                                         d_logit(cur_hyperparams['invlogit_betas']))
        all_x.append(results['x_final'])
        all_learning_curves.append(learning_curve)
        all_param_curves.append(params_curve)
        return hypergrads.vect

    add_fields = ['train_loss', 'valid_loss', 'tests_loss', 'iter_num']
    meta_results = {field : [] for field in add_fields + hyperparams.names}
    def meta_callback(hyperparam_vect, i, g):
        if i % N_meta_thin == 0:
            print "Meta iter {0}".format(i)
            x = all_x[-1]
            cur_hyperparams = hyperparams.new_vect(hyperparam_vect.copy())
            for field in cur_hyperparams.names:
                meta_results[field].append(cur_hyperparams[field])
            meta_results['train_loss'].append(loss_fun(x))
            meta_results['iter_num'].append(i)

    final_result = simple_sgd(hyperloss_grad, hyperparams.vect,
                            meta_callback, N_meta_iter, meta_alpha, meta_gamma)
    meta_results['all_learning_curves'] = all_learning_curves
    meta_results['all_param_curves'] = all_param_curves
    parser.vect = None # No need to pickle zeros
    return meta_results, parser

def plot():
    _, loss_fun = make_toy_funs()

    from mpl_toolkits.mplot3d import proj3d

    def orthogonal_proj(zfront, zback):
        a = (zfront+zback)/(zfront-zback)
        b = -2*(zfront*zback)/(zfront-zback)
        # -0.0001 added for numerical stability as suggested in:
        # http://stackoverflow.com/questions/23840756
        return np.array([[1,0,0,0],
                         [0,1,0,0],
                         [0,0,a,b],
                         [0,0,-0.0001,zback]])

    import matplotlib.pyplot as plt
    with open('results.pkl') as f:
        results, parser = pickle.load(f)
    plt.rcParams['figure.figsize'] = (9, 4)
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.size'] = '14'
    plt.rcParams['axes.labelsize'] = '14'
    plt.rcParams['axes.titlesize'] = '16'
    fig = plt.figure(0)
    fig.set_size_inches((9,4))
    fig = plt.figure(0)
    fig.set_size_inches((6,4))
    alpha=0.7
    # Show loss surface.
    x = np.arange(-1.0, 2.4, 0.05)
    y = np.arange(-0.0, 4.5, 0.05)
    X, Y = np.meshgrid(x, y)
    zs = np.array([loss_fun(np.concatenate(([x],[y]))) for x,y in zip(np.ravel(X), np.ravel(Y))])
    Z = zs.reshape(X.shape)

    v = np.linspace(-.1, 2.0, 15, endpoint=True)
    plt.contour(X[1], Y[:,0], Z, 25, linewidths=0.3, colors='black', alpha=alpha)
    plt.contourf(X[1], Y[:,0], Z, 25, cmap=plt.cm.RdYlGn, alpha=alpha)
    # x = plt.colorbar(ticks=v)
    # print x


    #
    # ax = fig.add_subplot(111, projection='3d')
    # ax.plot_wireframe(X, Y, Z, rstride=10, cstride=10, color='Black')
    # proj3d.persp_transformation = orthogonal_proj
    #
    colors = ['Red', 'Green', 'Blue']

    alpha1=0.8
    # ----- Primal learning curves -----
    for i, z in zip(results['iter_num'], results['all_learning_curves']):
        if i !=1:
            x, y = zip(*results['all_param_curves'][i])
            if i == 0:
                plt.plot([x[1]], [y[1]], '^', color='Black', label="Initial weights", markersize=10, alpha=alpha1)
            plt.plot(x, y, '-o', color=colors[i], markersize=2, linewidth=2, alpha=alpha1)
            if i ==2:
                plt.plot([x[-1]], [y[-1]], 'o', color=colors[i], label='Meta-iteration {0}'.format(i), markersize=9, alpha=alpha1)
            else:
                plt.plot([x[-1]], [y[-1]], 'o', color=colors[i], label='Meta-iteration {0}'.format(i+1), markersize=9, alpha=alpha1)
                        
            plt.legend(numpoints=1, loc=0, frameon=True,  bbox_to_anchor=(0.4, 0.95),
                  borderaxespad=0.0, prop={'family':'serif', 'size':'8'})
            if i ==0:
                plt.annotate("RMAD & Forward DrMAD", xy=(1.5, 1.3), xytext=(0.7, 0.5),
                             bbox=dict(boxstyle="round", fc=(1.0, 1.0, 1.0), ec="none"),
                             arrowprops=dict(facecolor='black', shrink=0.05, width=2), size=10)
            plt.xlabel('Elementary parameter 1')
            plt.ylabel('Elementary parameter 2')
            index = 40
            X = np.linspace(x[1], x[-1], index)
            Y = np.linspace(y[1], y[-1], index)

#            for k in range(0, index):
#                Y[k]=Y[0]+(pow(4,(X[k]-X[0]))-1)*(Y[index-1]-Y[0])/(pow(4,(X[index-1]-X[0]))-1)

            plt.plot(X, Y,'-o', color=colors[i], markersize=2, linewidth=2, alpha=alpha1)
            if i ==0:
                plt.annotate("DrMAD Backward", xy=(0.3, 1.9),xytext=(-0.7, 2.5),
                             bbox=dict(boxstyle="round", fc=(1.0, 1.0, 1.0), ec="none"),
                             arrowprops=dict(facecolor='black', shrink=0.05, width=2), size=10)

    # plt.show()

    frame = plt.gca()
    frame.axes.get_xaxis().set_ticks([])
    frame.axes.get_yaxis().set_ticks([])
    plt.savefig('learning_curves.png',dpi=1000, alpha=alpha)
    plt.savefig('learning_curves.pdf', pad_inches=0.05,dpi=1000, bbox_inches='tight', alpha=alpha)

if __name__ == '__main__':
    #results = run()
    #with open('results.pkl', 'w') as f:
    #    pickle.dump(results, f)
    plot()