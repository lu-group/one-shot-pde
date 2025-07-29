import time
import deepxde as dde
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
dde.config.set_default_float("float64")

fontsize=25
fig_params = {
    'font.size': fontsize,
    "savefig.dpi": 300, 
    "figure.figsize": (8, 6),
    'lines.linewidth': 2.5,
    'axes.linewidth': 2.5,
    'axes.titlesize' : fontsize+3,
    "axes.labelsize":fontsize+5,
    "xtick.labelsize":fontsize,
    "ytick.labelsize":fontsize,
    'xtick.direction':'in',
    'ytick.direction':'in',
    'xtick.major.size': 7,
    'xtick.minor.size': 5,
    'xtick.major.width': 3,
    'xtick.minor.width': 2,
    'xtick.major.pad': 6,
    'xtick.minor.pad': 5,
    'ytick.major.pad': 6,
    'ytick.minor.pad': 5,
    'ytick.major.size': 7,
    'ytick.minor.size': 5,
    'ytick.major.width': 3,
    'ytick.minor.width': 2,
    "mathtext.fontset":"cm"
}
plt.rcParams.update(fig_params)

def construct_more_data(N, f, u):
    M = len(f)
    step = int((M-1)/(N-1))
    outputs = u[step: -step].reshape((-1, 1))
    inputs = np.hstack((u[:-step*2].reshape((-1, 1)), u[2*step:].reshape((-1, 1)), f[step:-step].reshape((-1, 1))))
    return inputs, outputs

def construct_data(f, u):
    N = len(f)
    inputs = []
    outputs = []
    for i in range(1, N - 1):
        inputs.append([u[i - 1], u[i + 1], f[i]])
        outputs.append([u[i]])
    return np.array(inputs), np.array(outputs)

def load_trained_model(data, layers, best_step):
    net = dde.nn.FNN(layers, "tanh", "LeCun normal", use_bias = False) #, regularization=['l2', 1e-8])
    model = dde.Model(data, net)
    model.compile("adam", lr=1e-3, decay = ("inverse time", 20000, 0.5), metrics=["l2 relative error"])
    model.compile("L-BFGS-B",  metrics=["l2 relative error"])
    model.restore("model/model.ckpt-" + best_step + ".ckpt", verbose=1)
    return model

def LR_model(data):
    model = LinearRegression(fit_intercept=False).fit(data.train_x, data.train_y)
    print("Train MSE:", dde.metrics.mean_squared_error(data.train_y, model.predict(data.train_x)))
    print("Test MSE:",dde.metrics.mean_squared_error(data.test_y, model.predict(data.test_x)))
    print("Test L2 relative error:",dde.metrics.l2_relative_error(data.test_y, model.predict(data.test_x)))
    print("Model:", model.coef_, model.intercept_)
    return model

def fixed_point_iteration(model, d_num, method):
    # new source function
    f_new = np.loadtxt("data{}/f_new_grid.dat".format(d_num))[:, 1]
    u_new = np.loadtxt("data{}/u_new_grid.dat".format(d_num))[:, 1]
    # initial guess
    x = np.loadtxt("data{}/u_0_grid.dat".format(d_num))[:, 0]
    u = np.loadtxt("data{}/u_0_grid.dat".format(d_num))[:, 1]
    errors = [dde.metrics.l2_relative_error(u_new, u)]

    ts = time.time()
    count = 0
    min_err = errors[-1]
    while errors[-1] > 1e-4 and count <= 50000:
        inputs,_ = construct_data(f_new, u)
        u = np.ravel(model.predict(inputs))
        u = np.concatenate(([0], u, [0]))
        err = dde.metrics.l2_relative_error(u_new,u)
        if count < 20000 and count % 5000 == 0:
            print(count, err)
        errors.append(err)
        min_err = min(errors[-1],min_err)
        count += 1
    print("Error after {} iterations:".format(count), errors[-1])
    print("Minimum error is ", min_err)
    print("One-shot took {} s.".format(time.time()-ts))
    np.savetxt("data{}/u_FPI_{}.dat".format(d_num, method),u)
    np.savetxt("data{}/err_FPI_{}.dat".format(d_num, method),errors)
    return errors

def fixed_point_iteration_wt(d_num):
    # new source function
    f_new = np.loadtxt("data{}/f_new_grid.dat".format(d_num))[:, 1]
    u_new = np.loadtxt("data{}/u_new_grid.dat".format(d_num))[:, 1]
    # initial guess
    x = np.loadtxt("data{}/u_0_grid.dat".format(d_num))[:, 0]
    u = np.loadtxt("data{}/u_0_grid.dat".format(d_num))[:, 1]
    errors = [dde.metrics.l2_relative_error(u_new, u)]

    ts = time.time()
    count = 0
    min_err = errors[-1]
    while errors[-1] > 1e-14 and count <= 50000:
        inputs,_ = construct_data(f_new, u)
        u = (inputs[:, 0] + inputs[:, 1] - (1/100)**2*100*inputs[:, 2]) / 2
        u = np.concatenate(([0], u, [0]))
        err = dde.metrics.l2_relative_error(u_new,u)
        if count % 5000 == 0:
            print(count, err)
        errors.append(err)
        min_err = min(errors[-1],min_err)
        count += 1
    print("Error after {} iterations:".format(count), errors[-1])
    print("Minimum error is ", min_err)
    print("One-shot took {} s.".format(time.time()-ts))
    np.savetxt("data{}/u_FPI_fdm.dat".format(d_num),u)
    np.savetxt("data{}/err_FPI_fdm.dat".format(d_num),errors)
    return errors

def plot_errs(errors, d_num):
    plt.figure()
    plt.rcParams.update({'font.size': 25,"savefig.dpi": 200, "figure.figsize": (8, 6)})
    plt.subplots_adjust(left=0.18, right=0.95, top=0.9, bottom=0.18)
    plt.semilogy(errors)
    plt.yscale("log")
    plt.xlabel("No. of iterations")
    plt.ylabel("$L^2$ relative error")
    plt.show()
    return

def main():
    N = 101
    std = 0.15
    std_num = "0.15"
    method = "Truth"  #NN, Truth, LR
    i = 0
    d_num = "/data_{}_{}".format(i,std_num)
    f_T = np.loadtxt("data{}/f_T.dat".format(d_num))[:, 1]
    u_T = np.loadtxt("data{}/u_T.dat".format(d_num))[:, 1]
    d_T = construct_more_data(N, f_T, u_T)

    f_0 = np.loadtxt("data{}/f_0.dat".format(d_num))[:, 1]
    u_0 = np.loadtxt("data{}/u_0.dat".format(d_num))[:, 1]
    d_0 = construct_more_data(N, f_0, u_0)
    dataset_G = dde.data.DataSet(X_train=d_T[0], y_train=d_T[1], X_test=d_0[0], y_test=d_0[1])

    best_step = "207412"
    pre_layers = [3, 64, 1]
    for method in  ["Truth", "LR", "NN"]:  #"Truth", "LR"
        if method == "NN":
            model = load_trained_model(dataset_G, pre_layers, best_step)
            errs_nn = fixed_point_iteration(model, d_num, method)
            err_nn = errs_nn[-1]
        elif method == "LR":
            model = LR_model(dataset_G)
            errs_lr = fixed_point_iteration(model, d_num, method)
            err_lr = errs_lr[-1]
        elif method == "Truth":
            errs_t = fixed_point_iteration_wt(d_num)
            err_t = errs_t[-1]
    print(err_t, err_lr, err_nn)
    plt.figure()
    plt.rcParams.update({'font.size': 25,"savefig.dpi": 200, "figure.figsize": (8, 6)})
    plt.subplots_adjust(left=0.18, right=0.95, top=0.9, bottom=0.18)
    #plt.plot(errs_t, label = "FDM", c = 'g')
    plt.plot(errs_lr, label = "LR", c = 'b')
    plt.plot(errs_nn, label = "NN", c = 'red')
    plt.ylim([-0.01, 0.6])
    #plt.ylim([0, 0.15])
    plt.legend(fontsize=fontsize, frameon=False)
    #plt.yscale("log")
    plt.xlabel("No. of iterations")
    plt.ylabel("$L^2$ relative error")
    plt.show()


if __name__ == "__main__":
    i = 0
    d_num = "/data_{}_{}".format(i, "0.15")
    x = np.loadtxt("data{}/u_new_grid.dat".format(d_num))[:, 0]
    u_ref = np.loadtxt("data{}/u_new_grid.dat".format(d_num))[:, 1]
    f_new = np.loadtxt("data{}/f_new_grid.dat".format(d_num))[:, 1]
    u_fdm = np.loadtxt("data{}/u_FPI_fdm.dat".format(d_num))
    u_lr = np.loadtxt("data{}/u_FPI_lr.dat".format(d_num))
    u_nn = np.loadtxt("data{}/u_FPI_nn.dat".format(d_num))

    plt.rcParams.update({'font.size': fontsize,"savefig.dpi": 200, "figure.figsize": (8, 6)})
    plt.subplots_adjust(left=0.18, right=0.95, top=0.9, bottom=0.18)
    plt.plot(x, u_ref, "black", label = "Reference", linewidth=3.5)
    #plt.plot(x, u_fdm, "g-." ,label = "FDM", linewidth=2.5)
    plt.plot(x, u_nn, "r-" ,label = "NN", linewidth=2.5)
    plt.plot(x, u_lr, "b--" ,label = "LR", linewidth=2.5)
    plt.xlabel("$x$")
    plt.ylabel("$u$")
    plt.legend(fontsize=fontsize-6, frameon=False, loc = "upper left", labelspacing =0.2, bbox_to_anchor=(0.006, 1.02))
    plt.show()
    main()