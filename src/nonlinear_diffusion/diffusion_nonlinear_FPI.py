import os
import time
import deepxde as dde
import numpy as np
import matplotlib.pyplot as plt
from utils import gen_all_data, construct_data
dde.config.set_default_float("float64")
dde.config.disable_xla_jit()

def load_trained_model(data, layers, best_step):
    net = dde.nn.FNN(layers, "tanh", "LeCun normal", use_bias = True, regularization=['l2', 1e-8])
    model = dde.Model(data, net)
    model.compile("adam", lr=1e-3, decay = ("inverse time", 20000, 0.5), metrics=["l2 relative error"])
    model.compile("L-BFGS-B",  metrics=["l2 relative error"])
    model.restore("model/model.ckpt-" + best_step + ".ckpt", verbose=1)
    return model

def fixed_point_iteration(model, d_num):
    f_new = np.loadtxt("data{}/f_new_grid.dat".format(d_num))
    u_new = np.loadtxt("data{}/u_new_grid.dat".format(d_num))
    u = np.loadtxt("data{}/u_0_grid.dat".format(d_num))
    errors = [dde.metrics.l2_relative_error(u_new, u)]
    Nt, Nx = f_new.shape

    ts = time.time()
    count = 0
    min_err = errors[-1]
    while errors[-1] > 1e-4 and count < 500:
        inputs,_ = construct_data(f_new, u)
        u_outputs = model.predict(inputs)
        k = 0
        for i in range(1, Nx - 1):
            for j in range(1, Nt):
                u[i, j] = u_outputs[k, 0]
                k += 1
        errors.append(dde.metrics.l2_relative_error(u_new,u))
        count += 1
        min_err = min(errors[-1],min_err)

    print("Error after {} iterations:".format(count), errors[-1])
    print("Minimum error is ", min_err)
    print("One-shot took {} s.".format(time.time()-ts))
    np.savetxt("data{}/u_FPI.dat".format(d_num),u)
    np.savetxt("data{}/err_FPI.dat".format(d_num),errors)
    # plot
    plot_errs(errors, d_num)
    plt.rcParams.update({'font.size': 20,"savefig.dpi": 200, "figure.figsize": (8, 6)})
    plt.subplots_adjust(left=0.20, right=0.9, top=0.9, bottom=0.15)
    plt.imshow(np.rot90(abs(u - u_new)), cmap = "gnuplot", extent=(0,1,0,1), aspect='auto')
    plt.colorbar()
    plt.xlabel("x")
    plt.ylabel("t")
    #plt.savefig("data{}/res_FPI.png".format(d_num))
    plt.show()
    return errors[-1]

def plot_errs(errors, d_num):
    plt.figure()
    plt.rcParams.update({'font.size': 20,"savefig.dpi": 200, "figure.figsize": (8, 6)})
    plt.subplots_adjust(left=0.20, right=0.9, top=0.9, bottom=0.15)
    plt.semilogy(errors)
    plt.xlabel("#iterations")
    plt.ylabel("l2 error")
    #plt.savefig("data{}/errors.png".format(d_num))
    plt.show()
    return


def main():
    M = 1001
    Nx, Nt = 101, 101
    N_f = 101*101
    N_b = 0
    std = 0.10  # std for delta f
    std_num = "0.10"
    gen = False  # generate new data or not
    num = 100
    errs = []

    dataset_G, dataset = gen_all_data(M, Nx, Nt, N_f, N_b, 0.01, 0.0, 0.1, 0.10, False, "_0.10/data_0")
    best_step = "161943"
    pre_layers = [4, 64, 1]
    model = load_trained_model(dataset, pre_layers, best_step)

    for i in range(num):
        print("Dataset {}".format(i))
        os.makedirs("data_{}/data_{}".format(std_num, i), exist_ok = True)
        d_num = "_{}/data_{}".format(std_num, i)
        dataset_G, dataset = gen_all_data(M, Nx, Nt, N_f, N_b, 0.01, 0.1, 0.1, std, gen, d_num)
        err = fixed_point_iteration(model, d_num)
        errs.append(err)

if __name__ == "__main__":
    main()

