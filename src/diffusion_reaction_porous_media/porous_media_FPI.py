import os
import time
import deepxde as dde
import numpy as np
import matplotlib.pyplot as plt
from utils import gen_all_data, construct_data
dde.config.set_default_float("float64")

def load_trained_model(data, layers, best_step):
    net = dde.nn.FNN(layers, "tanh", "LeCun normal", use_bias = True, regularization=['l2', 1e-8])
    model = dde.Model(data, net)
    model.compile("adam", lr=1e-3, decay = ("inverse time", 20000, 0.5), metrics=["l2 relative error"])
    model.compile("L-BFGS-B",  metrics=["l2 relative error"])
    model.restore("model/model.ckpt-" + best_step + ".ckpt", verbose=1)
    return model

def fixed_point_iteration(model, d_num):
    # new source function
    f_new = np.loadtxt("data{}/f_new_grid.dat".format(d_num))
    Ca_new = np.loadtxt("data{}/Ca_new_grid.dat".format(d_num))
    Cb_new = np.loadtxt("data{}/Cb_new_grid.dat".format(d_num))
    # initial guess
    Ca = np.loadtxt("data{}/Ca_0_grid.dat".format(d_num))
    Cb = np.loadtxt("data{}/Cb_0_grid.dat".format(d_num))
    x = np.loadtxt("data{}/x_grid.dat".format(d_num))
    t = np.loadtxt("data{}/t_grid.dat".format(d_num))
    errors_a = [dde.metrics.l2_relative_error(Ca_new, Ca)]
    errors_b = [dde.metrics.l2_relative_error(Cb_new, Cb)]
    Nt, Nx = f_new.shape

    ts = time.time()
    count = 0
    min_err_a = errors_a[-1]
    min_err_b = errors_b[-1]
    while errors_a[-1] > 1e-4 and errors_b[-1] > 1e-4 and count < 500:
        inputs,_, _ = construct_data(f_new, Ca, Cb)
        u_outputs = model.predict(inputs)
        Ca_outputs = u_outputs[:, 0:1]
        Cb_outputs = u_outputs[:, 1:2]
        k = 0
        for i in range(1, Nx-1):
            for j in range(1, Nt):
                Ca[i, j] = Ca_outputs[k, 0]
                Cb[i, j] = Cb_outputs[k, 0]
                k += 1
        errors_a.append(dde.metrics.l2_relative_error(Ca_new,Ca))
        errors_b.append(dde.metrics.l2_relative_error(Cb_new,Cb))
        count += 1
        min_err_a = min(errors_a[-1],min_err_a)
        min_err_b = min(errors_b[-1],min_err_b)

    print("Error of Ca after {} iterations:".format(count), errors_a[-1])
    print("Minimum error is ", min_err_a)
    print("Error of Cb after {} iterations:".format(count), errors_b[-1])
    print("Minimum error is ", min_err_b)
    print("One-shot took {} s.".format(time.time()-ts))
    # save the results
    np.savetxt("data{}/Ca_FPI.dat".format(d_num),Ca)
    np.savetxt("data{}/Cb_FPI.dat".format(d_num),Cb)
    np.savetxt("data{}/err_FPI_a.dat".format(d_num),errors_a)
    np.savetxt("data{}/err_FPI_b.dat".format(d_num),errors_b)
    # plot
    plot_errs(errors_a, d_num)
    plot_errs(errors_b, d_num)
    """
    plt.imshow(np.rot90(Ca), cmap = "gnuplot", extent=(0,1,0,1), aspect='auto')
    plt.colorbar()
    plt.xlabel("x")
    plt.ylabel("t")
    plt.show()

    plt.imshow(np.rot90(Ca_new), cmap = "gnuplot", extent=(0,1,0,1), aspect='auto')
    plt.colorbar()
    plt.xlabel("x")
    plt.ylabel("t")
    plt.show()
    
    plt.rcParams.update({'font.size': 20,"savefig.dpi": 200, "figure.figsize": (8, 6)})
    plt.subplots_adjust(left=0.20, right=0.9, top=0.9, bottom=0.15)
    plt.imshow(np.rot90(abs(Ca - Ca_new)), cmap = "gnuplot", extent=(0,1,0,1), aspect='auto')
    plt.colorbar()
    plt.xlabel("x")
    plt.ylabel("t")
    #plt.savefig("data{}/res_FPI.png".format(d_num))
    plt.show()
    """
    return errors_a[-1], errors_b[-1]

def plot_errs(errors, d_num):
    plt.figure()
    plt.rcParams.update({'font.size': 20,"savefig.dpi": 200, "figure.figsize": (8, 6)})
    plt.subplots_adjust(left=0.20, right=0.9, top=0.9, bottom=0.15)
    plt.semilogy(errors)
    plt.xlabel("#iterations")
    plt.ylabel("l2 error")
    plt.savefig("data{}/errors.png".format(d_num))
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
    errs_a = []
    errs_b = []

    # load the pre-trained model
    dataset_G, dataset = gen_all_data(M, Nx, Nt, N_f, N_b, 0.01, 0.0, 0.1, 0.10, False, "_0.10/data_0")
    best_step = "117616"
    pre_layers = [7, 64, 2]
    model = load_trained_model(dataset, pre_layers, best_step)

    for i in range(num):
        print("Dataset {}".format(i))
        os.makedirs("data_{}/data_{}".format(std_num, i), exist_ok = True)
        d_num = "_{}/data_{}".format(std_num, i)
        dataset_G, dataset = gen_all_data(M, Nx, Nt, N_f, N_b, 0.01, 0.1, 0.1, std, gen, d_num)
        pre_layers = [7,64,2]
        err_a, err_b = fixed_point_iteration(model, d_num)
        errs_a.append(err_a)
        errs_b.append(err_b)

    print(errs_a, errs_b)
    np.savetxt("data_{}/errs_FPI_a.dat".format(std_num),errs_a)
    np.savetxt("data_{}/errs_FPI_b.dat".format(std_num),errs_b)
    print("The average l2 errora are ", sum(errs_a)/num, " and ", sum(errs_b)/num)

if __name__ == "__main__":
    main()

