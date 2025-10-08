import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import time
import argparse
import deepxde as dde
import numpy as np
import matplotlib.pyplot as plt
dde.config.set_default_float("float64")
dde.config.disable_xla_jit()
from utils import load_all_data, construct_data

def load_trained_model(data, layers, best_step):
    net = dde.nn.FNN(layers, "tanh", "LeCun normal", use_bias = True, regularization=['l2', 1e-8])
    model = dde.Model(data, net)
    model.compile("adam", lr=1e-3, decay = ("inverse time", 20000, 0.5), metrics=["l2 relative error"])
    model.compile("L-BFGS-B",  metrics=["l2 relative error"])
    model.restore("model/model.ckpt-" + best_step + ".ckpt", verbose=1)
    return model

def fixed_point_iteration(model, dataset, dname, isplot=False):
    # new source function
    f_new = np.loadtxt(f"{dname}/f_new_grid.dat")
    Ca_new = np.loadtxt(f"{dname}/Ca_new_grid.dat")
    Cb_new = np.loadtxt(f"{dname}/Cb_new_grid.dat")
    # initial guess
    Ca = np.loadtxt(f"{dname}/Ca_0_grid.dat")
    Cb = np.loadtxt(f"{dname}/Cb_0_grid.dat")
    x = np.loadtxt(f"{dname}/x_grid.dat")
    t = np.loadtxt(f"{dname}/t_grid.dat")
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
    np.savetxt(f"{dname}/Ca_FPI.dat", Ca)
    np.savetxt(f"{dname}/Cb_FPI.dat", Cb)
    np.savetxt(f"{dname}/err_FPI_a.dat", errors_a)
    np.savetxt(f"{dname}/err_FPI_b.dat", errors_b)
    # plot
    if isplot:
        plot_errs(errors_a, dname)
        plot_errs(errors_b, dname)
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
        plt.show()
        """
    return errors_a[-1], errors_b[-1]

def plot_errs(errors, dname):
    plt.figure()
    plt.rcParams.update({'font.size': 20,"savefig.dpi": 200, "figure.figsize": (8, 6)})
    plt.subplots_adjust(left=0.20, right=0.9, top=0.9, bottom=0.15)
    plt.semilogy(errors)
    plt.xlabel("#iterations")
    plt.ylabel("l2 error")
    plt.savefig(f"{dname}/errors.png")
    plt.show()
    return


def main(sigma, num_func, parent_dir = "../../data/", gen = False):
    M = 1001 # Number of points 
    Nx, Nt = 101, 101
    N_f = 101*101
    N_b = Nx*2 + Nt - 2
    l, a = 0.01, 0.1
    l_new, a_new = 0.1, float(sigma)

    # Create folders for the datasets
    new_dir = "data_porous_media"
    PATH = os.path.join(parent_dir, new_dir)
    
    # Load model
    best_step = "117616"
    pre_layers = [7, 64, 2]
    dataset_G, dataset = load_all_data(M, Nx, Nt, N_f, N_b, l, a, l_new, a_new, 
                                 f"{PATH}/data_{sigma}/data_0", gen, 
                                 correction = False, grid = True, isplot = False)
    model = load_trained_model(dataset_G, pre_layers, best_step)
    
    errs_a = []
    errs_b = []
    for i in range(num_func):
        print("Dataset {}".format(i))
        dname = f"{PATH}/data_{sigma}/data_{i}"
        err_a, err_b = fixed_point_iteration(model, dataset, dname, isplot=True)
        errs_a.append(err_a)
        errs_b.append(err_b)
    
    print(errs_a, errs_b)
    np.savetxt(os.path.join(f"{dname}", "errs_FPI_a.dat"), errs_a)
    np.savetxt(os.path.join(f"{dname}", "errs_FPI_b.dat"), errs_b)
    print("The average l2 errors are ", sum(errs_a)/num_func, " and ", sum(errs_b)/num_func)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num", type=int, default=3) # Number of functions
    parser.add_argument("--sigma", type=str, default="0.10") # Amplitude in the GRF
    args = parser.parse_args()
    main(args.sigma, args.num)

