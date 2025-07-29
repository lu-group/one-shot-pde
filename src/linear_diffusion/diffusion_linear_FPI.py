import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import time
import argparse
import deepxde as dde
import numpy as np
import matplotlib.pyplot as plt
dde.config.set_default_float("float64")
from utils import load_all_data, construct_data

def load_trained_model(data, layers, best_step):
    net = dde.nn.FNN(layers, "tanh", "LeCun normal", regularization=['l2', 1e-8])
    model = dde.Model(data, net)
    model.compile("adam", lr=1e-3, decay = ("inverse time", 20000, 0.5), metrics=["l2 relative error"])
    model.compile("L-BFGS-B",  metrics=["l2 relative error"])
    model.restore("model/model.ckpt-" + best_step + ".ckpt", verbose=1)
    return model

def fixed_point_iteration(model, dataset, dname, isplot=False):
    # new source function
    f_new = np.loadtxt(f"{dname}/f_new_grid.dat")
    u_new = np.loadtxt(f"{dname}/u_new_grid.dat")
    
    # initial guess
    u = np.loadtxt(f"{dname}/u_0_grid.dat")

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
    np.savetxt(f"{dname}/u_FPI.dat",u)
    np.savetxt(f"{dname}/err_FPI.dat",errors)
    if isplot:
        plt.rcParams.update({'font.size': 20,"savefig.dpi": 200, "figure.figsize": (8, 6)})
        plt.subplots_adjust(left=0.20, right=0.9, top=0.9, bottom=0.15)
        plt.imshow(np.rot90(abs(u - u_new)), cmap = "gnuplot", extent=(0,1,0,1), aspect='auto')
        plt.colorbar()
        plt.xlabel("x")
        plt.ylabel("t")
        # plt.savefig(f"{dname}/res_FPI.png")
        plt.show()
        plot_errs(errors, dname)
    return errors[-1]

def plot_errs(errors, dname):
    plt.figure()
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
    new_dir = "data_linear_diffusion"
    PATH = os.path.join(parent_dir, new_dir)

    # Load model
    best_step = "157037"
    pre_layers = [4, 64, 1]
    dataset_G, dataset = load_all_data(M, Nx, Nt, N_f, N_b, l, a, l_new, a_new, 
                                 f"{PATH}/data_{sigma}/data_0", gen, 
                                 correction = False, grid = True, isplot = False)
    model = load_trained_model(dataset_G, pre_layers, best_step)
    
    errs = []
    for i in range(num_func):
        print("Dataset {}".format(i))
        dname = f"{PATH}/data_{sigma}/data_{i}"
        err = fixed_point_iteration(model, dataset, dname, isplot=True)
        errs.append(err)

    print(errs)
    np.savetxt(os.path.join(f"{dname}", "errs_FPI.dat"), errs)
    print("The average l2 error is ", sum(errs)/num_func)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num", type=int, default=3) # Number of functions
    parser.add_argument("--sigma", type=str, default="0.05") # Amplitude in the GRF
    args = parser.parse_args()
    main(args.sigma, args.num)
