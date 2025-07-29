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
    net = dde.nn.FNN(layers, "tanh", "LeCun normal", use_bias = False) #, regularization=['l2', 1e-8])
    model = dde.Model(data, net)
    # restore model
    model.compile("adam", lr=1e-3, decay = ("inverse time", 20000, 0.5), metrics=["l2 relative error"])
    model.compile("L-BFGS-B",  metrics=["l2 relative error"])
    model.restore("model/model.ckpt-" + best_step + ".ckpt", verbose=1)
    return model

def fixed_point_iteration(model, data, dname, isplot=False):
    # new source function
    f_new = np.loadtxt(f"{dname}/f_new_grid.dat")[:, 1]
    u_new = np.loadtxt(f"{dname}/u_new_grid.dat")[:, 1]
    
    # initial guess
    x = np.loadtxt(f"{dname}/u_0_grid.dat")[:, 0]
    u = np.loadtxt(f"{dname}/u_0_grid.dat")[:, 1]

    errors = [dde.metrics.l2_relative_error(u_new, u)]

    ts = time.time()
    count = 0
    min_err = errors[-1]
    while errors[-1] > 1e-4 and count <= 20000:
        inputs,_ = construct_data(f_new, u)
        u = np.ravel(model.predict(inputs))
        # add boundary
        #u = np.concatenate(([0, (u[0] + 0)/2], u, [(u[-1] + 0)/2,0]))
        u = np.concatenate(([0], u, [0]))
        err = dde.metrics.l2_relative_error(u_new,u)
        if count < 50000 and count % 5000 == 0:
            print(count, err)
        errors.append(err)
        min_err = min(errors[-1],min_err)
        count += 1

    print("Error after {} iterations:".format(count), errors[-1])
    print("Minimum error is ", min_err)
    print("One-shot took {} s.".format(time.time()-ts))
    np.savetxt(f"{dname}/u_FPI.dat",u)
    np.savetxt(f"{dname}/err_FPI.dat",errors)
    if isplot:
        plt.figure()
        plt.plot(x, u_new, "k", label="Ref")
        plt.plot(x, u, "--r", label="NN")
        plt.legend()
        #plt.savefig(f"{dname}/u_FPI.png")
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
    N = 101
    N_f = 101
    N_b = 0
    l, a = 0.01, 0.5
    l_new, a_new = 0.1, float(sigma)

    # Create folders for the datasets
    new_dir = "data_1d_poisson"
    PATH = os.path.join(parent_dir, new_dir)
    
    # Load model
    best_step = "207412"
    pre_layers = [3, 64, 1]
    dataset_G, dataset = load_all_data(M, N, N_f, N_b, l, a, l_new, a_new, 
                                 f"{PATH}/data_{sigma}/data_0", gen, 
                                 correction = False, grid = True, isplot = False)
    model = load_trained_model(dataset_G, pre_layers, best_step)
    
    errs = []
    for i in range(num_func):
        print("Dataset {}".format(i))
        dname = f"{PATH}/data_{sigma}/data_{i}"
        err = fixed_point_iteration(model, dataset, dname, isplot=False)
        errs.append(err)

    print(errs)
    np.savetxt(os.path.join(f"{dname}", "errs_FPI.dat"), errs)
    print("The average l2 error is ", sum(errs)/num_func)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num", type=int, default=3) # Number of functions
    parser.add_argument("--sigma", type=str, default="0.02") # Amplitude in the GRF
    args = parser.parse_args()
    main(args.sigma, args.num)

