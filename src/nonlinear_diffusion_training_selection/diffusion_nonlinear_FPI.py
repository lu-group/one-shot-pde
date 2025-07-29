import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import time
import argparse
import numpy as np
import deepxde as dde
from utils import *
dde.backend.set_default_backend('tensorflow.compat.v1')
dde.config.set_default_float("float64")
dde.config.disable_xla_jit()

import sys
import logging
from typing import *
from datetime import datetime
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

class Logger(object):
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        # Restore stdout and close the log file properly
        sys.stdout = self.terminal
        self.log.close()

def load_trained_model(data, layers, best_step):
    net = dde.nn.FNN(layers, "tanh", "Glorot normal", use_bias = True, regularization=['l2', 1e-8])
    model = dde.Model(data, net)
    # restore model
    model.compile("adam", lr=1e-3, decay = ("inverse time", 20000, 0.5), metrics=["l2 relative error"])
    model.compile("L-BFGS-B",  metrics=["l2 relative error"])
    model.restore(f"model/{best_step}.ckpt", verbose=1)
    return model

def fixed_point_iteration(model, dname, ftype, isplot=False):
    # new source function
    f_new = np.loadtxt(f"{dname}/f_new_grid.dat")
    u_new = np.loadtxt(f"{dname}/u_new_grid.dat")
    
    # initial guess
    u = np.loadtxt(f"{dname}/u_0_grid.dat")
    
    errors = [dde.metrics.l2_relative_error(u_new, u)]
    print(errors[0])
    Nt, Nx = f_new.shape

    ts = time.time()
    count = 0
    min_err = errors[-1]
    while errors[-1] > 1e-4 and count < 500:
        inputs,_ = construct_data(f_new, u)
        u_outputs = model.predict(inputs)
        #print(u_outputs)
        k = 0
        for i in range(1, Nx - 1):
            for j in range(1, Nt):
                u[i, j] = u_outputs[k, 0]
                k += 1
        err = dde.metrics.l2_relative_error(u_new,u)
        if count < 500 and count % 100 == 0:
            print(count, err)
        errors.append(err)
        min_err = min(errors[-1],min_err)
        count += 1

    print("Error after {} iterations:".format(count), errors[-1])
    print("Minimum error is ", min_err)
    print("One-shot took {} s.".format(time.time()-ts))
    # save the results
    np.savetxt(f"{dname}/u_FPI_{ftype}.dat",u)
    np.savetxt(f"{dname}/err_FPI_{ftype}.dat",errors)
    # plot
    if isplot:
        plt.figure()
        plt.rcParams.update({'font.size': 20,"savefig.dpi": 200, "figure.figsize": (8, 6)})
        plt.subplots_adjust(left=0.20, right=0.9, top=0.9, bottom=0.15)
        plt.imshow(np.rot90(abs(u - u_new)), cmap = "gnuplot", extent=(0,1,0,1), aspect='auto')
        plt.colorbar()
        plt.xlabel("x")
        plt.ylabel("t")
        plt.savefig(f"{dname}/res_FPI_{ftype}.png")
        plt.show()
        plot_errs(errors, dname, ftype)
    
    return errors[-1]

def plot_errs(errors, dname, ftype):
    plt.figure()
    plt.semilogy(errors)
    plt.xlabel("#iterations")
    plt.ylabel("l2 error")
    plt.savefig(f"{dname}/errors_{ftype}.png")
    plt.show()
    return

def main(sigma, num_func, ftype, parent_dir = "../../data/", gen = False):
    M = 1001
    Nx, Nt = 101, 101
    N_f = 101*101
    N_b = Nx*2 + Nt - 2
    l, a = 0.01, 0.1
    l_new, a_new = 0.1, float(sigma)

    # Create folders for the datasets
    new_dir = "data_nonlinear_diffusion"
    PATH = os.path.join(parent_dir, new_dir)
    
    if ftype == "G":
        best_step = "model-161943" 
    elif ftype == "cos":
        best_step = f"model_{ftype}-152759"
    elif ftype == "sin":
        best_step = f"model_{ftype}-168381"
    elif ftype == "rf":
        best_step = f"model_{ftype}-156782"
    elif ftype == "freq":
        best_step = f"model_{ftype}-152826"
    else:
        print(f"ftype = {ftype} not implemented.")
    pre_layers = [4, 64, 1]
    # load the pre-trained model
    dataset_G, dataset = load_all_data(M, Nx, Nt, N_f, N_b, l, a, l_new, a_new,
                                 f"{PATH}/data_{sigma}/data_0", gen, 
                                 correction = False, grid = True, isplot = False)
    model = load_trained_model(dataset, pre_layers, best_step)
    
    errs = []
    for i in range(num_func):
        print("Dataset {}".format(i))
        dname = f"{PATH}/data_{sigma}/data_{i}"
        err = fixed_point_iteration(model, dname, ftype, isplot=False)
        errs.append(err)
    
    np.savetxt(os.path.join(f"{PATH}/data_{sigma}/", f"errs_FPI_{ftype}.dat"), errs)
    print("The average l2 error is ", sum(errs)/num_func)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num", type=int, default=100) # Number of functions
    parser.add_argument("--sigma", type=str, default="0.80") # Amplitude in the GRF
    parser.add_argument("--ftype", type=str, default="freq")
    args = parser.parse_args()
    
    print(f"Arguments received: num={args.num}, sigma={args.sigma}, ftype={args.ftype}")
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"{LOG_DIR}/log_{args.num}_{args.sigma}_{args.ftype}_{current_time}.log"
    logger = Logger(filename)
    sys.stdout = logger
    
    main(args.sigma, args.num, args.ftype)
    logger.close()

