import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import time
import argparse
import deepxde as dde
import numpy as np
import matplotlib.pyplot as plt
dde.config.set_default_float("float64")
dde.config.disable_xla_jit()
from utils import *

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
    net = dde.nn.FNN(layers, "tanh", "Glorot uniform", use_bias = True) #, regularization=['l2', 1e-8])
    model = dde.Model(data, net)
    model.compile("adam", lr=1e-3, decay = ("inverse time", 20000, 0.5), metrics=["l2 relative error"])
    model.compile("L-BFGS-B",  metrics=["l2 relative error"])
    model.restore(f"model/{best_step}.ckpt", verbose=1)
    return model

def plot_errs(errors, dname, size):
    plt.figure()
    plt.semilogy(errors)
    plt.xlabel("#iterations")
    plt.ylabel("l2 error")
    plt.savefig(f"{dname}/errors_{size}.png")
    #plt.show()
    return

def fixed_point_iteration(model, size, dname, isplot=False):
    # new source function
    f_new = np.loadtxt(f"{dname}/f_new_grid.dat")
    u_new = np.loadtxt(f"{dname}/u_new_grid.dat")
    # initial guess
    u = np.loadtxt(f"{dname}/u_0_grid.dat")
    errors = [dde.metrics.l2_relative_error(u_new, u)]
    print(errors[0])
    Nx, Ny = f_new.shape
    
    if size == 5:
        construct_more_data = construct_more_data5
    elif size == 9:
        construct_more_data = construct_more_data9
    elif size == 13:
        construct_more_data = construct_more_data13_5
    elif size == 25:
        construct_more_data = construct_more_data25_9
    elif size == 49:
        construct_more_data = construct_more_data49_25
    elif size == 81:
        construct_more_data = construct_more_data81_49
    
    _,outputs = construct_more_data(Nx, Ny, f_new, u)
    if outputs.shape[1] == 5:
        print("outputs.shape[1]=", outputs.shape[1])
        ts = time.time()
        count = 0
        min_err = errors[-1]
        while errors[-1] > 1e-4 and count < 15000:
            inputs,_ = construct_more_data(Nx, Ny, f_new, u)
            u_outputs = model.predict(inputs)
            # print(u_outputs.shape, u.shape)
            dist = int(abs(np.sqrt(inputs.shape[0]) - np.sqrt(Nx*Ny))/2)
            k = 0
            for i in range(dist, Nx - dist):
                for j in range(dist, Ny - dist):
                    if i == dist and j == dist:
                        u[i-1, j-1] = (u_outputs[k, 0] + u_outputs[k, 1])/2
                    elif i == dist and j == Ny - dist:
                        u[i-1, j+1] = (u_outputs[k, 0] + u_outputs[k, 3])/2
                    elif i == Nx - dist and j == dist:
                        u[i+1, j-1] = (u_outputs[k, 2] + u_outputs[k, 1])/2
                    elif i == Nx - dist and j == Ny - dist:
                        u[i+1, j+1] = (u_outputs[k, 2] + u_outputs[k, 3])/2
                        
                    if i == dist:
                        u[i-1, j] = u_outputs[k, 0]
                    elif j == dist:
                        u[i, j-1] = u_outputs[k, 1]
                    elif i == Nx-dist:
                        u[i+1, j] = u_outputs[k, 2]
                    elif j == Ny-dist:
                        u[i, j+1] = u_outputs[k, 3]
                    u[i, j] = u_outputs[k, -1]
                    k += 1
            errors.append(dde.metrics.l2_relative_error(u_new,u))
            count += 1
            if count % 1000 == 0:
                print(count,errors[-1])
            min_err = min(errors[-1],min_err)
    elif outputs.shape[1] == 9:
        print("outputs.shape[1]=", outputs.shape[1])
        ts = time.time()
        count = 0
        min_err = errors[-1]
        while errors[-1] > 1e-4 and count < 15000:
            inputs,_ = construct_more_data(Nx, Ny, f_new, u)
            u_outputs = model.predict(inputs)
            # print(u_outputs.shape, u.shape)
            dist = int(abs(np.sqrt(inputs.shape[0]) - np.sqrt(Nx*Ny))/2)
            k = 0
            for i in range(dist, Nx - dist):
                for j in range(dist, Ny - dist):
                    if i == dist and j == dist:
                        u[i-1, j-1] = u_outputs[k, 4]
                    elif i == dist and j == Ny - dist:
                        u[i-1, j+1] = u_outputs[k, 5]
                    elif i == Nx - dist and j == dist:
                        u[i+1, j-1] = u_outputs[k, 6]
                    elif i == Nx - dist and j == Ny - dist:
                        u[i+1, j+1] = u_outputs[k, 7]
                    
                    if i == dist:
                        u[i-1, j] = u_outputs[k, 3]
                    elif j == dist:
                        u[i, j-1] = u_outputs[k, 0]
                    elif i == Nx-dist:
                        u[i+1, j] = u_outputs[k, 1]
                    elif j == Ny-dist:
                        u[i, j+1] = u_outputs[k, 2]
                    u[i, j] = u_outputs[k, -1]
                    k += 1
            errors.append(dde.metrics.l2_relative_error(u_new,u))
            count += 1
            if count % 1000 == 0:
                print(count,errors[-1])
            min_err = min(errors[-1],min_err)
    elif outputs.shape[1] == 25:
        print("outputs.shape[1]=", outputs.shape[1])
        ts = time.time()
        count = 0
        min_err = errors[-1]
        while errors[-1] > 1e-4 and count < 15000:
            inputs,_ = construct_more_data(Nx, Ny, f_new, u)
            u_outputs = model.predict(inputs)
            # print(u_outputs.shape, u.shape)
            dist = int(abs(np.sqrt(inputs.shape[0]) - np.sqrt(Nx*Ny))/2)
            k = 0
            for i in range(dist, Nx - dist):
                for j in range(dist, Ny - dist):
                    if i == dist and j == dist:
                        u[i-1, j-1] = u_outputs[k, 18]
                        u[i-2, j-1] = u_outputs[k, 23]
                        u[i-1, j-2] = u_outputs[k, 19]
                        u[i-2, j-2] = u_outputs[k, 24]
                    elif i == dist and j == Ny - dist:
                        u[i-1, j+1] = u_outputs[k, 16]
                        u[i-2, j+1] = u_outputs[k, 21]
                        u[i-1, j+2] = u_outputs[k, 15]
                        u[i-2, j+2] = u_outputs[k, 20]
                    elif i == Nx - dist and j == dist:
                        u[i+1, j-1] = u_outputs[k, 8]
                        u[i+2, j-1] = u_outputs[k, 3]
                        u[i+1, j-2] = u_outputs[k, 9]
                        u[i+2, j-2] = u_outputs[k, 4]
                    elif i == Nx - dist and j == Ny - dist:
                        u[i+1, j+1] = u_outputs[k, 6]
                        u[i+2, j+1] = u_outputs[k, 1]
                        u[i+1, j+2] = u_outputs[k, 5]
                        u[i+2, j+2] = u_outputs[k, 0]
                    
                    if i == dist:
                        u[i-1, j] = u_outputs[k, 17]
                        u[i-2, j] = u_outputs[k, 22]
                    elif j == dist:
                        u[i, j-1] = u_outputs[k, 13]
                        u[i, j-2] = u_outputs[k, 14]
                    elif i == Nx-dist:
                        u[i+1, j] = u_outputs[k, 7]
                        u[i+2, j] = u_outputs[k, 2]
                    elif j == Ny-dist:
                        u[i, j+1] = u_outputs[k, 11]
                        u[i, j+2] = u_outputs[k, 10]
                    u[i, j] = u_outputs[k, 12]
                    k += 1
            errors.append(dde.metrics.l2_relative_error(u_new,u))
            count += 1
            if count % 1000 == 0:
                print(count,errors[-1])
            min_err = min(errors[-1],min_err)
    elif outputs.shape[1] == 49:
        print("outputs.shape[1]=", outputs.shape[1])
        ts = time.time()
        count = 0
        min_err = errors[-1]
        while errors[-1] > 1e-4 and count < 15000:
            inputs,_ = construct_more_data(Nx, Ny, f_new, u)
            u_outputs = model.predict(inputs)
            # print(u_outputs.shape, u.shape)
            dist = int(abs(np.sqrt(inputs.shape[0]) - np.sqrt(Nx*Ny))/2)
            k = 0
            for i in range(dist, Nx - dist):
                for j in range(dist, Ny - dist):
                    if i == dist and j == dist:
                        u[i-1, j-1] = u_outputs[k, 32]
                        u[i-2, j-1] = u_outputs[k, 39]
                        u[i-1, j-2] = u_outputs[k, 33]
                        u[i-2, j-2] = u_outputs[k, 40]
                        u[i-3, j-1] = u_outputs[k, 46]
                        u[i-1, j-3] = u_outputs[k, 34]
                        u[i-3, j-2] = u_outputs[k, 47]
                        u[i-2, j-3] = u_outputs[k, 41]
                        u[i-3, j-3] = u_outputs[k, 48]
                    elif i == dist and j == Ny - dist:
                        u[i-1, j+1] = u_outputs[k, 30]
                        u[i-2, j+1] = u_outputs[k, 37]
                        u[i-1, j+2] = u_outputs[k, 29]
                        u[i-2, j+2] = u_outputs[k, 36]
                        u[i-3, j+1] = u_outputs[k, 44]
                        u[i-1, j+3] = u_outputs[k, 28]
                        u[i-3, j+2] = u_outputs[k, 43]
                        u[i-2, j+3] = u_outputs[k, 35]
                        u[i-3, j+3] = u_outputs[k, 42]
                    elif i == Nx - dist and j == dist:
                        u[i+1, j-1] = u_outputs[k, 18]
                        u[i+2, j-1] = u_outputs[k, 11]
                        u[i+1, j-2] = u_outputs[k, 19]
                        u[i+2, j-2] = u_outputs[k, 12]
                        u[i+3, j-1] = u_outputs[k, 4]
                        u[i+1, j-3] = u_outputs[k, 20]
                        u[i+3, j-2] = u_outputs[k, 5]
                        u[i+2, j-3] = u_outputs[k, 13]
                        u[i+3, j-3] = u_outputs[k, 6]
                    elif i == Nx - dist and j == Ny - dist:
                        u[i+1, j+1] = u_outputs[k, 16]
                        u[i+2, j+1] = u_outputs[k, 9]
                        u[i+1, j+2] = u_outputs[k, 15]
                        u[i+2, j+2] = u_outputs[k, 8]
                        u[i+3, j+1] = u_outputs[k, 2]
                        u[i+1, j+3] = u_outputs[k, 14]
                        u[i+3, j+2] = u_outputs[k, 1]
                        u[i+2, j+3] = u_outputs[k, 7]
                        u[i+3, j+3] = u_outputs[k, 0]
                    
                    if i == dist:
                        u[i-1, j] = u_outputs[k, 31]
                        u[i-2, j] = u_outputs[k, 38]
                        u[i-3, j] = u_outputs[k, 45]
                    elif j == dist:
                        u[i, j-1] = u_outputs[k, 25]
                        u[i, j-2] = u_outputs[k, 26]
                        u[i, j-3] = u_outputs[k, 27]
                    elif i == Nx-dist:
                        u[i+1, j] = u_outputs[k, 17]
                        u[i+2, j] = u_outputs[k, 10]
                        u[i+3, j] = u_outputs[k, 3]
                    elif j == Ny-dist:
                        u[i, j+1] = u_outputs[k, 23]
                        u[i, j+2] = u_outputs[k, 22]
                        u[i, j+3] = u_outputs[k, 21]
                    u[i, j] = u_outputs[k, 24]
                    k += 1
            errors.append(dde.metrics.l2_relative_error(u_new,u))
            count += 1
            if count % 1000 == 0:
                print(count,errors[-1])
            min_err = min(errors[-1],min_err)
    elif outputs.shape[1] == 1:
        ts = time.time()
        count = 0
        min_err = errors[-1]
        while errors[-1] > 1e-4 and count < 15000:
            inputs,_ = construct_more_data(Nx, Ny, f_new, u)
            dist = int(abs(np.sqrt(inputs.shape[0]) - np.sqrt(Nx*Ny))/2)
            u_outputs = model.predict(inputs)
            k = 0
            for i in range(dist, Nx - dist):
                for j in range(dist, Ny - dist):
                    u[i, j] = u_outputs[k, 0]
                    k += 1
            errors.append(dde.metrics.l2_relative_error(u_new,u))
            count += 1
            if count % 1000 == 0:
                print(count,errors[-1])
            min_err = min(errors[-1],min_err)

    print("Error after {} iterations:".format(count), errors[-1])
    print("Minimum error is ", min_err)
    print("One-shot took {} s.".format(time.time()-ts))
    # save the results
    np.savetxt(f"{dname}/u_FPI_{size}.dat",u)
    np.savetxt(f"{dname}/err_FPI_{size}.dat",errors)

    if isplot:
        plt.figure()
        plt.rcParams.update({'font.size': 20,"savefig.dpi": 200, "figure.figsize": (8, 6)})
        plt.subplots_adjust(left=0.20, right=0.9, top=0.9, bottom=0.15)
        plt.imshow(np.rot90(abs(u - u_new)), cmap = "gnuplot", extent=(0,1,0,1), aspect='auto')
        plt.colorbar()
        plt.xlabel("x")
        plt.ylabel("t")
        plt.savefig(f"{dname}/res_FPI_{size}.png")
        #plt.show()
        plot_errs(errors, dname, size)
    return errors[-1]

def main(sigma, num_func, size, parent_dir = "../../data/", gen = False):
    M = 201
    Nx, Ny = 101, 101
    N_f = 101*101
    N_b = 0
    l, a = 0.01, 0.10
    l_new, a_new = 0.1, float(sigma)
    
    new_dir = "data_nonlinear_poisson"
    PATH = os.path.join(parent_dir, new_dir)
    
    if size == 5:
        best_step = "model.ckpt-101090" #"model_5-100213"
        pre_layers = [5, 64, 1] #[5, 64, 1]
    elif size == 9:
        best_step = "model.ckpt-101228" #"model_9-100195"
        pre_layers = [9, 64, 1]
    elif size == 13:
        # best_step = "model_13-100391"#"model_13-100084"
        # pre_layers = [13,64,1]
        best_step = "model_13-105158" #104961" # "model_13-104070"
        pre_layers = [13,64,5]
    elif size == 25:
        # best_step = "model_25-100162"#"model_25-100089"
        # pre_layers = [25, 64, 1]
        best_step = "model_25-102896"#                                                                                "model_25-100089"
        pre_layers = [25, 64, 9]
    elif size == 49:
        best_step = "model_49-101966"
        pre_layers = [49, 64, 25]
    elif size == 81:
        best_step = "model_81-103176" #3.84e-04
        pre_layers = [81, 64, 49]
        
    print(f"Size: {size}, Best step: {best_step}, Pre-layers: {pre_layers}")
        
    _, dataset = load_all_data(M, Nx, Ny, N_f, N_b, l, a, l_new, a_new, f"{PATH}/data_{sigma}/data_0", size, gen, 
                                      correction = False, grid = True, isplot = False)
    model = load_trained_model(dataset, pre_layers, best_step)
    
    errs = []
    for i in range(num_func):
        print("Dataset {}".format(i))
        dname = f"{PATH}/data_{sigma}/data_{i}"
        err = fixed_point_iteration(model, size, dname, isplot=True)
        errs.append(err)
    np.savetxt(os.path.join(f"{PATH}/data_{sigma}/", f"errs_FPI_{size}.dat"), errs)
    print("The average l2 error is ", sum(errs)/num_func)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=5) # size of local domains
    parser.add_argument("--num", type=int, default=5) # Number of functions
    parser.add_argument("--sigma", type=str, default="0.20") # Amplitude in the GRF
    args = parser.parse_args()
    
    # save to log file
    logging.info(f"Arguments received: size={args.size}, num={args.num}, sigma={args.sigma}")
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"{LOG_DIR}/log_FPI_{args.size}_{args.num}_{args.sigma}_{current_time}.log"
    logger = Logger(filename)
    sys.stdout = logger
    
    main(args.sigma, args.num, args.size)
    logger.close()