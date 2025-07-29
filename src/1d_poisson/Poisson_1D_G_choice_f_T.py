# Train the local solution operator
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import random
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

def pre_trained_NN(data, layers, l, a):
    net = dde.nn.FNN(layers, "tanh", "Glorot uniform", use_bias = False)
    model = dde.Model(data, net)
    checker = dde.callbacks.ModelCheckpoint(f"model/model_{l}_{a}", save_better_only=True, period=100000)
    model.compile("adam",lr=1e-3,decay =  ("inverse time", 20000, 0.5), metrics=["l2 relative error"])
    losshistory, train_state = model.train(epochs=200000,callbacks=[checker])
    model.compile("L-BFGS-B", metrics=["l2 relative error"])
    losshistory, train_state = model.train(callbacks=[checker], model_save_path = f"model/model_{l}_{a}")
    dde.utils.external.plot_loss_history(losshistory, fname = f"model/loss_{l}_{a}-{str(train_state.best_step)}.png")
    dde.utils.external.save_loss_history(losshistory, fname = f"model/loss_{l}_{a}-{str(train_state.best_step)}.dat")
    return model, str(train_state.best_step)

def main(sigmat, sigma, l, l_new, num_func, parent_dir = "../../data/", gen = True):
    M = 1001 # Number of points 
    N = 101
    N_f = 101
    N_b = 0
    l, a = l, float(sigmat)
    l_new, a_new = l_new, float(sigma)

    # Create folders for the datasets
    new_dir = "data_1d_poisson"
    PATH = os.path.join(parent_dir, new_dir)
    os.makedirs(f"{PATH}/data_{sigmat}_{l}", exist_ok = True)

    i = 3 #random.randint(0, num_func)
    print("Dataset {}".format(i))
    dname = f"{PATH}/data_{sigmat}_{l}/data_{i}"
    os.makedirs(dname, exist_ok = True)
    if gen:
        geom = dde.geometry.Interval(0, 1)
        gen_data_GRF(M, N, l, a, geom, dname, True)
        f_T = np.loadtxt(f"{dname}/f_T.dat")[:, 1]
        u_T = np.loadtxt(f"{dname}/u_T.dat")[:, 1]
        print(f"Loaded f_T {f_T.shape} and u_T {u_T.shape} for training the local solution operator.")
        d_T = construct_more_data(N, f_T, u_T)

        f_0 = np.loadtxt(f"{PATH}/data_0.02/data_0/f_0_grid.dat")[:, 1]
        u_0 = np.loadtxt(f"{PATH}/data_0.02/data_0/u_0_grid.dat")[:, 1]
        print(f"Loaded f_0_grid {f_0.shape} and u_0_grid {u_0.shape} for testing the local solution operator.")
        d_0 = construct_more_data(N, f_0, u_0)
        
        dataset_G = dde.data.DataSet(X_train=d_T[0], y_train=d_T[1], X_test=d_0[0], y_test=d_0[1])

    pre_layers = [3, 64, 1]
    model, best_step = pre_trained_NN(dataset_G, pre_layers, l, a)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num", type=int, default=10) # Number of functions
    parser.add_argument("--sigmat", type=str, default="0.02") # Amplitude in the GRF
    parser.add_argument("--sigma", type=str, default="0.02")
    parser.add_argument("--l", type=float, default=0.01) 
    parser.add_argument("--l_new", type=float, default=0.10)
    args = parser.parse_args()

    # save to log file
    print(f"Arguments received: num={args.num}, sigma={args.sigmat}, sigma={args.sigma}, l={args.l}, l_new={args.l_new}")
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"{LOG_DIR}/log_{args.num}_{args.sigmat}_{args.sigma}_{args.l}_{args.l_new}_{current_time}.log"
    logger = Logger(filename)
    sys.stdout = logger
    
    main(args.sigmat, args.sigma, args.l, args.l_new, args.num)
    logger.close()