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

def pre_trained_NN(data, layers, ftype):
    net = dde.nn.FNN(layers, "tanh", "Glorot normal", use_bias=True, regularization=['l2', 1e-8])
    model = dde.Model(data, net)
    checker = dde.callbacks.ModelCheckpoint(f"model/model_{ftype}", save_better_only=True, period=100000)
    model.compile("adam",lr=1e-3,decay = ("inverse time", 20000, 0.5), metrics=["l2 relative error"])
    losshistory, train_state = model.train(epochs=150000,callbacks=[checker])
    model.compile("L-BFGS-B", metrics=["l2 relative error"])
    losshistory, train_state = model.train(callbacks=[checker], model_save_path = f"model/model_{ftype}")
    dde.utils.external.plot_loss_history(losshistory, fname = f"model/loss_{ftype}-{str(train_state.best_step)}.png")
    dde.utils.external.save_loss_history(losshistory, fname = f"model/loss_{ftype}-{str(train_state.best_step)}.dat")
    return model, str(train_state.best_step)

def main(sigma, num_func, ftype, i, parent_dir = "../../data/", gen = False):
    M = 1001
    Nx, Nt = 101, 101
    N_f = 101*101
    N_b = Nx*2 + Nt - 2
    l, a = 0.01, 0.1
    l_new, a_new = 0.1, float(sigma)

    # Create folders for the datasets
    new_dir = "data_nonlinear_diffusion"
    PATH = os.path.join(parent_dir, new_dir)
    os.makedirs(f"{PATH}/data_{ftype}", exist_ok = True)
    
    print("Dataset {}".format(i))
    dname = f"{PATH}/data_{ftype}/data_{i}"
    os.makedirs(dname, exist_ok = True)
    if ftype == "freq":
        gen = True
    if gen:
        if ftype == "G":
            gen_data_GRF(M, Nx, Nt, l, a, dname, True)
        elif ftype == "cos":
            gen_data_GRF_cos(M, Nx, Nt, l, a, dname, True)
        elif ftype == "sin":
            gen_data_GRF_sin(M, Nx, Nt, l, a, dname, True)
        elif ftype == "rf":
            a = 1.0
            print(l,a)
            gen_data_GRF_rf(M, Nx, Nt, l, a, dname, True)
        elif ftype == "freq":
            gen_data_GRF_freq(M, Nx, Nt, l, a, dname, True)
        else:
            print(f"ftype = {ftype} not implemented.")
    
    f_T = np.loadtxt(f"{dname}/f_T.dat")
    u_T = np.loadtxt(f"{dname}/u_T.dat")
    print(f"Loaded f_T {f_T.shape} and u_T {u_T.shape} for training the local solution operator.")
    d_T = construct_more_data(Nx, Nt, f_T, u_T)

    f_0 = np.loadtxt(f"{PATH}/data_0.50/data_0/f_0_grid.dat")
    u_0 = np.loadtxt(f"{PATH}/data_0.50/data_0/u_0_grid.dat")
    print(f"Loaded f_0_grid {f_0.shape} and u_0_grid {u_0.shape} for testing the local solution operator.")
    d_0 = construct_more_data(Nx, Nt, f_0, u_0)
    
    dataset_G = dde.data.DataSet(X_train=d_T[0], y_train=d_T[1], X_test=d_0[0], y_test=d_0[1])

    pre_layers = [4, 64, 1]
    print(pre_layers)
    model, best_step = pre_trained_NN(dataset_G, pre_layers, ftype)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num", type=int, default=3) # Number of functions
    parser.add_argument("--sigma", type=str, default="0.80") # Amplitude in the GRF
    parser.add_argument("--ftype", type=str, default="freq")
    parser.add_argument("--i", type=int, default=1)
    args = parser.parse_args()
    
    print(f"Arguments received: num={args.num}, sigma={args.sigma}, ftype={args.ftype}, i={args.i}")
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"{LOG_DIR}/log_{args.num}_{args.sigma}_{args.ftype}_{args.i}_{current_time}.log"
    logger = Logger(filename)
    sys.stdout = logger
    
    main(args.sigma, args.num, args.ftype, args.i)
    logger.close()
