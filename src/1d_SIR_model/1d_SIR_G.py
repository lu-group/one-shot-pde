# Train the local solution operator
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import random
import argparse
import numpy as np
import deepxde as dde
import deepxde.backend as bkd
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

def pre_trained_NN(data, layers): 
    def mean_squared_error(y_true, y_pred):
        return [1*bkd.reduce_mean(bkd.square(y_true[:, 0:3] - y_pred[:, 0:3])),
                1.3*bkd.reduce_mean(bkd.square(y_true[:, 3:6] - y_pred[:, 3:6]))]
        
    def mean_l2_relative_errorS(y_true, y_pred):
        y_true = y_true[:, 0:3]
        y_pred = y_pred[:, 0:3]
        """Compute the average of L2 relative error along the first axis."""
        return np.mean(
            np.linalg.norm(y_true - y_pred, axis=1) / np.linalg.norm(y_true, axis=1)
        )
    def mean_l2_relative_errorI(y_true, y_pred):
        y_true = y_true[:, 3:6]
        y_pred = y_pred[:, 3:6]
        """Compute the average of L2 relative error along the first axis."""
        return np.mean(
            np.linalg.norm(y_true - y_pred, axis=1) / np.linalg.norm(y_true, axis=1)
        )
    
    net = dde.nn.FNN(layers, "silu", "Glorot uniform")
    model = dde.Model(data, net)
    checker = dde.callbacks.ModelCheckpoint(f"model/model", save_better_only=True, period=100000)
    model.compile("adam",lr=1e-3,decay =  ("inverse time", 10000, 0.5), metrics=[mean_l2_relative_errorS, mean_l2_relative_errorI], loss=mean_squared_error)
    losshistory, train_state = model.train(epochs=20000,callbacks=[checker])
    model.compile("L-BFGS-B", metrics=[mean_l2_relative_errorS, mean_l2_relative_errorI], loss=mean_squared_error)
    losshistory, train_state = model.train(callbacks=[checker], model_save_path = f"model/model")
    # dde.utils.external.plot_loss_history(losshistory, fname = f"model/loss-{str(train_state.best_step)}.png")
    dde.utils.external.save_loss_history(losshistory, fname = f"model/loss-{str(train_state.best_step)}.dat")
    return model, str(train_state.best_step)

def main(sigmat, l, i, num_func, parent_dir = "../../data/", gen = 0):
    Mx, Mt = 401, 1001
    Nx, Nt = 101, 201
    N_f = 101*201
    N_b = 0
    l, a = l, float(sigmat)

    # Create folders for the datasets
    new_dir = "data_SIR"
    PATH = os.path.join(parent_dir, new_dir)
    os.makedirs(f"{PATH}/data_{sigmat}_{l}", exist_ok = True)

    print("Dataset {}".format(i))
    dname = f"{PATH}/data_{sigmat}_{l}/data_{i}"
    os.makedirs(dname, exist_ok = True)
    
    f_0 = np.loadtxt(f"{PATH}/data_0.10/data_0/f_0_grid.dat")[:,1]
    S_0 = np.loadtxt(f"{PATH}/data_0.10/data_0/S_0_grid.dat")
    I_0 = np.loadtxt(f"{PATH}/data_0.10/data_0/I_0_grid.dat")
    print(f"Loaded f_0_grid {f_0.shape} S_0_grid {S_0.shape}, I_0_grid {I_0.shape} for testing the local solution operator.")
    d_0 = construct_more_data(Nx, Nt, f_0, S_0, I_0)

    if gen == 1:
        gen_data_GRF(Mx, Mt, Nx, Nt, l, a, dname, True)

    f_T = np.loadtxt(f"{dname}/f_T.dat")[:,1]
    S_T = np.loadtxt(f"{dname}/S_T.dat")
    I_T = np.loadtxt(f"{dname}/I_T.dat")
    print(f"Loaded f_T {f_T.shape}, S_T {S_T.shape}, I_T {I_T.shape} for training the local solution operator.")
    d_T = construct_more_data(Nx, Nt, f_T, S_T, I_T)
    
    dataset_G = dde.data.DataSet(
        X_train=d_T[0],
        y_train=np.hstack((d_T[1], d_T[2])),
        X_test=d_0[0],
        y_test=np.hstack((d_0[1], d_0[2])),
    )
    print(np.mean(d_T[0]), np.mean(d_T[1]), np.mean(d_T[2]))
    print(np.mean(d_0[0]), np.mean(d_0[1]), np.mean(d_0[2]))

    pre_layers = [21, 128, 6] #[7,64,2]
    print(pre_layers)
    model, best_step = pre_trained_NN(dataset_G, pre_layers)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num", type=int, default=3) # Number of functions
    parser.add_argument("--sigmat", type=str, default="0.10") # Amplitude in the GRF
    parser.add_argument("--l", type=float, default=0.01) 
    parser.add_argument("--i", type=int, default=0) 
    parser.add_argument("--gen", type=int, default=0) 
    args = parser.parse_args()

    # save to log file
    print(f"Arguments received: num={args.num}, sigma={args.sigmat}, l={args.l}, gen = {args.gen}")
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"{LOG_DIR}/log_G_{args.num}_{args.sigmat}_{args.l}_{current_time}.log"
    logger = Logger(filename)
    sys.stdout = logger
    
    main(args.sigmat, args.l, args.i, args.num, gen = args.gen)
    logger.close()