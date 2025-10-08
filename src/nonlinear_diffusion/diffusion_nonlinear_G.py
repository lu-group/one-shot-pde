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

def pre_trained_NN(data, layers):
    net = dde.nn.FNN(layers, "tanh", "LeCun normal", use_bias=True, regularization=['l2', 1e-8])
    model = dde.Model(data, net)
    checker = dde.callbacks.ModelCheckpoint(f"model/model", save_better_only=True, period=100000)
    model.compile("adam",lr=1e-3,decay = ("inverse time", 20000, 0.5), metrics=["l2 relative error"])
    losshistory, train_state = model.train(epochs=150000,callbacks=[checker])
    model.compile("L-BFGS-B", metrics=["l2 relative error"])
    losshistory, train_state = model.train(callbacks=[checker], model_save_path = f"model/model")
    return model, str(train_state.best_step)

def main(sigma, num_func, parent_dir = "../../data/", gen = False):
    M = 1001
    Nx, Nt = 101, 101
    N_f = 101*101
    N_b = Nx*2 + Nt - 2
    l, a = 0.01, 0.1
    l_new, a_new = 0.1, float(sigma)

    # Create folders for the datasets
    new_dir = "data_nonlinear_diffusion"
    PATH = os.path.join(parent_dir, new_dir)
    
    i = random.randint(0, num_func)
    print("Dataset {}".format(i))
    dname = f"{PATH}/data_{sigma}/data_{i}"
    
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
    model, best_step = pre_trained_NN(dataset_G, pre_layers)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num", type=int, default=1) # Number of functions
    parser.add_argument("--sigma", type=str, default="0.50") # Amplitude in the GRF
    args = parser.parse_args()

    main(args.sigma, args.num)

