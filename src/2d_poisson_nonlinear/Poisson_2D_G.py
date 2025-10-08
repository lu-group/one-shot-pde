import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import argparse
import deepxde as dde
import matplotlib.pyplot as plt
from utils  import load_all_data
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
    # define the network and model
    net = dde.nn.FNN(layers, "tanh", "Glorot uniform", use_bias = True)#, regularization=['l2', 1e-9])
    model = dde.Model(data, net)
    checker = dde.callbacks.ModelCheckpoint(f"model/model_{layers[0]}.ckpt", save_better_only=True, period=100000)
    # train
    model.compile("adam", lr=1e-3, decay = ("inverse time", 20000, 0.5), metrics=["l2 relative error"])
    losshistory, train_state = model.train(epochs=10000, callbacks=[checker], model_save_path = f"model/model_{layers[0]}")
    model.compile("L-BFGS-B", metrics=["l2 relative error"])
    losshistory, train_state = model.train(callbacks=[checker], model_save_path = f"model/model_{layers[0]}")
    #dde.saveplot(losshistory, train_state, issave=True, isplot=True, output_dir = "model")
    dde.utils.external.plot_loss_history(losshistory, fname = f"model/loss_{layers[0]}-{str(train_state.best_step)}.png")
    dde.utils.external.save_loss_history(losshistory, fname = f"model/loss_{layers[0]}-{str(train_state.best_step)}.dat")
    # model.restore(f"model/model_{layers[0]}-" + str(train_state.best_step)+".ckpt", verbose=1)
    # print("Train MSE:", dde.metrics.mean_squared_error(data.train_y, model.predict(data.train_x)))
    # print("Test MSE:",dde.metrics.mean_squared_error(data.test_y, model.predict(data.test_x)))
    #model.print_model()
    return model, str(train_state.best_step)

def main(sigma, num_func, size, parent_dir = "../../data/", gen = False):
    M = 201
    Nx, Ny = 101, 101
    N_f = 101*101
    N_b = 0
    l, a = 0.01, 0.10
    l_new, a_new = 0.1, float(sigma)
    
    # Create folders for the datasets
    new_dir = "data_nonlinear_poisson"
    PATH = os.path.join(parent_dir, new_dir)

    i = 1
    print("Dataset {}".format(i))
    dname = f"{PATH}/data_{sigma}/data_{i}"

    dataset_G, dataset = load_all_data(M, Nx, Ny, N_f, N_b, l, a, l_new, a_new, dname, size, gen, 
                                      correction = False, grid = True, isplot = False)

    if size == 5:
        pre_layers = [5,64,1]
    elif size == 9:
        pre_layers = [9,64,1]
    elif size == 13:
        pre_layers = [13,64,5]
    elif size == 25:
        pre_layers = [25,64,9]
    elif size == 49: 
        pre_layers = [49,64,25]
    elif size == 81:
        pre_layers = [81,64,49]
    else:
        raise ValueError("Unsupported `size` parameter.")
        
    pre_trained_NN(dataset_G, pre_layers)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=5) # size of local domains
    parser.add_argument("--num", type=int, default=5) # Number of functions
    parser.add_argument("--sigma", type=str, default="0.20") # Amplitude in the GRF
    args = parser.parse_args()
    # save to log file
    logging.info(f"Arguments received: size={args.size}, num={args.num}, sigma={args.sigma}")
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"{LOG_DIR}/log_{args.size}_{args.num}_{args.sigma}_{current_time}.log"
    logger = Logger(filename)
    sys.stdout = logger
    
    main(args.sigma, args.num, args.size)
    logger.close()
