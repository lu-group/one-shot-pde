# Train the local solution operator
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import random
import argparse
import deepxde as dde
from utils import load_all_data
dde.backend.set_default_backend('tensorflow.compat.v1')
dde.config.set_default_float("float64")
dde.config.disable_xla_jit()

def pre_trained_NN(data, layers):
    net = dde.nn.FNN(layers, "tanh", "LeCun normal", use_bias = True, regularization=['l2', 1e-8])
    model = dde.Model(data, net)
    checker = dde.callbacks.ModelCheckpoint("model/model.ckpt", save_better_only=True, period=1000)
    model.compile("adam", lr=1e-3, decay = ("inverse time", 20000, 0.5), metrics=["l2 relative error"])
    losshistory, train_state = model.train(epochs=200000,callbacks=[checker], model_save_path = "model/model.ckpt")
    model.compile("L-BFGS-B", metrics=["l2 relative error"])
    losshistory, train_state = model.train(callbacks=[checker], model_save_path = "model/model.ckpt")
    return model, str(train_state.best_step)

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
    
    i = random.randint(0, num_func)
    print("Dataset {}".format(i))
    dname = f"{PATH}/data_{sigma}/data_{i}"

    dataset_G, _ = load_all_data(M, Nx, Nt, N_f, N_b, l, a, l_new, a_new, dname, gen, 
                                        correction = False, grid = True, isplot = False)
    
    pre_layers = [7,64,2]
    model, best_step = pre_trained_NN(dataset_G, pre_layers)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num", type=int, default=1) # Number of functions
    parser.add_argument("--sigma", type=str, default="0.10") # Amplitude in the GRF
    args = parser.parse_args()
    main(args.sigma, args.num)

