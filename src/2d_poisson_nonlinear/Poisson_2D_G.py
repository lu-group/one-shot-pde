import os
import deepxde as dde
import matplotlib.pyplot as plt
import tensorflow as tf
from utils import gen_all_data
dde.config.set_default_float("float64")
#dde.config.disable_xla_jit()

def pre_trained_NN(data, layers):
    net = dde.nn.FNN(layers, "tanh", "LeCun normal", use_bias = True, regularization=['l2', 1e-8])
    model = dde.Model(data, net)
    checker = dde.callbacks.ModelCheckpoint("model/model.ckpt", save_better_only=True, period=1000)
    model.compile("adam", lr=1e-3, decay = ("inverse time", 20000, 0.5), metrics=["l2 relative error"]) #decay = ("inverse time", 5000, 0.1)
    losshistory, train_state = model.train(epochs=100000,callbacks=[checker], model_save_path = "model/model.ckpt")
    model.compile("L-BFGS-B", metrics=["l2 relative error"])
    losshistory, train_state = model.train(callbacks=[checker], model_save_path = "model/model.ckpt")
    return model, str(train_state.best_step)

def main():
    M = 201
    Nx, Ny = 101, 101
    N_f = 101*101
    N_b = 0
    std = 0.05  # std for delta f
    std_num = "0.05"
    gen = False  # generate new data or not
    num = 100
    errs = []

    for i in range(num):
        print("Dataset {}".format(i))
        os.makedirs("data_{}/data_{}".format(std_num, i), exist_ok = True)
        d_num = "1_{}/data_{}".format(std_num, i)
        dataset_G, dataset = gen_all_data(M, Nx, Ny, N_f, N_b, 0.01, 0.1, 0.1, std, gen, d_num)
        pre_layers = [5,64,1]
        model, best_step = pre_trained_NN(dataset_G, pre_layers)

if __name__ == "__main__":
    main()
