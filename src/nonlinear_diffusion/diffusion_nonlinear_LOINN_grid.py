import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
from utils import *

import time
import argparse
import deepxde as dde
from deepxde.callbacks import Callback
import tensorflow as tf
import numpy as np
from multiprocessing import Pool
dde.config.disable_xla_jit()
dde.config.set_default_float("float64")

class UpdateOutput(Callback):
    def __init__(self, Nx, Nt, dataset, net, pre_layers, best_step, x_train, dname):
        super(UpdateOutput, self).__init__()
        self.graph = tf.Graph()
        self.dataset = dataset
        self.Nx = Nx
        self.Nt = Nt
        self.x_train = x_train
        self.f_new = np.loadtxt(f"{dname}/f_new_grid.dat")
        self.u_new = np.loadtxt(f"{dname}/u_new_grid.dat")
        self.net_outputs = net.outputs
        self.feed_dict = net.feed_dict(False, self.x_train)
        with self.graph.as_default():
            self.trained_model = self.load_trained_model(dataset, pre_layers, best_step)

    def on_batch_begin(self):
        outputs = self.model.sess.run(self.net_outputs, feed_dict = self.feed_dict)
        outputs = outputs.reshape((self.Nx, self.Nt))

        with self.graph.as_default():
            train_u,_ = construct_data(self.f_new, outputs)
            pred_u = self.trained_model.predict(train_u)

        pred_u = pred_u.reshape((self.Nx-2, self.Nt-1))
        pred_u = np.concatenate((np.zeros((self.Nx-2, 1)), pred_u), axis = 1)
        pred_u = np.concatenate((np.zeros((1,self.Nt)), pred_u, np.zeros((1, self.Nt))), axis = 0)
        self.model.data.train_y = pred_u.reshape((-1, 1))

    def load_trained_model(self, data, layers, best_step):
        net = dde.nn.FNN(layers, "tanh", "LeCun normal", use_bias = True)
        model = dde.Model(data, net)
        model.compile("adam", lr=1e-3, decay = ("inverse time", 20000, 0.5), metrics=["l2 relative error"])
        model.compile("L-BFGS-B",  metrics=["l2 relative error"])
        model.restore("model/model.ckpt-" + best_step + ".ckpt", verbose=1)
        return model

def solve_nn(Nx, Nt, dataset_G, data, pre_layers, best_step, dname, isplot=False):
    sname = "_g"
    os.makedirs(f"{dname}/history_LOINN{sname}", exist_ok = True)

    x_train = data.train_x

    # define the new model
    net = dde.nn.FNN([2] + [128]*2 + [1], "tanh", "LeCun normal")

    model = dde.Model(data, net)

    checker = dde.callbacks.ModelCheckpoint("model/lmodel.ckpt", save_better_only=True, period=100000)
    model.compile("adam", lr=1e-3, decay = ("inverse time", 10000, 0.8), metrics=["l2 relative error"])
    update = UpdateOutput(Nx, Nt, dataset_G, net, pre_layers, best_step, x_train, dname)
    losshistory, train_state = model.train(epochs=2000)
    losshistory, train_state = model.train(epochs=1000,  callbacks=[update])
    losshistory, train_state = model.train(epochs=97000, disregard_previous_best=True,  callbacks=[update, checker], model_save_path = "model/lmodel.ckpt")
    dde.saveplot(losshistory, train_state, issave=True, isplot=False, output_dir = f"{dname}/history_LOINN{sname}")
    # model.restore("model/lmodel.ckpt-" + str(train_state.best_step)+".ckpt", verbose=1)

    #predict
    u_pred = model.predict(data.test_x)
    u_new = data.test_y
    err = dde.metrics.l2_relative_error(u_pred, u_new)
    np.savetxt(f"{dname}/u_LOINN{sname}.dat",u_pred)
    print("l2 relative error: ", err)
    u_pred = u_pred.reshape((Nx, Nt))
    u_new = u_new.reshape((Nx, Nt))
    if isplot:
        plt.imshow(np.rot90(abs(u_pred - u_new)), cmap = "gnuplot", extent=(0,1,0,1), aspect='auto')
        plt.colorbar()
        plt.xlabel("x")
        plt.ylabel("t")
        # plt.savefig(f"{dname}/res_LOINN{sname}.png")
        plt.show()
    return err, train_state.best_step

def main(sigma, num_func, parent_dir = "../../data/", gen = False):
    M = 1001
    Nx, Nt = 101, 101
    N_f = 101*101
    N_b = Nx*2 + Nt - 2
    l, a = 0.01, 0.1
    l_new, a_new = 0.1, float(sigma)
    sname = "_g"

    # Create folders for the datasets
    new_dir = "data_nonlinear_diffusion"
    PATH = os.path.join(parent_dir, new_dir)
    
    best_step = "161943" 
    pre_layers = [4, 64, 1]
    errs = []
    b_steps = [[0] for i in range(num_func)]
    for i in range(num_func):
        print("Dataset {}".format(i))
        dname = f"{PATH}/data_{sigma}/data_{i}"
        dataset_G, dataset = load_all_data(M, Nx, Nt, N_f, N_b, l, a, l_new, a_new, 
                                 f"{PATH}/data_{sigma}/data_0", gen, 
                                 correction = False, grid = True, isplot = False)
        ts = time.time()
        err,b_step = solve_nn(Nx, Nt, dataset_G, dataset, pre_layers, best_step, dname, True)
        print("LOINN took {} s.".format(time.time()-ts))
        errs.append(err)
        b_steps[i][0] = b_step
        np.savetxt(os.path.join(f"{dname}", f"b_steps_LOINN{sname}.dat"), b_steps)

    print(errs)
    np.savetxt(os.path.join(f"{dname}", f"errs_LOINN{sname}.dat"), errs)
    print("The average l2 error is ", sum(errs)/num_func)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num", type=int, default=3) # Number of functions
    parser.add_argument("--sigma", type=str, default="0.50") # Amplitude in the GRF
    args = parser.parse_args()
    print(args)
    main(args.sigma, args.num)


