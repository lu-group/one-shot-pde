import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.environ["TF_XLA_FLAGS"] = '--tf_xla_cpu_global_jit'

import time
import argparse
import deepxde as dde
from deepxde.callbacks import Callback
import numpy as np
import tensorflow as tf
dde.config.set_default_float("float64")
dde.config.disable_xla_jit()
from utils import *

class UpdateOutput(Callback):
    def __init__(self, Nx, Nt, dataset, net, pre_layers, best_step, x_train, l2_errs, dname):
        super(UpdateOutput, self).__init__()
        # load the pre-trained model
        self.graph = tf.Graph()
        self.dataset = dataset
        self.l2_errs = l2_errs
        self.x_train = x_train
        self.Nx = Nx
        self.Nt = Nt
        self.f_new = np.loadtxt(f"{dname}/f_new_grid.dat")
        self.u_new = np.loadtxt(f"{dname}/u_new_grid.dat")
        self.u_0 = np.loadtxt(f"{dname}/u_0_grid.dat")
        self.net_outputs = net.outputs
        self.feed_dict = net.feed_dict(False, self.x_train)
        self.l2_errs = l2_errs
        with self.graph.as_default():
            self.trained_model = self.load_trained_model(dataset, pre_layers, best_step)

    def on_batch_begin(self):
        outputs = self.model.sess.run(self.net_outputs, feed_dict = self.feed_dict)
        outputs = outputs.reshape((self.Nx, self.Nt)) + self.u_0

        with self.graph.as_default():
            train_u,_ = construct_data(self.f_new, outputs)
            pred_u = self.trained_model.predict(train_u)

        pred_u = pred_u.reshape((self.Nx-2, self.Nt-1))
        pred_u = np.concatenate((np.zeros((self.Nx-2, 1)), pred_u), axis = 1)
        pred_u = np.concatenate((np.zeros((1, self.Nt)), pred_u, np.zeros((1, self.Nt))), axis = 0)

        self.model.data.train_y = pred_u.reshape((-1, 1)) - self.u_0.reshape((-1, 1))

        if self.model.train_state.epoch % 1000 == 0:
            err = np.linalg.norm(outputs - self.u_new)/np.linalg.norm(self.u_new)
            self.l2_errs.append([self.model.train_state.epoch, err])
            print(self.model.train_state.epoch, "l2 relative error: ",err)

    def load_trained_model(self, data, layers, best_step):
        net = dde.nn.FNN(layers, "tanh", "LeCun normal", use_bias = True, regularization=['l2', 1e-8])
        model = dde.Model(data, net)
        model.compile("adam", lr=1e-3, decay = ("inverse time", 20000, 0.5), metrics=["l2 relative error"])
        model.compile("L-BFGS-B",  metrics=["l2 relative error"])
        model.restore("model/model.ckpt-" + best_step + ".ckpt", verbose=1)
        return model


def solve_nn(Nx, Nt, dataset_G, data, pre_layers, best_step, dname, PATH, isplot=False):
    sname = "_g"
    os.makedirs(f"{dname}/history_cLOINN{sname}", exist_ok = True)
    x_train = data.train_x
    l2_errs = []
    net = dde.nn.FNN([2] + [128]*2 + [1], "tanh", "LeCun normal")
    model = dde.Model(data, net)

    iters = 100000
    checker = dde.callbacks.ModelCheckpoint("model/clmodel.ckpt", save_better_only=True, period=100000)
    model.compile("adam", lr=1e-3, decay = ("inverse time", 10000, 0.8), metrics=["l2 relative error"])
    update = UpdateOutput(Nx, Nt, dataset_G, net, pre_layers, best_step, x_train, l2_errs, dname)
    losshistory, train_state = model.train(epochs=iters,disregard_previous_best=True,  callbacks=[update, checker], model_save_path = "model/clmodel.ckpt")
    dde.saveplot(losshistory, train_state, issave=True, isplot = False, output_dir = f"{dname}/history_cLOINN{sname}")

    u_0 = np.loadtxt(f"{dname}/u_0_grid.dat").reshape((-1,1))
    u_true = np.loadtxt(f"{dname}/u_new_grid.dat").reshape((-1,1))

    u_pred = model.predict(data.test_x)+u_0
    u_pred = u_pred.reshape((-1,1))
    err = dde.metrics.l2_relative_error(u_pred, u_true)
    print("l2 relative error: ", err)
    np.savetxt(f"{dname}/u_cLOINN{sname}.dat",u_pred)

    l2_errs.append([iters,  err])
    l2_errs = np.array(l2_errs).reshape((-1,2))
    print(l2_errs)
    np.savetxt(f"{dname}/err_cLOINN{sname}.dat", l2_errs)
    if isplot:
        fig = plt.figure()
        plt.rcParams.update({'font.size': 20})
        plt.plot(l2_errs[:,0], l2_errs[:, 1])
        plt.xlabel("# Epochs")
        plt.ylabel("$L^2$ relative error")
        plt.savefig(f"{dname}/err_cLOINN{sname}.png")
        plt.show()
    return err, train_state.best_step

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

    best_step = "161943"
    pre_layers = [4, 64, 1]
    errs = []
    b_steps = [[0] for i in range(num_func)]
    for i in range(num_func):
        print("Dataset {}".format(i))
        dname = f"{PATH}/data_{sigma}/data_{i}"
        dataset_G, dataset = load_all_data(M, Nx, Nt, N_f, N_b, l, a, l_new, a_new, dname, gen, 
                                      correction = True, grid = True, isplot = False)
        #err,b_step = apply(solve_nn, (Nx, Ny, dataset_G, dataset, pre_layers, best_step, dname, PATH, True))
        err,b_step = solve_nn(Nx, Nt, dataset_G, dataset, pre_layers, best_step, dname, PATH)
        ts = time.time()
        print("cLOINN took {} s.".format(time.time()-ts))
        errs.append(err)
        b_steps[i][0] = b_step
        np.savetxt(os.path.join(f"{PATH}/data_{sigma}/", f"b_steps_cLOINN_g.dat"), b_steps)

    print(errs)
    np.savetxt(os.path.join(f"{PATH}/data_{sigma}/", f"errs_cLOINN_g.dat"), errs)
    print("The average l2 error is ", sum(errs)/num_func)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num", type=int, default=5) # Number of functions
    parser.add_argument("--sigma", type=str, default="0.50") # Amplitude in the GRF
    args = parser.parse_args()
    
    main(args.sigma, args.num)

