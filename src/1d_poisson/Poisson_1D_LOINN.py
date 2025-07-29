import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import argparse
import deepxde as dde
from deepxde.callbacks import Callback
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool
dde.config.disable_xla_jit()
dde.config.set_default_float("float64")
from utils import load_all_data

def apply(func, args=None, kwds=None):
    with Pool(1) as p:
        if args is None and kwds is None:
            r = p.apply(func)
        elif kwds is None:
            r = p.apply(func, args=args)
        elif args is None:
            r = p.apply(func, kwds=kwds)
        else:
            r = p.apply(func, args=args, kwds=kwds)
    return r

class UpdateOutput(Callback):
    def __init__(self, N, N_b, dataset, net, pre_layers, best_step, x_train, dname, grid):
        super(UpdateOutput, self).__init__()
        self.graph = tf.Graph()
        self.dataset = dataset
        self.Nx = len(x_train)
        self.h = 1/(N-1)
        self.N_b = N_b
        self.x = x_train[:]
        self.x_train = x_train
        if grid:
            f_new = np.loadtxt(f"{dname}/f_new_grid.dat")[:, 1]
        else:
            f_new = np.loadtxt(f"{dname}/f_new.dat")[:, 1]
        self.f_new = f_new.reshape((-1, 1))
        self.net_outputs = net.outputs
        self.feed_dict = net.feed_dict(False, self.get_inputs().reshape((-1, 1)))
        with self.graph.as_default():
            self.trained_model = self.load_trained_model(dataset, pre_layers, best_step)

    def on_batch_begin(self):
        outputs = self.model.sess.run(self.net_outputs, feed_dict = self.feed_dict)
        outputs_f = outputs
        with self.graph.as_default():
            train_u, _ = self.construct_local_domain(self.f_new, outputs_f)
            pred_u = self.trained_model.predict(train_u).reshape((-1,1))        
        self.model.data.train_y = pred_u

    def get_inputs(self):
        x_l = np.array([[xi - self.h] for xi in self.x]).reshape((-1,1))
        x_r = np.array([[xi + self.h] for xi in self.x]).reshape((-1,1))
        inputs = np.ravel(np.concatenate((x_l, self.x, x_r), axis = 1))
        return inputs

    def construct_local_domain(self, f, outputs):
        outputs_u = outputs[1::3]
        u_l = outputs[0::3]
        u_r = outputs[2::3]
        inputs = np.concatenate((u_l, u_r, f), axis = 1)
        return np.array(inputs), outputs_u
    
    def load_trained_model(self, data, layers, best_step):
        net = dde.nn.FNN(layers, "tanh", "LeCun normal", use_bias = False)
        model = dde.Model(data, net)
        model.compile("adam", lr=1e-3, metrics=["l2 relative error"])
        model.compile("L-BFGS-B",  metrics=["l2 relative error"])
        model.restore("model/model.ckpt-" + best_step + ".ckpt", verbose=1)
        return model


def solve_nn(N, N_b, dataset_G, data, pre_layers, best_step, dname, grid, isplot=False):
    sname = "_g" if grid else "_r"
    os.makedirs(f"{dname}/history_LOINN{sname}", exist_ok = True)
    x_train = data.train_x
    net = dde.nn.FNN([1] + [128]*2 + [1], "tanh", "LeCun normal")   #, regularization=['l2', 1e-8]
    def output_transform(x, y):
        return x * (x - 1) * y 
    net.apply_output_transform(output_transform)
    model = dde.Model(data, net)

    checker = dde.callbacks.ModelCheckpoint("model/lmodel.ckpt", save_better_only=True, period=1000)
    model.compile("adam", lr=0.001, decay = ("inverse time", 20000, 0.5), metrics=["l2 relative error"])
    update = UpdateOutput(N, N_b, dataset_G, net, pre_layers, best_step, x_train, dname, grid)
    losshistory, train_state = model.train(epochs=2000)
    losshistory, train_state = model.train(epochs=1000,  callbacks=[update])
    losshistory, train_state = model.train(epochs=17000, disregard_previous_best=True,  callbacks=[update, checker], model_save_path = "model/lmodel.ckpt")
    dde.saveplot(losshistory, train_state, issave=False, isplot=isplot, output_dir = f"{dname}/history_LOINN{sname}")
    model.restore("model/lmodel.ckpt-" + str(train_state.best_step)+".ckpt", verbose=1)

    x_test = data.test_x
    u_new = data.test_y
    u_pred = model.predict(x_test)
    np.savetxt(f"{dname}/u_LOINN{sname}.dat",u_pred)
    err = dde.metrics.l2_relative_error(u_pred, u_new)
    print("l2 relative error: ", err)
    return err, train_state.best_step

def main(sigma, num_func, grid, parent_dir = "../../data/", gen = False):
    M = 1001 # Number of points 
    N = 101
    N_f = 101
    N_b = 0
    l, a = 0.01, 0.5
    l_new, a_new = 0.1, float(sigma)
    sname = "_g" if grid else "_r"

    # Create folders for the datasets
    new_dir = "data_1d_poisson"
    PATH = os.path.join(parent_dir, new_dir)

    # Load model
    best_step = "207412"
    pre_layers = [3, 64, 1]

    errs = []
    b_steps = [[0] for i in range(num_func)]
    for i in range(num_func):
        print("Dataset {}".format(i))
        dname = f"{PATH}/data_{sigma}/data_{i}"
        dataset_G, dataset = load_all_data(M, N, N_f, N_b, l, a, l_new, a_new, 
                                           dname, gen, correction = False, grid = grid, isplot = False)
        err,b_step = apply(solve_nn, (N, N_b, dataset_G, dataset, pre_layers, best_step, dname, grid, False))
        errs.append(err)
        b_steps[i][0] = b_step
        np.savetxt(os.path.join(f"{dname}", f"b_steps_LOINN{sname}.dat"), b_steps)

    print(errs)
    np.savetxt(os.path.join(f"{dname}", f"errs_LOINN{sname}.dat"), errs)
    print("The average l2 error is ", sum(errs)/num_func)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid", action="store_true") # If use grid data, add --grid
    parser.add_argument("--num", type=int, default=3) # Number of functions
    parser.add_argument("--sigma", type=str, default="0.02") # Amplitude in the GRF
    args = parser.parse_args()
    print(args)
    main(args.sigma, args.num, args.grid)

