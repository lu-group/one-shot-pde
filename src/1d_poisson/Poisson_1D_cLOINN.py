import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
from utils import load_all_data, construct_data

import time
import argparse
import deepxde as dde
from deepxde.callbacks import Callback
import tensorflow as tf
import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt
from multiprocessing import Pool
dde.config.disable_xla_jit()
dde.config.set_default_float("float64")

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
    def __init__(self, N, N_b, data_G, data, net, pre_layers, best_step, x_train, l2_errs, dname, grid):
        super(UpdateOutput, self).__init__()
        self.graph = tf.Graph()
        self.data_G = data_G
        self.Nx = len(x_train)
        self.h = 1/(N-1)
        self.N_b = N_b
        self.x = x_train[:] 
        self.x_train = x_train 
        self.x_grid = data.test_x
        self.net_outputs = net.outputs
        self.l2_errs = l2_errs
        self.grid = grid
        self.construct_locals = construct_data if grid else self.construct_local_domain

        self.u_init = np.loadtxt(f"{dname}/u_init.dat")[:, 1].reshape((-1,1))
        self.u_0_grid = np.loadtxt(f"{dname}/u_0_grid.dat")[:, 1].reshape((-1, 1))
        self.u_new_grid = np.loadtxt(f"{dname}/u_new_grid.dat")[:, 1].reshape((-1, 1))
        if grid:
            self.u_0 = self.u_0_grid
            self.u_new = self.u_new_grid
            f_new = np.loadtxt(f"{dname}/f_new_grid.dat")[:, 1].flatten()
            self.feed_dict = net.feed_dict(False, self.x_train)
        else:
            self.inputs = self.get_inputs()
            self.u_0 = self.get_u_0(dname)
            self.u_new = np.loadtxt(f"{dname}/u_new.dat")[:, 1].reshape((-1,1))
            f_new = np.loadtxt(f"{dname}/f_new.dat")[:, 1].reshape((-1, 1))
            self.feed_dict = net.feed_dict(False, self.inputs.reshape((-1, 1)))
        self.f_new = f_new
        print("Initial error:", dde.metrics.l2_relative_error(self.u_new_grid, self.u_0_grid))
        
        with self.graph.as_default():
            self.trained_model = self.load_trained_model(data_G, pre_layers, best_step)

    def on_batch_begin(self):
        outputs = self.model.sess.run(self.net_outputs, feed_dict = self.feed_dict)
        outputs = outputs + self.u_0
        outputs = np.ravel(outputs) if self.grid else outputs

        with self.graph.as_default():
            train_u, _ = self.construct_locals(self.f_new, outputs)
            pred_u = self.trained_model.predict(train_u).reshape((-1,1))
        
        self.model.data.train_y = np.concatenate(([[0]], pred_u - self.u_0[1:-1,:], [[0]])) if self.grid else pred_u - self.u_init
        
        if self.model.train_state.epoch % 1000 == 0:
            err = np.linalg.norm(self.model.predict(self.x_grid) + self.u_0_grid - self.u_new_grid)/np.linalg.norm(self.u_new_grid)
            self.l2_errs.append([self.model.train_state.epoch, err])
            print(self.model.train_state.epoch, "Prediction l2 relative error: ",err)
        
        if self.model.train_state.epoch % 1000 == 0:
            pred_new = outputs.reshape((-1, 1)) if self.grid else outputs[1::3]
            err = np.linalg.norm(pred_new - self.u_new)/np.linalg.norm(self.u_new)
            print(self.model.train_state.epoch, "l2 relative error: ",err)

    def get_u_0(self, dname):
        u_0_data = np.loadtxt(f"{dname}/u_0.dat")
        interp = interpolate.interp1d(u_0_data[:, 0], u_0_data[:, 1], kind = "cubic")
        return interp(self.inputs).reshape((-1,1))

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
    os.makedirs(f"{dname}/history_cLOINN{sname}", exist_ok = True)
    x_train = data.train_x
    net = dde.nn.FNN([1] + [128]*2 + [1], "relu","LeCun normal")  #,regularization=['l2', 1e-8])

    def output_transform(x, y):
        return x * (x - 1) * y
    
    net.apply_output_transform(output_transform)
    model = dde.Model(data, net)
    
    l2_errs = []
    iters = 20000
    checker = dde.callbacks.ModelCheckpoint("model/clmodel.ckpt", save_better_only=True, period=1000)
    model.compile("adam", lr=0.001, decay = ("inverse time", 20000, 0.5), metrics=["l2 relative error"])
    update = UpdateOutput(N, N_b, dataset_G, data, net, pre_layers, best_step, x_train, l2_errs, dname, grid)
    losshistory, train_state = model.train(iterations=iters,  callbacks=[update, checker], model_save_path = "model/clmodel.ckpt")
    dde.saveplot(losshistory, train_state, issave=False, isplot=isplot, output_dir = f"{dname}/history_cLOINN{sname}")
    model.restore("model/clmodel.ckpt-" + "{}.ckpt".format(iters), verbose=1)

    x_test = data.test_x
    u_0 = np.loadtxt(f"{dname}/u_0_grid.dat")[:, 1].reshape((-1, 1))
    u_pred = model.predict(x_test).reshape((-1, 1)) + u_0
    np.savetxt(f"{dname}/u_cLOINN{sname}.dat",u_pred)
    u_new = np.loadtxt(f"{dname}/u_new_grid.dat")[:, 1].reshape((-1, 1))
    err = np.linalg.norm(u_pred - u_new)/np.linalg.norm(u_new)
    print("l2 relative error: ",err)
    
    if isplot:
        fig = plt.figure()
        plt.rcParams.update({'font.size': 25})
        plt.plot(x_test, u_new, "k", label="Ref")
        plt.plot(x_test, u_pred, "--r", label="NN")
        plt.xlabel("x")
        plt.ylabel("u")
        plt.legend()
        #plt.savefig("data{}/u_FPC.png".format(d_num))
        plt.show()

    l2_errs.append([iters,  err])
    l2_errs = np.array(l2_errs).reshape((-1,2))
    np.savetxt(f"{dname}/err_cLOINN{sname}.dat",l2_errs)
    if isplot:
        fig = plt.figure()
        plt.rcParams.update({'font.size': 25})
        plt.plot(l2_errs[:,0], l2_errs[:, 1])
        plt.xlabel("# Epochs")
        plt.ylabel("$L^2$ relative error")
        plt.show()
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
                                           dname, gen, correction = True, grid = grid, isplot = False)
        ts = time.time()
        # err,b_step = apply(solve_nn, (N, N_b, dataset_G, dataset, pre_layers, best_step, dname, grid, False))
        err,b_step = solve_nn(N, N_b, dataset_G, dataset, pre_layers, best_step, dname, grid, False)
        print("cLOINN took {} s.".format(time.time()-ts))
        errs.append(err)
        b_steps[i][0] = b_step
        np.savetxt(os.path.join(f"{dname}", f"b_steps_cLOINN{sname}.dat"), b_steps)

    print(errs)
    np.savetxt(os.path.join(f"{dname}", f"errs_cLOINN{sname}.dat"), errs)
    print("The average l2 error is ", sum(errs)/num_func)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid", action="store_true") # If use grid data, add --grid
    parser.add_argument("--num", type=int, default=3) # Number of functions
    parser.add_argument("--sigma", type=str, default="0.10") # Amplitude in the GRF
    args = parser.parse_args()
    print(args)
    main(args.sigma, args.num, args.grid)


