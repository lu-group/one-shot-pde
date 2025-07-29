import os
os.environ["TF_XLA_FLAGS"] = '--tf_xla_cpu_global_jit'
os.chdir(os.path.dirname(os.path.abspath(__file__)))
from utils import load_all_data

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
    def __init__(self, Nx, Ny, dataset, net, pre_layers, best_step, x_train, l2_errs, dname, PATH):
        super(UpdateOutput, self).__init__()
        self.graph = tf.Graph()
        self.dataset = dataset
        self.l2_errs = l2_errs
        self.x_train = x_train
        self.Nx = Nx
        self.Ny = Ny
        self.hx = 1/Nx
        self.hy = 1/Ny
        self.net = net
        self.u_0_grid = np.loadtxt(f"{PATH}/data_0.05/data_0/data_0_grid.txt")[:,4].reshape((-1, 1))
        self.u_new_grid  = np.loadtxt(f"{dname}/data_new_grid.txt")[:,4].reshape((-1, 1))
        self.f_new = np.loadtxt(f"{dname}/data_new.txt")[:, 5].reshape((-1, 1))
        self.u_new = np.loadtxt(f"{dname}/data_new.txt")[:, 4].reshape((-1, 1))
        self.u_init = np.loadtxt(f"{PATH}/data_0.05/data_0/data_0.txt")[:, 4].reshape((-1, 1))
        self.f_0 = np.loadtxt(f"{PATH}/data_0.05/data_0/data_0.txt")[:, 5].reshape((-1, 1))
        self.net_outputs = net.outputs
        self.inputs = self.get_inputs()
        self.feed_dict = net.feed_dict(False, self.inputs)
        self.u_0 = np.loadtxt(f"{PATH}/data_0.05/data_0/u_0_inputs.txt", skiprows=9)[:, 4].reshape((-1, 1))
        self.l2_errs = l2_errs
        with self.graph.as_default():
            self.trained_model = self.load_trained_model(dataset, pre_layers, best_step)

    def on_batch_begin(self):
        outputs = self.model.sess.run(self.net_outputs, feed_dict = self.feed_dict)
        pred_grid = self.model.sess.run(self.net_outputs, feed_dict = self.net.feed_dict(False, self.model.data.test_x))
        outputs = outputs + self.u_0

        with self.graph.as_default():
            train_u,_ = self.construct_local_domain(self.f_new, outputs)
            pred_u = self.trained_model.predict(train_u)

        self.model.data.train_y = pred_u - self.u_init
        if self.model.train_state.epoch % 1000 == 0:
            err = np.linalg.norm(pred_grid + self.u_0_grid - self.u_new_grid)/np.linalg.norm(self.u_new_grid)
            self.l2_errs.append([self.model.train_state.epoch, err])
            print(self.model.train_state.epoch, "Prediction l2 relative error: ",err)

        if self.model.train_state.epoch % 1000 == 0:
            err = np.linalg.norm(outputs[4::5] - self.u_new)/np.linalg.norm(self.u_new)
            print(self.model.train_state.epoch, "l2 relative error: ",err)

    def get_u_0(self, PATH):
        data_0_grid = np.loadtxt(f"{PATH}/data_0.05/data_0/f_0_grid.txt", skiprows=9)
        data_0_grid[np.isnan(data_0_grid)] = 0
        x = np.linspace(0.005, 0.995, 100)
        y = np.linspace(0.005, 0.995, 100)
        xx, yy = np.meshgrid(x,y)
        u_0 = data_0_grid[:, 4].reshape((100, 100))
        interp = interpolate.RegularGridInterpolator((x, y), u_0, method='cubic', bounds_error = False, fill_value=0)
        u_0_new = interp((self.inputs[:,1], self.inputs[:, 0])).reshape((-1, 1))
        return u_0_new

    def get_inputs(self):
        # u[i-1, j], u[i, j-1], u[i+1, j], f[i, j]
        x_l = np.array([[[xt[0] - self.hx, xt[1]]] for xt in self.x_train])
        x_b = np.array([[[xt[0], xt[1] - self.hy]] for xt in self.x_train])
        x_r = np.array([[[xt[0] + self.hx, xt[1]]] for xt in self.x_train])
        x_t = np.array([[[xt[0], xt[1] + self.hy]] for xt in self.x_train])
        x = np.array([[[xt[0], xt[1]]] for xt in self.x_train])
        inputs = np.concatenate((x_l, x_b, x_r, x_t, x), axis = 1).reshape((-1, 2))
        return inputs


    def construct_local_domain(self, f, outputs):
        outputs_u = outputs[4::5]
        u_l = outputs[0::5]
        u_b = outputs[1::5]
        u_r = outputs[2::5]
        u_t = outputs[3::5]
        inputs = np.concatenate((u_l, u_b, u_r, u_t, f), axis = 1)
        return np.array(inputs), outputs_u

    def load_trained_model(self, data, layers, best_step):
        net = dde.nn.FNN(layers, "tanh", "LeCun normal", use_bias = True)#, regularization=['l2', 1e-8])
        model = dde.Model(data, net)
        # restore model
        model.compile("adam", lr=1e-3, decay = ("inverse time", 20000, 0.5), metrics=["l2 relative error"])
        model.compile("L-BFGS-B",  metrics=["l2 relative error"])
        model.restore("model/model.ckpt-" + best_step + ".ckpt", verbose=1)
        return model

def solve_nn(Nx, Nt, dataset_G, data, pre_layers, best_step, dname, PATH, isplot=False):
    os.makedirs(f"{dname}/history_cLOINN_r", exist_ok = True)
    x_train = data.train_x
    l2_errs = []
    net = dde.nn.FNN([2] + [128]*2 + [1], "relu", "Glorot normal")

    def output_transform(x, y):
        x0, y0 = x[:, 0:1], x[:, 1:2]
        return tf.math.tanh(y0) * tf.math.tanh(y0 - 1)  * tf.math.tanh(x0) * tf.math.tanh(x0 - 1) * y * tf.math.tanh((x0 - 0.5)**2 + (y0 - 0.5)**2 - 0.2**2)

    net.apply_output_transform(output_transform)
    model = dde.Model(data, net)

    l2_errs = []
    iters = 150000
    checker = dde.callbacks.ModelCheckpoint("model/clmodel.ckpt", save_better_only=True, period=1000)
    model.compile("adam", lr=1e-3, decay = ("inverse time", 20000, 0.5), metrics=["l2 relative error"])
    update = UpdateOutput(Nx, Nt, dataset_G, net, pre_layers, best_step, x_train, l2_errs, dname, PATH)
    losshistory, train_state = model.train(iterations=iters,  callbacks=[update, checker], model_save_path = "model/clmodel.ckpt")
    dde.saveplot(losshistory, train_state, issave=False, isplot=isplot, output_dir = f"{dname}/history_cLOINN_r")
    model.restore("model/clmodel.ckpt-" + "{}.ckpt".format(iters), verbose=1)

    #predict
    x_test = data.test_x
    u_0 = np.loadtxt(f"{PATH}/data_0.05/data_0/data_0_grid.txt")[:, 4].reshape((-1, 1))
    u_pred = model.predict(x_test).reshape((-1, 1)) + u_0
    u_true = np.loadtxt(f"{dname}/data_new_grid.txt")[:, 4].reshape((-1, 1))
    np.savetxt(f"{dname}/u_cLOINN_r.dat",u_pred)
    err = np.linalg.norm(u_pred - u_true)/np.linalg.norm(u_true)
    print("l2 relative error: ",err)

    l2_errs.append([iters,  err])
    l2_errs = np.array(l2_errs).reshape((-1,2))
    np.savetxt(f"{dname}/err_cLOINN_r.dat",l2_errs)

    if isplot:
        fig = plt.figure()
        plt.rcParams.update({'font.size': 25})
        plt.plot(l2_errs[:,0], l2_errs[:, 1])
        plt.xlabel("# Epochs")
        plt.ylabel("$L^2$ relative error")
        plt.show()
    return err, train_state.best_step

def main(sigma, num_func, parent_dir = "../../data/", gen = False):
    errs = []
    b_steps = [[0] for i in range(100)]
    Nx = Ny = 100
    l, a = 0.05, 0.10
    l_new, a_new = 0.1, float(sigma)

    # Create folders for the datasets
    new_dir = "data_nonlinear_poisson_circle"
    PATH = os.path.join(parent_dir, new_dir)

    # Load model
    pre_layers = [5, 64, 1]
    best_step =  "150994"

    errs = []
    b_steps = [[0] for i in range(num_func)]

    for i in range(num_func):
        print("Dataset {}".format(i))
        dname = f"{PATH}/data_{sigma}/data_{i}"

        dataset_G, dataset = load_all_data(Nx, Ny, dname, PATH, correction=True, isplot=False, issave=False)
        #err,b_step = apply(solve_nn, (Nx, Ny, dataset_G, dataset, pre_layers, best_step, dname, True))
        err,b_step = solve_nn(Nx, Ny, dataset_G, dataset, pre_layers, best_step, dname, PATH, True)
        ts = time.time()
        print("cLOINN took {} s.".format(time.time()-ts))
        errs.append(err)
        b_steps[i][0] = b_step
        np.savetxt(os.path.join(f"{dname}", f"b_steps_cLOINN_r.dat"), b_steps)

    print(errs)
    np.savetxt(os.path.join(f"{dname}", f"errs_cLOINN_r.dat"), errs)
    print("The average l2 error is ", sum(errs)/num_func)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num", type=int, default=3) # Number of functions
    parser.add_argument("--sigma", type=str, default="0.05") # Amplitude in the GRF
    args = parser.parse_args()
    main(args.sigma, args.num)