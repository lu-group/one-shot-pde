import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
#os.environ["TF_XLA_FLAGS"] = '--tf_xla_cpu_global_jit'

import time
import argparse
import deepxde as dde
from deepxde.callbacks import Callback
import numpy as np
from scipy import interpolate
import tensorflow as tf
dde.config.set_default_float("float64")
dde.config.disable_xla_jit()
from utils import *

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


class UpdateOutput(Callback):
    def __init__(self, Nx, Nt, dataset, net, pre_layers, best_step, x_train, l2_errs, dname, PATH):
        super(UpdateOutput, self).__init__()
        # load the pre-trained model
        self.graph = tf.Graph()
        self.dataset = dataset
        self.l2_errs = l2_errs
        self.x_train = x_train
        self.Nx = Nx
        self.Nt = Nt
        self.hx = 1/(Nx-1)
        self.ht = 1/(Nt-1)
        self.net = net
        self.u_0_grid = np.loadtxt(f"{dname}/u_0_grid.dat").reshape((-1, 1))
        self.u_new_grid = np.loadtxt(f"{dname}/u_new_grid.dat").reshape((-1, 1))
        self.f_new = np.loadtxt(f"{dname}/f_new.dat")[:, 2].reshape((-1, 1))
        self.u_new = np.loadtxt(f"{dname}/u_new.dat")[:, 2].reshape((-1, 1))
        self.u_init = np.loadtxt(f"{dname}/u_init.dat")[:, 2].reshape((-1, 1))
        self.net_outputs = net.outputs
        self.inputs = self.get_inputs()
        #print(self.inputs.shape)
        self.feed_dict = net.feed_dict(False, self.inputs)
        self.u_0 = self.get_u_0(PATH)
        self.l2_errs = l2_errs
        with self.graph.as_default():
            self.trained_model = self.load_trained_model(dataset, pre_layers, best_step)

    def on_batch_begin(self):
        # get the outputs  
        outputs = self.model.sess.run(self.net_outputs, feed_dict = self.feed_dict)
        pred_grid = self.model.sess.run(self.net_outputs, feed_dict = self.net.feed_dict(False, self.model.data.test_x))
        outputs = outputs + self.u_0

        with self.graph.as_default():
            train_u,_ = self.construct_local_domain(self.f_new, outputs)  # (9900, 4)
            pred_u = self.trained_model.predict(train_u) #(9900, 1)

        #print(pred_u.shape, self.u_init.shape)
        self.model.data.train_y = pred_u - self.u_init
        
        
        if self.model.train_state.epoch % 1000 == 0:
            err = np.linalg.norm(pred_grid + self.u_0_grid - self.u_new_grid)/np.linalg.norm(self.u_new_grid)
            self.l2_errs.append([self.model.train_state.epoch, err])
            print(self.model.train_state.epoch, "Prediction l2 relative error: ",err)

        if self.model.train_state.epoch % 1000 == 0:
            err = np.linalg.norm(outputs[3::4] - self.u_new)/np.linalg.norm(self.u_new)
            #self.l2_errs.append([self.model.train_state.epoch, err])
            print(self.model.train_state.epoch, "l2 relative error: ",err)

    def get_u_0(self, PATH):
        x = np.linspace(0,1,1001)
        t = np.linspace(0,1,1001)
        u_0 = np.loadtxt(f"{PATH}/data_G/u_0.dat")
        interp = interpolate.RegularGridInterpolator((x, t), u_0, method='cubic', bounds_error=False, fill_value=0 )
        u_0_new = np.array([interp((i[0], i[1])) for i in self.inputs]).reshape((-1, 1))
        return u_0_new

    def get_inputs(self):
        # u[i-1, j], u[i, j-1], u[i+1, j], f[i, j]
        print(self.x_train.shape)
        x_l = np.array([[[xt[0] - self.hx, xt[1]]] for xt in self.x_train])
        x_b = np.array([[[xt[0], xt[1] - self.ht]] for xt in self.x_train])
        x_r = np.array([[[xt[0] + self.hx, xt[1]]] for xt in self.x_train])
        x = np.array([[[xt[0], xt[1]]] for xt in self.x_train])
        #print(x_l, x_b, x_r, x)
        inputs = np.concatenate((x_l, x_b, x_r, x), axis = 1).reshape((-1, 2))  #(300,)
        return inputs

    def construct_local_domain(self, f, outputs):
        #print(outputs.shape)
        outputs_u = outputs[3::4]# (103,1)
        u_l = outputs[0::4]
        u_b = outputs[1::4]
        u_r = outputs[2::4]
        #print(u_l, u_b, u_r)
        inputs = np.concatenate((u_l, u_b, u_r, f), axis = 1)
        return np.array(inputs), outputs_u

    def load_trained_model(self, data, layers, best_step):
        net = dde.nn.FNN(layers, "tanh", "LeCun normal", use_bias = True, regularization=['l2', 1e-8])
        model = dde.Model(data, net)
        # restore model
        model.compile("adam", lr=1e-3, decay = ("inverse time", 2000, 0.5), metrics=["l2 relative error"])
        model.compile("L-BFGS-B",  metrics=["l2 relative error"])
        model.restore("model/model.ckpt-" + best_step + ".ckpt", verbose=1)
        return model

def solve_nn(Nx, Nt, dataset_G, data, pre_layers, best_step, dname, PATH, isplot=False):
    os.makedirs(f"{dname}/history_cLOINN_r", exist_ok = True)
    # new source function
    x_train = data.train_x

    l2_errs = []
    # define the new model
    net = dde.nn.FNN([2] + [128]*2 + [1], "tanh", "LeCun normal")

    def output_transform(x, y):
        x0, t = x[:, 0:1], x[:, 1:2]
        return x0 * (x0 - 1) * t * y

    net.apply_output_transform(output_transform)
    model = dde.Model(data, net)

    iters = 100000
    checker = dde.callbacks.ModelCheckpoint("model/clmodel.ckpt", save_better_only=True, period=10000)
    model.compile("adam", lr=1e-3, decay = ("inverse time", iters // 10, 0.8), metrics=["l2 relative error"])
    update = UpdateOutput(Nx, Nt, dataset_G, net, pre_layers, best_step, x_train, l2_errs, dname, PATH)
    losshistory, train_state = model.train(epochs=iters,disregard_previous_best=True,  callbacks=[update, checker], model_save_path = f"model/clmodel.ckpt")
    dde.saveplot(losshistory, train_state, issave=True, isplot = False, output_dir = f"{dname}/history_cLOINN_r")
    # model.restore(f"model/clmodel.ckpt-{iters}.ckpt", verbose=1)

    #predict
    u_0 = np.loadtxt(f"{dname}/u_0_grid.dat").reshape((-1,1))
    u_true = np.loadtxt(f"{dname}/u_new_grid.dat").reshape((-1,1))
    u_pred = model.predict(data.test_x)+u_0
    u_pred = u_pred.reshape((-1,1))
    err = dde.metrics.l2_relative_error(u_pred, u_true)
    print("l2 relative error: ", err)
    print(u_pred.shape)
    np.savetxt(f"{dname}/u_cLOINN_r.dat",u_pred)

    l2_errs.append([iters,  err])
    l2_errs = np.array(l2_errs).reshape((-1,2))
    print(l2_errs)
    np.savetxt(f"{dname}/err_cLOINN_r.dat",l2_errs)
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
                                      correction = True, grid = False, isplot = False)
        #err,b_step = apply(solve_nn, (Nx, Ny, dataset_G, dataset, pre_layers, best_step, dname, PATH, True))
        err,b_step = solve_nn(Nx, Nt, dataset_G, dataset, pre_layers, best_step, dname, PATH)
        ts = time.time()
        print("cLOINN took {} s.".format(time.time()-ts))
        errs.append(err)
        b_steps[i][0] = b_step
        np.savetxt(os.path.join(f"{PATH}/data_{sigma}/", f"b_steps_cLOINN_r.dat"), b_steps)

    print(errs)
    np.savetxt(os.path.join(f"{PATH}/data_{sigma}/", f"errs_cLOINN_r.dat"), errs)
    print("The average l2 error is ", sum(errs)/num_func)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num", type=int, default=5) # Number of functions
    parser.add_argument("--sigma", type=str, default="0.50") # Amplitude in the GRF
    args = parser.parse_args()
    
    # save to log file
    logging.info(f"Arguments received: num={args.num}, sigma={args.sigma}")
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"{LOG_DIR}/log_cLOINN_{args.num}_{args.sigma}_{current_time}.log"
    logger = Logger(filename)
    sys.stdout = logger
    
    main(args.sigma, args.num)
    logger.close()