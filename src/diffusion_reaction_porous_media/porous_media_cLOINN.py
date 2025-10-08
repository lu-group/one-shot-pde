import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
from utils import load_all_data, construct_data

import time
import argparse
import deepxde as dde
dde.backend.set_default_backend('tensorflow.compat.v1')
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
    def __init__(self, Nx, Nt, dataset, net, pre_layers, best_step, x_train, l2_errs_a, l2_errs_b, dname):
        super(UpdateOutput, self).__init__()
        # load the pre-trained model
        self.graph = tf.Graph()
        self.dataset = dataset
        self.x_train = x_train
        self.Nx = Nx
        self.Nt = Nt
        self.hx = 1/(Nx-1)
        self.ht = 1/(Nt-1)
        self.net = net
        self.d = np.loadtxt(f"{dname}/f_new.dat")
        self.Ca_0_grid = np.loadtxt(f"{dname}/Ca_0_grid.dat").reshape((-1, 1))
        self.Cb_0_grid = np.loadtxt(f"{dname}/Cb_0_grid.dat").reshape((-1, 1))
        self.Ca_new_grid = np.loadtxt(f"{dname}/Ca_new_grid.dat").reshape((-1, 1))
        self.Cb_new_grid = np.loadtxt(f"{dname}/Cb_new_grid.dat").reshape((-1, 1))
        self.f_new = np.loadtxt(f"{dname}/f_new.dat")[:, 2].reshape((-1, 1))
        self.Ca_new = np.loadtxt(f"{dname}/Ca_new.dat")[:, 2].reshape((-1, 1))
        self.Cb_new = np.loadtxt(f"{dname}/Cb_new.dat")[:, 2].reshape((-1, 1))
        self.Ca_init = np.loadtxt(f"{dname}/Ca_init.dat")[:, 2].reshape((-1, 1))
        self.Cb_init = np.loadtxt(f"{dname}/Cb_init.dat")[:, 2].reshape((-1, 1))
        self.net_outputs = net.outputs
        self.inputs = self.get_inputs()
        self.feed_dict = net.feed_dict(False, self.inputs)
        self.Ca_0, self.Cb_0 = self.get_u_0(dname)
        self.l2_errs_a = l2_errs_a
        self.l2_errs_b = l2_errs_b
        with self.graph.as_default():
            self.trained_model = self.load_trained_model(dataset, pre_layers, best_step)
    
    def on_batch_begin(self):
        outputs = self.model.sess.run(self.net_outputs, feed_dict = self.feed_dict)
        pred_grid = self.model.sess.run(self.net_outputs, feed_dict = self.net.feed_dict(False, self.model.data.test_x))
        outputs_a = outputs[:][ :1][0] + self.Ca_0
        outputs_b = outputs[:][ 1:][0] + self.Cb_0

        with self.graph.as_default():
            inputs,_, _ = self.construct_local_domain(self.f_new, outputs_a, outputs_b)
            pred_u = self.trained_model.predict(inputs)

        pred_Ca = pred_u[:, 0:1]
        pred_Cb = pred_u[:, 1:]
        self.model.data.train_y = np.hstack((pred_Ca - self.Ca_init, pred_Cb - self.Cb_init))

        if self.model.train_state.epoch % 1000 == 0:
            err_a = np.linalg.norm(pred_grid[:][0:1][0] + self.Ca_0_grid - self.Ca_new_grid)/np.linalg.norm(self.Ca_new_grid)
            err_b = np.linalg.norm(pred_grid[:][1:][0] + self.Cb_0_grid - self.Cb_new_grid)/np.linalg.norm(self.Cb_new_grid)
            self.l2_errs_a.append([self.model.train_state.epoch, err_a])
            self.l2_errs_b.append([self.model.train_state.epoch, err_b])
            print(self.model.train_state.epoch, "Prediction l2 relative error: ",err_a, " and ", err_b)

        if self.model.train_state.epoch % 1000 == 0:
            err_a = np.linalg.norm(outputs_a[0::4] - self.Ca_new)/np.linalg.norm(self.Ca_new)
            err_b = np.linalg.norm(outputs_b[0::4] - self.Cb_new)/np.linalg.norm(self.Cb_new)
            print(self.model.train_state.epoch, "l2 relative error: ",err_a, " and ", err_b)

    def get_u_0(self,d_num):
        x = np.linspace(0,1,1001)
        t = np.linspace(0,1,1001)
        Ca_0 = np.loadtxt(f"{d_num}/Ca_0.dat")
        Cb_0 = np.loadtxt(f"{d_num}/Cb_0.dat")
        interp_a = interpolate.RegularGridInterpolator((x, t), Ca_0, method='cubic', bounds_error=False, fill_value=0 )
        interp_b = interpolate.RegularGridInterpolator((x, t), Cb_0, method='cubic', bounds_error=False, fill_value=0 )
        Ca_0_new = np.array([interp_a((i[0], i[1])) for i in self.inputs]).reshape((-1, 1))
        Cb_0_new = np.array([interp_b((i[0], i[1])) for i in self.inputs]).reshape((-1, 1))
        return Ca_0_new, Cb_0_new

    def get_inputs(self):
        #f[i, j], u[i-1, j], u[i, j-1], u[i+1, j]
        x_l = np.array([[[xt[0] - self.hx, xt[1]]] for xt in self.x_train])
        x_b = np.array([[[xt[0], xt[1] - self.ht]] for xt in self.x_train])
        x_r = np.array([[[xt[0] + self.hx, xt[1]]] for xt in self.x_train])
        x = np.array([[[xt[0], xt[1]]] for xt in self.x_train])
        inputs = np.concatenate((x, x_l, x_b, x_r), axis = 1).reshape((-1, 2))
        return inputs

    def construct_local_domain(self, f, outputs1, outputs2):
        outputs_Ca = outputs1[0::4]
        outputs_Cb = outputs2[0::4]
        u1_l = outputs1[1::4]
        u1_b = outputs1[2::4]
        u1_r = outputs1[3::4]
        u2_l = outputs2[1::4]
        u2_b = outputs2[2::4]
        u2_r = outputs2[3::4]
        inputs = np.concatenate((f, u1_l, u1_b, u1_r, u2_l, u2_b, u2_r), axis = 1)
        return np.array(inputs), outputs_Ca, outputs_Cb

    def load_trained_model(self, data, layers, best_step):
        net = dde.nn.FNN(layers, "tanh", "LeCun normal", use_bias = True, regularization=['l2', 1e-8])
        model = dde.Model(data, net)
        model.compile("adam", lr=1e-3, decay = ("inverse time", 20000, 0.5), metrics=["l2 relative error"])
        model.compile("L-BFGS-B",  metrics=["l2 relative error"])
        model.restore("model/model.ckpt-" + best_step + ".ckpt", verbose=1)
        return model

def solve_nn(Nx, Nt, N_b, dataset_G, data, pre_layers, best_step, dname, isplot=False):
    sname = "_r"
    os.makedirs(f"{dname}/history_cLOINN{sname}", exist_ok = True)
    x_train = data.train_x
    l2_errs_a, l2_errs_b = [], []
    net = dde.nn.FNN([2] + [32]*2 + [2], "tanh", "LeCun normal")
  
    def output_transform(x, y):
        x0, t0 = x[:, 0:1], x[:, 1:2]
        Ca, Cb = y[:, 0:1], y[:, 1:2]
        return [Ca*tf.math.tanh(x0)*tf.math.tanh(x0-1)*tf.math.tanh(t0), Cb*tf.math.tanh(x0)*tf.math.tanh(x0-1)*tf.math.tanh(t0)]

    net.apply_output_transform(output_transform)
    model = dde.Model(data, net)

    iters = 30000
    checker = dde.callbacks.ModelCheckpoint("model/clmodel.ckpt", save_better_only=True, period=1000)
    model.compile("adam", lr=1e-3, decay = ("inverse time", 10000, 0.5), metrics=["l2 relative error"])
    update = UpdateOutput(Nx, Nt, dataset_G, net, pre_layers, best_step, x_train, l2_errs_a, l2_errs_b, dname)
    
    losshistory, train_state = model.train(iterations=iters, disregard_previous_best=True, callbacks=[update, checker], model_save_path = "model/clmodel.ckpt")
    dde.saveplot(losshistory, train_state, issave=False, isplot=isplot, output_dir = f"{dname}/history_cLOINN{sname}")
    model.restore("model/clmodel.ckpt-" + "{}.ckpt".format(iters), verbose=1)
    
    #predict
    Ca_0 = np.loadtxt(f"{dname}/Ca_0_grid.dat").reshape((-1,1))
    Cb_0 = np.loadtxt(f"{dname}/Cb_0_grid.dat").reshape((-1,1))
    Ca_true = np.loadtxt(f"{dname}/Ca_new_grid.dat").reshape((-1,1))
    Cb_true = np.loadtxt(f"{dname}/Cb_new_grid.dat").reshape((-1,1))
    u_pred = model.predict(data.test_x)
    Ca_pred = u_pred[:][ :1] + Ca_0
    Cb_pred = u_pred[:][1:] + Cb_0
    Ca_pred, Cb_pred = Ca_pred.reshape((-1,1)), Cb_pred.reshape((-1,1))
    err_a = dde.metrics.l2_relative_error(Ca_pred, Ca_true)
    err_b = dde.metrics.l2_relative_error(Cb_pred, Cb_true)
    print("l2 relative error: ", err_a, " and ", err_b)
    np.savetxt(f"{dname}/Ca_FPC.dat", Ca_pred)
    np.savetxt(f"{dname}/Cb_FPC.dat", Cb_pred)

    l2_errs_a.append([iters,  err_a])
    l2_errs_b.append([iters,  err_b])
    l2_errs_a = np.array(l2_errs_a).reshape((-1,2))
    l2_errs_b = np.array(l2_errs_b).reshape((-1,2))
    np.savetxt(f"{dname}/err_FPC_a.dat", l2_errs_a)
    np.savetxt(f"{dname}/err_FPC_b.dat", l2_errs_b)
    # fig2 = plt.figure()
    # plt.rcParams.update({'font.size': 20})
    # plt.plot(l2_errs[:,0], l2_errs[:, 1])
    # plt.xlabel("# Epochs")
    # plt.ylabel("$L^2$ relative error")
    #plt.savefig("data{}/err_FPC.png".format(d_num))
    return err_a, err_b, train_state.best_step


def main(sigma, num_func, parent_dir = "../../data/", gen = False):
    M = 1001 # Number of points 
    Nx, Nt = 101, 101
    N_f = 101*101
    N_b = 0
    l, a = 0.01, 0.1
    l_new, a_new = 0.1, float(sigma)

    # Create folders for the datasets
    new_dir = "data_porous_media"
    PATH = os.path.join(parent_dir, new_dir)
    
    # Load model
    best_step = "117616"
    pre_layers = [7, 64, 2]
    
    errs_a, errs_b = [], []
    b_steps = [[0] for i in range(num_func)]
    for i in range(num_func):
        print("Dataset {}".format(i))
        dname = f"{PATH}/data_{sigma}/data_{i}"
        dataset_G, dataset = load_all_data(M, Nx, Nt, N_f, N_b, l, a, l_new, a_new, 
                                 f"{PATH}/data_{sigma}/data_0", gen, 
                                 correction = True, grid = False, isplot = False)
        ts = time.time()
        err_a, err_b, b_step = solve_nn(Nx, Nt, N_b, dataset_G, dataset, pre_layers, best_step, dname, True)
        print("cLOINN took {} s.".format(time.time()-ts))
        errs_a.append(err_a)
        errs_b.append(err_b)
        b_steps[i][0] = b_step
        print(b_steps)
        np.savetxt(os.path.join(f"{dname}", f"b_steps_cLOINN_r.dat"), b_steps)

    print(b_steps)
    np.savetxt(os.path.join(f"{dname}", f"errs_a_cLOINN_r.dat"), errs_a)
    np.savetxt(os.path.join(f"{dname}", f"errs_b_cLOINN_r.dat"), errs_b)
    print("The average l2 errors are ", sum(errs_a)/num_func, " ", sum(errs_b)/num_func)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num", type=int, default=3) # Number of functions
    parser.add_argument("--sigma", type=str, default="0.10") # Amplitude in the GRF
    args = parser.parse_args()
    print(args)
    main(args.sigma, args.num)


