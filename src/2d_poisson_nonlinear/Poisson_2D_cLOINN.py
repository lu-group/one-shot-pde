import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
#os.environ["TF_XLA_FLAGS"] = '--tf_xla_cpu_global_jit'

import time
import argparse
import deepxde as dde
from deepxde.callbacks import Callback
from multiprocessing import Pool
import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt
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
    def __init__(self, Nx, Ny, dataset, net, pre_layers, best_step, x_train, l2_errs, size, dname, PATH):
        super(UpdateOutput, self).__init__()
        # load the pre-trained model
        self.graph = tf.Graph()
        self.dataset = dataset
        self.l2_errs = l2_errs
        self.x_train = x_train
        self.hx = 1/(Nx-1)
        self.ht = 1/(Ny-1)
        self.net = net
        self.size = size
        
        self.d = np.loadtxt(f"{dname}/f_new.dat")
        self.x_grid = np.loadtxt(f"{dname}/x_grid.dat").reshape((-1, 1))
        self.y_grid = np.loadtxt(f"{dname}/y_grid.dat").reshape((-1, 1))
        
        self.u_0_grid = np.loadtxt(f"{PATH}/data_0.20/data_0/u_0_grid.dat").reshape((-1, 1))
        self.u_new_grid  = np.loadtxt(f"{dname}/u_new_grid.dat").reshape((-1, 1))
        self.f_new = np.loadtxt(f"{dname}/f_new.dat")[:, 2].reshape((-1, 1))
        self.u_new = np.loadtxt(f"{dname}/u_new.dat")[:, 2].reshape((-1, 1))
        self.u_init = np.loadtxt(f"{PATH}/data_0.20/data_0/u_init.dat")[:, 2].reshape((-1, 1))
        
        self.net_outputs = net.outputs
        self.inputs = self.get_inputs(size)
        if size > 10:
            self.f_new = self.f_new[~self.mask]
            self.u_new = self.u_new[~self.mask]
            self.x_train = self.x_train[~self.mask]
            self.interp_f = interpolate.griddata(self.x_train, self.f_new.flatten(), 
                                                 self.inputs, method='cubic', 
                                                 fill_value=0).reshape((-1,1))
        self.feed_dict = net.feed_dict(False, self.inputs)
        self.u_0 = self.get_u_0(PATH)
        self.l2_errs = l2_errs
        with self.graph.as_default():
            self.trained_model = self.load_trained_model(dataset, pre_layers, best_step)

    def on_batch_begin(self):
        outputs = self.model.sess.run(self.net_outputs, feed_dict = self.feed_dict)
        pred_grid = self.model.sess.run(self.net_outputs, feed_dict = self.net.feed_dict(False, self.model.data.test_x))
        
        outputs = outputs + self.u_0
        
        # if self.model.train_state.epoch % 1000 == 0:
        #    plt.figure()
        #    plt.scatter(self.inputs[:, 0],self. inputs[:, 1], c = outputs)
        #    plt.colorbar()
        #    plt.savefig("test.png")
        #    plt.show()

        #print(self.f_new.shape, outputs.shape)
        with self.graph.as_default():
            train_u,_ = self.construct_local_domain(self.f_new, outputs, self.size)
            pred_u = self.trained_model.predict(train_u)
            if self.size == 49:
                pred_u = pred_u[:,12:13]
            if self.size == 81:
                pred_u = pred_u[:,24:25]
            else:
                pred_u = pred_u[:,-1:]

        # if self.model.train_state.epoch % 1000 == 0:
        #     plt.figure()
        #     plt.scatter(self.d[:, 0][~self.mask],self. d[:, 1][~self.mask], c = pred_u - self.u_init[~self.mask])
        #     plt.colorbar()
        #     plt.savefig("test1.png")
        #     plt.show()
            
        #     plt.figure()
        #     plt.scatter(self.x_grid,self.y_grid, c = self.u_new_grid - self.u_0_grid)
        #     plt.colorbar()
        #     plt.savefig("test2.png")
        #     plt.show()
            
        #print(pred_u.shape, self.u_init.shape)
        if self.size > 10:
            self.model.data.train_y[~self.mask] = pred_u - self.u_init[~self.mask]
        else:
            self.model.data.train_y = pred_u - self.u_init

        if self.model.train_state.epoch % 1000 == 0:
            err = np.linalg.norm(pred_grid + self.u_0_grid - self.u_new_grid)/np.linalg.norm(self.u_new_grid)
            self.l2_errs.append([self.model.train_state.epoch, err])
            print(self.model.train_state.epoch, "Prediction l2 relative error: ",err)

        if self.model.train_state.epoch % 1000 == 0:
            if self.size == 5:
                outputs_pred = outputs[4::5]
            elif self.size == 9:
                outputs_pred = outputs[8::9]
            elif self.size == 13:
                outputs_pred = outputs[12::13]
            elif self.size == 25:
                outputs_pred = outputs[24::25]
            elif self.size == 49:
                outputs_pred = outputs[12::49]
            elif self.size == 81:
                outputs_pred = outputs[24::81]
            err = np.linalg.norm(outputs_pred - self.u_new)/np.linalg.norm(self.u_new)
            print(self.model.train_state.epoch, "l2 relative error: ",err)

    def get_u_0(self,PATH):
        x = np.linspace(0,1,201)
        t = np.linspace(0,1,201)
        u_0 = np.loadtxt(f"{PATH}/data_0.20/data_0/u_0.dat")
        interp = interpolate.RegularGridInterpolator((x, t), u_0, method='cubic', bounds_error=False, fill_value=0 ) #interpolate.interp2d(t, x, u_0, kind='cubic')
        u_0_new = np.array([interp((i[0], i[1])) for i in self.inputs]).reshape((-1, 1))
        return u_0_new

    def get_inputs(self, size):
        if size == 5:
            x_l = np.array([[[xt[0] - self.hx, xt[1]]] for xt in self.x_train])
            x_b = np.array([[[xt[0], xt[1] - self.ht]] for xt in self.x_train])
            x_r = np.array([[[xt[0] + self.hx, xt[1]]] for xt in self.x_train])
            x_t = np.array([[[xt[0], xt[1] + self.ht]] for xt in self.x_train])
            x = np.array([[[xt[0], xt[1]]] for xt in self.x_train])
            inputs = np.concatenate((x_l, x_b, x_r, x_t, x), axis = 1).reshape((-1, 2)) 
            
        elif size == 9:
            x_l = np.array([[[xt[0] - self.hx, xt[1]]] for xt in self.x_train])
            x_b = np.array([[[xt[0], xt[1] - self.ht]] for xt in self.x_train])
            x_r = np.array([[[xt[0] + self.hx, xt[1]]] for xt in self.x_train])
            x_t = np.array([[[xt[0], xt[1] + self.ht]] for xt in self.x_train])
            x_ll = np.array([[[xt[0] - self.hx, xt[1] - self.ht]] for xt in self.x_train])
            x_bb = np.array([[[xt[0] - self.hx, xt[1] + self.ht]] for xt in self.x_train])
            x_rr = np.array([[[xt[0] + self.hx, xt[1] - self.ht]] for xt in self.x_train])
            x_tt = np.array([[[xt[0] + self.hx, xt[1] + self.ht]] for xt in self.x_train])
            x = np.array([[[xt[0], xt[1]]] for xt in self.x_train])
            inputs = np.concatenate((x_l, x_b, x_r, x_t,x_ll, x_bb, x_rr, x_tt, x), axis = 1).reshape((-1, 2))
            
        elif size == 13:
            x_l  = np.array([[xt[0] - self.hx, xt[1]] for xt in self.x_train])
            x_b  = np.array([[xt[0], xt[1] - self.ht] for xt in self.x_train])
            x_r  = np.array([[xt[0] + self.hx, xt[1]] for xt in self.x_train])
            x_t  = np.array([[xt[0], xt[1] + self.ht] for xt in self.x_train])
            x_ll = np.array([[xt[0] - self.hx, xt[1] - self.ht] for xt in self.x_train])
            x_bb = np.array([[xt[0] - self.hx, xt[1] + self.ht] for xt in self.x_train])
            x_rr = np.array([[xt[0] + self.hx, xt[1] - self.ht] for xt in self.x_train])
            x_tt = np.array([[xt[0] + self.hx, xt[1] + self.ht] for xt in self.x_train])
            
            x_l2 = np.array([[xt[0] - 2*self.hx, xt[1]] for xt in self.x_train])
            x_b2 = np.array([[xt[0], xt[1] - 2*self.ht] for xt in self.x_train])
            x_r2 = np.array([[xt[0] + 2*self.hx, xt[1]] for xt in self.x_train])
            x_t2 = np.array([[xt[0], xt[1] + 2*self.ht] for xt in self.x_train])
            x_c  = np.array([[xt[0], xt[1]] for xt in self.x_train])
            components = [x_l, x_b, x_r, x_t, x_ll, x_bb, x_rr, x_tt, x_l2, x_b2, x_r2, x_t2, x_c]
            self.mask = np.any(np.hstack(components) < 0, axis=1)
            components = [comp[~self.mask] for comp in components]
            inputs = np.concatenate(components,axis=1).reshape((-1, 2))
            
        elif size == 25:
            x_b  = np.array([[xt[0], xt[1] - self.ht] for xt in self.x_train])
            x_r  = np.array([[xt[0] + self.hx, xt[1]] for xt in self.x_train])
            x_t  = np.array([[xt[0], xt[1] + self.ht] for xt in self.x_train])
            x_l  = np.array([[xt[0] - self.hx, xt[1]] for xt in self.x_train])

            x_ll = np.array([[xt[0] - self.hx, xt[1] - self.ht] for xt in self.x_train])
            x_bb = np.array([[xt[0] - self.hx, xt[1] + self.ht] for xt in self.x_train])
            x_rr = np.array([[xt[0] + self.hx, xt[1] - self.ht] for xt in self.x_train])
            x_tt = np.array([[xt[0] + self.hx, xt[1] + self.ht] for xt in self.x_train])

            x_l2 = np.array([[xt[0] - 2*self.hx, xt[1]] for xt in self.x_train])
            x_b2 = np.array([[xt[0], xt[1] - 2*self.ht] for xt in self.x_train])
            x_r2 = np.array([[xt[0] + 2*self.hx, xt[1]] for xt in self.x_train])
            x_t2 = np.array([[xt[0], xt[1] + 2*self.ht] for xt in self.x_train])

            x_ll2 = np.array([[xt[0] - 2*self.hx, xt[1] - self.ht] for xt in self.x_train])
            x_bb2 = np.array([[xt[0] - 2*self.hx, xt[1] + self.ht] for xt in self.x_train])
            x_rr2 = np.array([[xt[0] + 2*self.hx, xt[1] - self.ht] for xt in self.x_train])
            x_tt2 = np.array([[xt[0] + 2*self.hx, xt[1] + self.ht] for xt in self.x_train])

            x_ll3 = np.array([[xt[0] - self.hx, xt[1] - 2*self.ht] for xt in self.x_train])
            x_bb3 = np.array([[xt[0] + self.hx, xt[1] - 2*self.ht] for xt in self.x_train])
            x_rr3 = np.array([[xt[0] - self.hx, xt[1] + 2*self.ht] for xt in self.x_train])
            x_tt3 = np.array([[xt[0] + self.hx, xt[1] + 2*self.ht] for xt in self.x_train])

            x_ll4 = np.array([[xt[0] - 2*self.hx, xt[1] - 2*self.ht] for xt in self.x_train])
            x_bb4 = np.array([[xt[0] - 2*self.hx, xt[1] + 2*self.ht] for xt in self.x_train])
            x_rr4 = np.array([[xt[0] + 2*self.hx, xt[1] - 2*self.ht] for xt in self.x_train])
            x_tt4 = np.array([[xt[0] + 2*self.hx, xt[1] + 2*self.ht] for xt in self.x_train])

            x_c = np.array([[xt[0], xt[1]] for xt in self.x_train])
            
            components = [
                x_b, x_r, x_t, x_l,
                x_ll, x_bb, x_rr, x_tt,
                x_l2, x_b2, x_r2, x_t2,
                x_ll2, x_bb2, x_rr2, x_tt2,
                x_ll3, x_bb3, x_rr3, x_tt3,
                x_ll4, x_bb4, x_rr4, x_tt4,
                x_c
            ]
            
            self.mask = np.any(np.hstack(components) < 0, axis=1)
            components = [comp[~self.mask] for comp in components]
            inputs = np.concatenate(components,axis=1).reshape((-1, 2))
            
        elif size == 49:
            main = []
            for nx in range(-2, 3):
                for ny in range(-2, 3):
                    x_offset = np.array([
                        [xt[0] - nx * self.hx, xt[1] - ny * self.ht]
                        for xt in self.x_train
                    ])
                    main.append(x_offset)
            edge1 = []
            for ny in range(-2, 4):
                x_edge = np.array([[xt[0] + 3*self.hx, xt[1] - ny*self.ht] for xt in self.x_train])
                edge1.append(x_edge)

            edge2 = []
            for nx in range(-2, 4):
                x_edge = np.array([[xt[0] - nx*self.hx, xt[1] + 3*self.ht] for xt in self.x_train])
                edge2.append(x_edge)
            
            edge3 = []
            for ny in range(-2, 4):
                x_edge = np.array([[xt[0] - 3*self.hx, xt[1] - ny*self.ht] for xt in self.x_train])
                edge3.append(x_edge)
            
            edge4 = []
            for nx in range(-2, 3):
                x_edge = np.array([[xt[0] - nx*self.hx, xt[1] - 3*self.ht] for xt in self.x_train])
                edge4.append(x_edge)

            corner = np.array([[xt[0] + 3*self.hx, xt[1] + 3*self.ht] for xt in self.x_train])
            
            components = main + edge1 + edge2 + edge3 + edge4 + [corner]            
            self.mask = np.any(np.hstack(components) < 0, axis=1)
            components = [comp[~self.mask] for comp in components]
            inputs = np.concatenate(components,axis=1).reshape((-1, 2))
            
        elif size == 81:
            main = []
            i = 0
            for nx in range(-3, 4):
                for ny in range(-3, 4):
                    i += 1
                    x_offset = np.array([
                        [xt[0] - nx * self.hx, xt[1] - ny * self.ht]
                        for xt in self.x_train
                    ])
                    main.append(x_offset)
            edge1 = []
            for ny in range(-3, 5):
                x_edge = np.array([[xt[0] + 4*self.hx, xt[1] - ny*self.ht] for xt in self.x_train])
                edge1.append(x_edge)

            edge2 = []
            for nx in range(-3, 5):
                x_edge = np.array([[xt[0] - nx*self.hx, xt[1] + 4*self.ht] for xt in self.x_train])
                edge2.append(x_edge)
            
            edge3 = []
            for ny in range(-3, 5):
                x_edge = np.array([[xt[0] - 4*self.hx, xt[1] - ny*self.ht] for xt in self.x_train])
                edge3.append(x_edge)
            
            edge4 = []
            for nx in range(-3, 4):
                x_edge = np.array([[xt[0] - nx*self.hx, xt[1] - 4*self.ht] for xt in self.x_train])
                edge4.append(x_edge)

            corner = np.array([[xt[0] + 4*self.hx, xt[1] + 4*self.ht] for xt in self.x_train])
            
            components = main + edge1 + edge2 + edge3 + edge4 + [corner]            
            self.mask = np.any(np.hstack(components) < 0, axis=1)
            components = [comp[~self.mask] for comp in components]
            inputs = np.concatenate(components,axis=1).reshape((-1, 2))
        return inputs

    def construct_local_domain(self, f, outputs, size):
        if size == 5:
            outputs_u = outputs[4::5]
            u_l = outputs[0::5]
            u_b = outputs[1::5]
            u_r = outputs[2::5]
            u_t = outputs[3::5]
            inputs = np.concatenate((u_l, u_b, u_r, u_t, f), axis = 1)
        elif size == 9:
            outputs_u = outputs[8::9]
            u_l = outputs[0::9]
            u_b = outputs[1::9]
            u_r = outputs[2::9]
            u_t = outputs[3::9]
            u_ll = outputs[4::9]
            u_bb = outputs[5::9]
            u_rr = outputs[6::9]
            u_tt = outputs[7::9]
            inputs = np.concatenate((u_l, u_b, u_r, u_t, u_ll, u_bb, u_rr, u_tt, f), axis = 1)
        elif size == 13:
            outputs_u = outputs[12::13]
            u_l   = self.interp_f[0::13]
            u_r   = self.interp_f[1::13]
            u_b   = self.interp_f[2::13]
            u_t   = self.interp_f[3::13]
            u_ll  = outputs[4::13]
            u_bb  = outputs[5::13]
            u_rr  = outputs[6::13]
            u_tt  = outputs[7::13]
            u_l2  = outputs[8::13]
            u_r2  = outputs[9::13]
            u_b2  = outputs[10::13]
            u_t2  = outputs[11::13]
            inputs = np.concatenate(
                (u_l, u_r, u_b, u_t,
                u_ll, u_bb, u_rr, u_tt,
                u_l2, u_r2, u_b2, u_t2,
                f),
                axis=1
            )

        elif size == 25:
            outputs_u = outputs[24::25]
            arr = []
            for i in range(24):
                if i < 8:
                    arr.append(self.interp_f[i::25])  # from self.interp_f
                else:
                    arr.append(outputs[i::25])
            arr.append(f)
            inputs = np.concatenate(arr, axis=1)
        
        elif size == 49:
            outputs_u = outputs[12::13]
            arr = []
            for i in range(49):
                if i == 12:
                    arr.append(f)
                elif i < 25:
                    arr.append(self.interp_f[i::49])
                else:
                    arr.append(outputs[i::49])
            inputs = np.concatenate(arr, axis=1)
        
        elif size == 81:
            outputs_u = outputs[24::81]
            arr = []
            for i in range(81):
                if i == 24:
                    arr.append(f)
                elif i < 49:
                    arr.append(self.interp_f[i::81])
                else:
                    arr.append(outputs[i::81])
            inputs = np.concatenate(arr, axis=1)
        return np.array(inputs), outputs_u

    def load_trained_model(self, data, layers, best_step):
        net = dde.nn.FNN(layers, "tanh", "Glorot uniform", use_bias = True) #, regularization=['l2', 1e-8])
        model = dde.Model(data, net)
        model.compile("adam", lr=1e-3, decay = ("inverse time", 20000, 0.5), metrics=["l2 relative error"])
        model.compile("L-BFGS-B",  metrics=["l2 relative error"])
        model.restore(f"model/{best_step}.ckpt", verbose=1)
        return model

def solve_nn(Nx, Nt, dataset_G, data, size, pre_layers, best_step, dname, PATH, isplot=False):
    os.makedirs(f"{dname}/history_cLOINN_r_{size}", exist_ok = True)
    x_train = data.train_x

    l2_errs = []
    net = dde.nn.FNN([2] + [16] + [1], "tanh", "Glorot normal")

    def output_transform(x, y):
        x0, y0 = x[:, 0:1], x[:, 1:2]
        return y0 * (y0 - 1)  * x0 * (x0 - 1) * y
        
    net.apply_output_transform(output_transform)
    model = dde.Model(data, net)
    
    l2_errs = []
    iters = 20000
    checker = dde.callbacks.ModelCheckpoint(f"model/clmodel_{size}", save_better_only=True, period=1000)
    model.compile("adam", lr=1e-3, decay = ("inverse time", 5000, 1), metrics=["l2 relative error"])

    update = UpdateOutput(Nx, Nt, dataset_G, net, pre_layers, best_step, x_train, l2_errs, size, dname, PATH)
    losshistory, train_state = model.train(iterations=iters,  callbacks=[update, checker], model_save_path = f"model/clmodel_{size}")
    dde.saveplot(losshistory, train_state, issave=False, isplot=isplot, output_dir = f"{dname}/history_cLOINN_r_{size}")
    model.restore(f"model/clmodel_{size}-{iters}.ckpt", verbose=1)

    #predict
    u_0 = np.loadtxt(f"{PATH}/data_0.20/data_0/u_0_grid.dat").reshape((-1,1))
    u_true = np.loadtxt(f"{dname}/u_new_grid.dat").reshape((-1,1))
    u_pred = model.predict(data.test_x).reshape((-1,1))+u_0
    err = np.linalg.norm(u_pred - u_true)/np.linalg.norm(u_true)
    print("l2 relative error: ", err)
    np.savetxt(f"{dname}/u_cLOINN_r_{size}.dat",u_pred)

    l2_errs.append([iters,  err])
    l2_errs = np.array(l2_errs).reshape((-1,2))
    np.savetxt(f"{dname}/err_cLOINN_r_{size}.dat",l2_errs)
    if isplot:
        fig = plt.figure()
        plt.rcParams.update({'font.size': 25})
        plt.plot(l2_errs[:,0], l2_errs[:, 1])
        plt.xlabel("# Epochs")
        plt.ylabel("$L^2$ relative error")
        plt.show()
    return err, train_state.best_step

def main(sigma, num_func, size, parent_dir = "../../data/", gen = False):
    M = 201
    Nx, Ny = 101, 101
    N_f = 101*101
    N_b = 0
    l, a = 0.01, 0.10
    l_new, a_new = 0.1, float(sigma)
    
    new_dir = "data_nonlinear_poisson"
    PATH = os.path.join(parent_dir, new_dir)
    
    if size == 5:
        best_step = "model.ckpt-101090" #"model_5-100213"
        pre_layers = [5, 64, 1] #[5, 64, 1]
    elif size == 9:
        best_step = "model.ckpt-101228" #"model_9-100195"
        pre_layers = [9, 64, 1]
    elif size == 13:
        # best_step = "model_13-100391"#"model_13-100084"
        # pre_layers = [13,64,1]
        best_step = "model_13-105158" #104961" # "model_13-104070"
        pre_layers = [13,64,5]
    elif size == 25:
        # best_step = "model_25-100162"#"model_25-100089"
        # pre_layers = [25, 64, 1]
        best_step = "model_25-102896"#                                                                                "model_25-100089"
        pre_layers = [25, 64, 9]
    elif size == 49:
        best_step = "model_49-101966"
        pre_layers = [49, 64, 25]
    elif size == 81:
        best_step = "model_81-103176" #3.84e-04
        pre_layers = [81, 64, 49]
    print(f"Size: {size}, Best step: {best_step}, Pre-layers: {pre_layers}")
    
    errs = []
    b_steps = [[0] for i in range(num_func)]

    for i in range(num_func):
        print("Dataset {}".format(i))
        dname = f"{PATH}/data_{sigma}/data_{i}"

        dataset_G, dataset = load_all_data(M, Nx, Ny, N_f, N_b, l, a, l_new, a_new, dname, size, gen, 
                                      correction = True, grid = False, isplot = False)
        #err,b_step = apply(solve_nn, (Nx, Ny, dataset_G, dataset, pre_layers, best_step, dname, PATH, True))
        err,b_step = solve_nn(Nx, Ny, dataset_G, dataset, size, pre_layers, best_step, dname, PATH, True)
        ts = time.time()
        print("cLOINN took {} s.".format(time.time()-ts))
        errs.append(err)
        b_steps[i][0] = b_step
        np.savetxt(os.path.join(f"{PATH}/data_{sigma}/", f"b_steps_cLOINN_r_{size}.dat"), b_steps)

    print(errs)
    np.savetxt(os.path.join(f"{PATH}/data_{sigma}/", f"errs_cLOINN_r_{size}.dat"), errs)
    print("The average l2 error is ", sum(errs)/num_func)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=5) # size of local domains
    parser.add_argument("--num", type=int, default=5) # Number of functions
    parser.add_argument("--sigma", type=str, default="0.20") # Amplitude in the GRF
    args = parser.parse_args()
    
    # save to log file
    logging.info(f"Arguments received: size={args.size}, num={args.num}, sigma={args.sigma}")
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"{LOG_DIR}/log_FPC_{args.size}_{args.num}_{args.sigma}_{current_time}.log"
    logger = Logger(filename)
    sys.stdout = logger
    
    main(args.sigma, args.num, args.size)
    logger.close()
