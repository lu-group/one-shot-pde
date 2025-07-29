import os
os.environ["TF_USE_LEGACY_KERAS"]="1"
os.environ["TF_XLA_FLAGS"] = '--tf_xla_cpu_global_jit'
os.chdir(os.path.dirname(os.path.abspath(__file__)))
from utils import *

import time
import argparse
import deepxde as dde
import deepxde.backend as bkd
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

def S_0_func_np(x):
    return (1.0 - 0.5*np.cos(4 * np.pi * x)) #1 + 0.5*np.cos(x * np.pi *3) #1.3 + 1.0 * np.cos(x * np.pi * 2)

def I_0_func_np(x):
    return 0.3*np.exp(-((x - 2/3)**2) / (2 * 0.15**2))#0.01 * np.exp(-1000*x)

class UpdateOutput(Callback):
    def __init__(self, Nx, Nt, dataset, net, pre_layers, best_step, x_train, l2_errs_S, l2_errs_I, dname, PATH):
        super(UpdateOutput, self).__init__()
        self.graph = tf.Graph()
        self.dataset = dataset
        self.x_train = x_train
        self.Nx = Nx
        self.Nt = Nt
        self.hx = 1/(Nx-1)
        self.ht = 10/(Nt-1)
        self.net = net
    
        self.d = np.loadtxt(f"{dname}/f_new.dat")
        self.x_grid = np.loadtxt(f"{PATH}/data_0.10/x_grid.dat").reshape((-1, 1))
        self.t_grid = np.loadtxt(f"{PATH}/data_0.10/t_grid.dat").reshape((-1, 1))
        
        self.f_new = np.loadtxt(f"{dname}/f_new.dat")[:, -1].reshape((-1, 1))
        self.f_0 = np.loadtxt(f"{PATH}/data_0.10/data_0/f_0_grid.dat")[:, -1].reshape((-1, 1))
        self.S_0_grid = np.loadtxt(f"{PATH}/data_0.10/data_0/S_0_grid.dat").reshape((-1, 1))
        self.I_0_grid = np.loadtxt(f"{PATH}/data_0.10/data_0/I_0_grid.dat").reshape((-1, 1))
        self.S_new_grid  = np.loadtxt(f"{dname}/S_new_grid.dat").reshape((-1, 1))
        self.I_new_grid  = np.loadtxt(f"{dname}/I_new_grid.dat").reshape((-1, 1))
        self.S_new = np.loadtxt(f"{dname}/S_new.dat")[:, -1].reshape((-1, 1))
        self.I_new = np.loadtxt(f"{dname}/I_new.dat")[:, -1].reshape((-1, 1))
        self.S_init = np.loadtxt(f"{PATH}/data_0.10/data_0/S_init.dat")[:, -1].reshape((-1, 1))
        self.I_init = np.loadtxt(f"{PATH}/data_0.10/data_0/I_init.dat")[:, -1].reshape((-1, 1))
        
        self.inputs = self.get_inputs(PATH) #np.loadtxt(f"{PATH}/data_0.10/inputs.txt") #self.get_inputs(PATH)
        self.x_t0 = self.inputs[:-len(self.bc_points), 0:1][self.mask_t0]
        interp_f = interpolate.interp1d(self.x_train[:, 0][~self.mask_bc], self.f_new.flatten()[~self.mask_bc], kind='cubic', bounds_error=False, fill_value='extrapolate')
        # x_0 = np.loadtxt(f"{PATH}/data_0.10/data_0/f_0_grid.dat")[:, 0].reshape((-1, 1))
        # interp_f = interpolate.interp1d(x_0.flatten(), self.f_0.flatten(), kind='cubic', bounds_error=False, fill_value='extrapolate')
        self.interp_f = interp_f(self.inputs[:,0]).reshape((-1,1))
        self.feed_dict = net.feed_dict(False, self.inputs)
        self.S_0, self.I_0 = self.get_u_0(PATH)
        self.l2_errs_S = l2_errs_S
        self.l2_errs_I = l2_errs_I
        with self.graph.as_default():
            self.trained_model = self.load_trained_model(dataset, pre_layers, best_step)

    def on_batch_begin(self):
        outputs = self.model.sess.run(self.net.outputs, feed_dict = self.feed_dict)
        pred_grid = self.model.sess.run(self.net.outputs, feed_dict = self.net.feed_dict(False, self.model.data.test_x))
        outputs_S = outputs[:, 0:1] + self.S_0
        outputs_I = outputs[:, 1:2] + self.I_0

        outputs_S[:-len(self.bc_points)][self.mask_t0] = self.S_0[:-len(self.bc_points)][self.mask_t0]
        outputs_I[:-len(self.bc_points)][self.mask_t0] = self.I_0[:-len(self.bc_points)][self.mask_t0]

        with self.graph.as_default():
            train_u,So,Io = self.construct_local_domain(self.interp_f, outputs_S, outputs_I) #self.S_0, self.I_0)
            pred_u = self.trained_model.predict(train_u)
            pred_u = np.clip(pred_u, 1e-3, None)

        pred_S = np.vstack((pred_u[:, 1:2], outputs_S[-len(self.bc_points):]))
        pred_I = np.vstack((pred_u[:, 4:5], outputs_I[-len(self.bc_points):]))
        self.model.data.train_y = np.hstack((pred_S - self.S_init, 
                                             pred_I - self.I_init))
        
        if self.model.train_state.epoch % 1000 == 0:
            err_S = np.linalg.norm(pred_grid[:, 0:1] + self.S_0_grid - self.S_new_grid)/np.linalg.norm(self.S_new_grid)
            err_I = np.linalg.norm(pred_grid[:, 1:2] + self.I_0_grid - self.I_new_grid)/np.linalg.norm(self.I_new_grid)
            self.l2_errs_S.append([self.model.train_state.epoch, err_S])
            self.l2_errs_I.append([self.model.train_state.epoch, err_I])
            print(self.model.train_state.epoch, "Prediction l2 relative error: S -",err_S, " and I -", err_I)

        if self.model.train_state.epoch % 1000 == 0:
            err_S = np.linalg.norm(pred_S - self.S_new)/np.linalg.norm(self.S_new)
            err_I = np.linalg.norm(pred_I - self.I_new)/np.linalg.norm(self.I_new)
            print(self.model.train_state.epoch, "l2 relative error: S -",err_S, " and I -", err_I)
        
    def get_u_0(self, PATH):
        x = np.loadtxt(f"{PATH}/data_0.10/data_0/x.dat")
        t = np.loadtxt(f"{PATH}/data_0.10/data_0/t.dat")
        S_0 = np.loadtxt(f"{PATH}/data_0.10/data_0/S_0.dat")
        I_0 = np.loadtxt(f"{PATH}/data_0.10/data_0/I_0.dat")
        
        interp_S = interpolate.RegularGridInterpolator((x, t), S_0, method='cubic', bounds_error=False, fill_value=None)
        interp_I = interpolate.RegularGridInterpolator((x, t), I_0, method='cubic', bounds_error=False, fill_value=None)
        S_0_new = interp_S(self.inputs).reshape((-1, 1))
        I_0_new = interp_I(self.inputs).reshape((-1, 1))
        np.savetxt(f"{PATH}/data_0.10/S_0_inputs.txt", S_0_new)
        np.savetxt(f"{PATH}/data_0.10/I_0_inputs.txt", I_0_new)
        return S_0_new, I_0_new

    def get_inputs(self, PATH):
        self.mask_bc = (self.x_train[:, 0] == 0) | (self.x_train[:, 0] == 1) | (self.x_train[:, 1] == 0) 
        self.mask_bc_x0 = (self.x_train[:, 0] == 0)
        self.mask_bc_x1 = (self.x_train[:, 0] == 1)
        self.mask_bc_t0 = (self.x_train[:, 1] == 0)
        self.bc_points = self.x_train[self.mask_bc]
        self.x_points = self.x_train[~self.mask_bc]
        # u[i-1, j], u[i, j-1], u[i+1, j], f[i, j]
        x_l = np.array([[[xt[0] - self.hx, xt[1]]] for xt in self.x_points])
        x_l1 = np.array([[[xt[0] - 2*self.hx, xt[1]]] for xt in self.x_points])
        x_b = np.array([[[xt[0] + self.hx, xt[1]]] for xt in self.x_points])
        x_b1 = np.array([[[xt[0] + 2*self.hx, xt[1]]] for xt in self.x_points])
        
        x_r = np.array([[[xt[0] - self.hx, xt[1] - self.ht]] for xt in self.x_points])
        x_r1 = np.array([[[xt[0] - 2*self.hx, xt[1] - self.ht]] for xt in self.x_points])
        x1 = np.array([[[xt[0], xt[1] - self.ht]] for xt in self.x_points])
        x_t = np.array([[[xt[0] + self.hx, xt[1] - self.ht]] for xt in self.x_points])
        x_t1 = np.array([[[xt[0] + 2*self.hx, xt[1] - self.ht]] for xt in self.x_points])
        
        x = np.array([[[xt[0], xt[1]]] for xt in self.x_points])
        inputs = np.concatenate((x_l, x_l1, x_b, x_b1, x_r, x_r1, x1, x_t,x_t1, x), axis = 1).reshape((-1,2))
        self.mask_x0 = np.isclose(inputs[:, 0], 0, atol=self.hx)
        self.mask_x1 = np.isclose(inputs[:, 0], 1, atol=self.hx)
        self.mask_t0 = np.isclose(inputs[:, 1], 0, atol=self.ht)
        inputs = np.concatenate((inputs, self.bc_points))
        np.savetxt(f"{PATH}/data_0.10/inputs.txt", inputs)
        return inputs


    def construct_local_domain(self, f, outputs1, outputs2): 
        outputs1 = outputs1[:-len(self.bc_points)]
        outputs2 = outputs2[:-len(self.bc_points)]
        f = f[:-len(self.bc_points)]
        inputs = np.concatenate((outputs1[0::10], outputs1[1::10], outputs1[2::10],
                                 outputs1[3::10], outputs1[4::10], outputs1[5::10],
                                 outputs1[6::10], outputs1[7::10], outputs1[8::10],
                                 outputs2[0::10], outputs2[1::10], outputs2[2::10],
                                 outputs2[3::10], outputs2[4::10], outputs2[5::10],
                                 outputs2[6::10], outputs2[7::10], outputs2[8::10],
                                 f[4::10], f[9::10], f[7::10] ), axis = 1)
        outputs_S = outputs1[9::10]
        outputs_I = outputs2[9::10]
        return np.array(inputs), outputs_S, outputs_I

    def load_trained_model(self, data, layers, best_step):
        net = dde.nn.FNN(layers, "silu", "Glorot uniform", use_bias = True) #, regularization=['l2', 1e-8])
        model = dde.Model(data, net)
        # restore model
        model.compile("adam", lr=1e-3, decay = ("inverse time", 20000, 0.5), metrics=["l2 relative error"])
        model.compile("L-BFGS-B",  metrics=["l2 relative error"])
        model.restore(f"model/{best_step}.ckpt", verbose=1)
        return model

def S_0_func(x):
    return (1.0 - 0.5*tf.cos(4 * np.pi * x))

def I_0_func(x):
    return 0.3*tf.exp(-((x - 2/3)**2) / (2 * 0.15**2))

def solve_nn(Nx, Nt, dataset_G, data, pre_layers, best_step, dname, PATH, sigma, isplot=False):
    os.makedirs(f"{dname}/history_cLOINN_r", exist_ok = True)
    x_train = data.train_x
    l2_errs = []
    net = dde.nn.FNN([2] + [128]*2 + [2], "tanh", "Glorot uniform")

    def output_transform(x, y):
        """
        Positive outputs; Dirichlet IC; Nuemann BC
        """
        x0, t0 = x[:, 0:1], x[:, 1:2]
        S, I = y[:, 0:1], y[:, 1:2]
        S = tf.nn.softplus(S) - 0.1
        I = tf.nn.softplus(I) - 0.1
        # S = tf.nn.silu(S)
        # I = tf.nn.silu(I)
        multiplier = x0**2 * (1 - x0)**2
        S_out  = 0 + t0 * S * multiplier
        I_out = 0 + t0 * I * multiplier
        return tf.concat([S_out, I_out], axis=1)

    net.apply_output_transform(output_transform)
    model = dde.Model(data, net)

    def mean_squared_error(y_true, y_pred):
        S_true = y_true[:, :1]
        I_true = y_true[:, 1:]
        S_pred = y_pred[:, :1]
        I_pred = y_pred[:, 1:]

        mse_S = bkd.reduce_mean(bkd.square(S_true - S_pred))
        mse_I = bkd.reduce_mean(bkd.square(I_true - I_pred))

        return [1*mse_S, 10*mse_I]

    l2_errs_S, l2_errs_I = [], []
    iters = 60000
    checker = dde.callbacks.ModelCheckpoint(f"model/clmodel_{sigma}", save_better_only=True, period=1000)
    model.compile("adam", lr=1e-3, decay = ("inverse time", 20000, 0.5), metrics=["l2 relative error"], loss=mean_squared_error)
    update = UpdateOutput(Nx, Nt, dataset_G, net, pre_layers, best_step, x_train, l2_errs_S, l2_errs_I, dname, PATH)
    losshistory, train_state = model.train(iterations=iters,  callbacks=[update, checker], model_save_path = f"model/clmodel_{sigma}")
    dde.saveplot(losshistory, train_state, issave=False, isplot=isplot, output_dir = f"{dname}/history_cLOINN_r")
    best_step = train_state.best_step
    if best_step == 0:
        best_step = iters
    model.restore(f"model/clmodel_{sigma}-{best_step}.ckpt", verbose=1)

    #predict
    x_test = data.test_x
    S_0_grid = np.loadtxt(f"{PATH}/data_0.10/data_0/S_0_grid.dat").reshape((-1, 1))
    I_0_grid = np.loadtxt(f"{PATH}/data_0.10/data_0/I_0_grid.dat").reshape((-1, 1))
    u_pred = model.predict(x_test)
    S_pred, I_pred = u_pred[:, 0:1] +S_0_grid, u_pred[:, 1:2]+I_0_grid
    S_true  = np.loadtxt(f"{dname}/S_new_grid.dat").reshape((-1, 1))
    I_true  = np.loadtxt(f"{dname}/I_new_grid.dat").reshape((-1, 1))
    
    err_S = np.linalg.norm(S_pred - S_true)/np.linalg.norm(S_true)
    err_I = np.linalg.norm(I_pred - I_true)/np.linalg.norm(I_true)
    print("l2 relative error: ",err_S, err_I)
    np.savetxt(f"{dname}/S_cLOINN_r.dat",S_pred)
    np.savetxt(f"{dname}/I_cLOINN_r.dat",I_pred)

    l2_errs_S.append([iters,  err_S])
    l2_errs_I.append([iters,  err_I])
    l2_errs_S = np.array(l2_errs_S).reshape((-1,2))
    l2_errs_I = np.array(l2_errs_I).reshape((-1,2))
    np.savetxt(f"{dname}/err_S_cLOINN_r.dat",l2_errs_S)
    np.savetxt(f"{dname}/err_I_cLOINN_r.dat",l2_errs_I)

    if isplot:
        fig = plt.figure()
        plt.rcParams.update({'font.size': 20,"savefig.dpi": 200, "figure.figsize": (8, 6)})
        plt.subplots_adjust(left=0.20, right=0.9, top=0.9, bottom=0.15)
        plt.plot(l2_errs_S[:,0], l2_errs_S[:, 1])
        plt.xlabel("# Epochs")
        plt.ylabel("$L^2$ relative error")
        plt.savefig(f"{dname}/errors_FPC_S.png")
        plt.close()

        fig = plt.figure()
        plt.rcParams.update({'font.size': 20,"savefig.dpi": 200, "figure.figsize": (8, 6)})
        plt.subplots_adjust(left=0.20, right=0.9, top=0.9, bottom=0.15)
        plt.plot(l2_errs_I[:,0], l2_errs_I[:, 1])
        plt.xlabel("# Epochs")
        plt.ylabel("$L^2$ relative error")
        plt.savefig(f"{dname}/errors_FPC_I.png")
        plt.close()

        plt.figure()
        plt.rcParams.update({'font.size': 20,"savefig.dpi": 200, "figure.figsize": (8, 6)})
        plt.subplots_adjust(left=0.20, right=0.9, top=0.9, bottom=0.15)
        plt.imshow(np.rot90(abs(S_pred - S_true).reshape((101, 201))), cmap = "rainbow", extent=(0,1,0,1), aspect='auto')
        plt.colorbar()
        plt.xlabel("x")
        plt.ylabel("t")
        plt.savefig(f"{dname}/res_FPC_S.png")
        plt.close()

        plt.figure()
        plt.rcParams.update({'font.size': 20,"savefig.dpi": 200, "figure.figsize": (8, 6)})
        plt.subplots_adjust(left=0.20, right=0.9, top=0.9, bottom=0.15)
        plt.imshow(np.rot90(abs(I_pred - I_true).reshape((101, 201))), cmap = "rainbow", extent=(0,1,0,1), aspect='auto')
        plt.colorbar()
        plt.xlabel("x")
        plt.ylabel("t")
        plt.savefig(f"{dname}/res_FPC_I.png")
        plt.close()
    return err_S, err_I, best_step


def main(sigma, num_func, parent_dir = "../../data/", gen = False):
    Mx, Mt = 401, 1001
    Nx, Nt = 101, 201
    N_f = 101*201
    N_b = 0
    l, a = 0.05, "0.10"
    l_new, a_new = 0.1, float(sigma)

    # Create folders for the datasets
    new_dir = "data_SIR"
    PATH = os.path.join(parent_dir, new_dir)

   # Load model
    best_step = "model-35697"
    pre_layers = [21, 128, 6]
    
    errs_S = []
    errs_I = []

    b_steps = [[0] for i in range(num_func)]

    for i in range(num_func):
        print("Dataset {}".format(i))
        dname = f"{PATH}/data_{sigma}/data_{i}"

        dataset_G, dataset = load_all_data(Mx, Mt, Nx, Nt, N_f, N_b, l, a, l_new, a_new, 
                                 dname, gen, 
                                 correction = True, grid = False, isplot = False)
        #err,b_step = apply(solve_nn, (Nx, Ny, dataset_G, dataset, pre_layers, best_step, dname, PATH, True))
        err_S, err_I, b_step = solve_nn(Nx, Nt, dataset_G, dataset, pre_layers, best_step, dname, PATH, sigma, False)
        errs_S.append(err_S)
        errs_I.append(err_I)
        b_steps[i][0] = b_step
        np.savetxt(os.path.join(f"{PATH}/data_{sigma}/", f"b_steps_cLOINN_r.dat"), b_steps)

    np.savetxt(os.path.join(f"{PATH}/data_{sigma}/", f"errs_S_cLOINN_r.dat"), errs_S)
    np.savetxt(os.path.join(f"{PATH}/data_{sigma}/", f"errs_I_cLOINN_r.dat"), errs_I)
    print("The average l2 error is ", sum(errs_S)/num_func, sum(errs_I)/num_func)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num", type=int, default=3) # Number of functions
    parser.add_argument("--sigma", type=str, default="0.10") # Amplitude in the GRF
    args = parser.parse_args()
    main(args.sigma, args.num)