import os
os.environ["TF_XLA_FLAGS"] = '--tf_xla_cpu_global_jit'
import deepxde as dde
from deepxde.callbacks import Callback
from multiprocessing import Pool
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from utils import gen_all_data_grid, construct_data
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
    def __init__(self, Nx, Nt, dataset, net, pre_layers, best_step, x_train, l2_errs, d_num):
        super(UpdateOutput, self).__init__()
        self.graph = tf.Graph()
        self.dataset = dataset
        self.l2_errs = l2_errs
        self.x_train = x_train
        self.Nx = Nx
        self.Nt = Nt
        self.x = np.linspace(0, 1, Nx)
        self.t = np.linspace(0, 1, Nt)
        self.f_new = np.loadtxt("data{}/f_new_grid.dat".format(d_num))
        self.u_new = np.loadtxt("data{}/u_new_grid.dat".format(d_num))
        self.u_0 = np.loadtxt("data{}/u_0_grid.dat".format(d_num))
        self.net_outputs = net.outputs
        self.feed_dict = net.feed_dict(False, self.x_train)
        self.l2_errs = l2_errs
        self.boundary = np.array([np.sin(np.pi*i) for i in self.t[1:]])
        self.init_boundary = np.array([i**2 for i in self.x[:]])
        with self.graph.as_default():
            self.trained_model = self.load_trained_model(dataset, pre_layers, best_step)

    def on_batch_begin(self):
        outputs = self.model.sess.run(self.net_outputs, feed_dict = self.feed_dict)
        outputs = outputs.reshape((self.Nx, self.Nt)) + self.u_0

        with self.graph.as_default():
            train_u,_ = construct_data(self.f_new, outputs)
            pred_u = self.trained_model.predict(train_u)

        pred_u = pred_u.reshape((self.Nx-1, self.Nt-1))
        pred_u = np.concatenate((self.boundary.reshape((1,self.Nt - 1)), pred_u), axis = 0)
        pred_u = np.concatenate((self.init_boundary.reshape((self.Nx, 1)), pred_u), axis = 1)

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

def solve_nn(Nx, Nt, dataset_G, data, pre_layers, best_step, d_num):
    os.makedirs("data{}/history_FPC_grid".format(d_num), exist_ok = True)
    x_train = data.train_x
    l2_errs = []
    net = dde.nn.FNN([2] + [64]*2 + [1], "tanh", "LeCun normal")
    model = dde.Model(data, net)

    iters = 100000
    checker = dde.callbacks.ModelCheckpoint("model2/model.ckpt", save_better_only=True, period=1000)
    model.compile("adam", lr=1e-3, decay = ("inverse time", 10000, 0.8), metrics=["l2 relative error"])
    update = UpdateOutput(Nx, Nt, dataset_G, net, pre_layers, best_step, x_train, l2_errs, d_num)
    losshistory, train_state = model.train(epochs=iters,disregard_previous_best=True,  callbacks=[update, checker], model_save_path = "model2/model.ckpt".format(d_num))
    dde.saveplot(losshistory, train_state, issave=True, isplot = False, output_dir = "data{}/history_FPC_grid".format(d_num))
    model.restore("model2/model.ckpt-".format(d_num) + "{}.ckpt".format(iters), verbose=1)

    #predict
    u_0 = np.loadtxt("data{}/u_0_grid.dat".format(d_num)).reshape((-1,1))
    u_true = np.loadtxt("data{}/u_new_grid.dat".format(d_num)).reshape((-1,1))

    u_pred = model.predict(x_train)+u_0
    u_pred = u_pred.reshape((-1,1))
    err = dde.metrics.l2_relative_error(u_pred, u_true)
    print("l2 relative error: ", err)
    np.savetxt("data{}/u_FPC_grid.dat".format(d_num),u_pred)

    l2_errs.append([iters,  err])
    l2_errs = np.array(l2_errs).reshape((-1,2))
    print(l2_errs)
    np.savetxt("data{}/err_FPC_grid.dat".format(d_num),l2_errs)
    # fig2 = plt.figure()
    # plt.rcParams.update({'font.size': 20})
    # plt.plot(l2_errs[:,0], l2_errs[:, 1])
    # plt.xlabel("# Epochs")
    # plt.ylabel("$L^2$ relative error")
    # plt.savefig("data{}/err_FPC_grid.png".format(d_num))
    return err, train_state.best_step

def main():
    errs = []
    b_steps = [[0] for i in range(100)]
    num = 100
    num_std = "0.50"
    std = 0.50  # std for delta f
    gen = False # generate new data or not
    M = 1001
    Nx, Nt = 101, 101
    N_f = 101*101
    N_b = 0

    for i in range(num):
        print("Dataset {}".format(i))
        os.makedirs("data_{}/data_{}".format(num_std, i), exist_ok = True)
        d_num = "_{}/data_{}".format(num_std, i)
        dataset_G, dataset = gen_all_data_grid(M, Nx, Nt, N_f, N_b, 0.01, 0.1, 0.1, std, gen, d_num, correction = True)
        pre_layers = [4, 64, 1]
        best_step = "118513"
        err,b_step = apply(solve_nn, (Nx, Nt, dataset_G, dataset, pre_layers, best_step, d_num))
        errs.append(err)
        b_steps[i][0] = b_step
        print(b_steps)
        np.savetxt("data_{}/b_steps_FPC_grid.dat".format(num_std),b_steps)

    print(b_steps)
    #np.savetxt("data_{}/b_steps_FPC.dat".format(num_std),b_steps)
    np.savetxt("data_{}/errs_FPC_grid.dat".format(num_std),errs)
    print("The average l2 error is ", sum(errs)/num)


if __name__ == "__main__":
    main()
