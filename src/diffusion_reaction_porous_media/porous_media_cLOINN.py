import os
import deepxde as dde
from deepxde.callbacks import Callback
from multiprocessing import Pool
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from utils import gen_all_data
from scipy import interpolate
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
    def __init__(self, Nx, Nt, dataset, net, pre_layers, best_step, x_train, l2_errs_a, l2_errs_b, d_num):
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
        self.d = np.loadtxt("data{}/f_new.dat".format(d_num))
        self.Ca_0_grid = np.loadtxt("data{}/Ca_0_grid.dat".format(d_num)).reshape((-1, 1))
        self.Cb_0_grid = np.loadtxt("data{}/Cb_0_grid.dat".format(d_num)).reshape((-1, 1))
        self.Ca_new_grid = np.loadtxt("data{}/Ca_new_grid.dat".format(d_num)).reshape((-1, 1))
        self.Cb_new_grid = np.loadtxt("data{}/Cb_new_grid.dat".format(d_num)).reshape((-1, 1))
        self.f_new = np.loadtxt("data{}/f_new.dat".format(d_num))[:, 2].reshape((-1, 1))
        self.Ca_new = np.loadtxt("data{}/Ca_new.dat".format(d_num))[:, 2].reshape((-1, 1))
        self.Cb_new = np.loadtxt("data{}/Cb_new.dat".format(d_num))[:, 2].reshape((-1, 1))
        self.Ca_init = np.loadtxt("data{}/Ca_init.dat".format(d_num))[:, 2].reshape((-1, 1))
        self.Cb_init = np.loadtxt("data{}/Cb_init.dat".format(d_num))[:, 2].reshape((-1, 1))
        self.net_outputs = net.outputs
        self.inputs = self.get_inputs()
        self.feed_dict = net.feed_dict(False, self.inputs)
        self.Ca_0, self.Cb_0 = self.get_u_0(d_num)
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
        Ca_0 = np.loadtxt("data{}/Ca_0.dat".format(d_num))
        Cb_0 = np.loadtxt("data{}/Cb_0.dat".format(d_num))
        interp_a = interpolate.interp2d(t, x, Ca_0, kind='cubic')
        interp_b = interpolate.interp2d(t, x, Cb_0, kind='cubic')
        Ca_0_new = np.array([interp_a(i[1], i[0]) for i in self.inputs]).reshape((-1, 1))
        Cb_0_new = np.array([interp_b(i[1], i[0]) for i in self.inputs]).reshape((-1, 1))
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

def solve_nn(Nx, Nt, dataset_G, data, pre_layers, best_step, d_num):
    os.makedirs("data{}/history_FPC".format(d_num), exist_ok = True)
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
    checker = dde.callbacks.ModelCheckpoint("model2/model.ckpt", save_better_only=True, period=1000)
    model.compile("adam", lr=1e-3, decay = ("inverse time", 10000, 0.5), metrics=["l2 relative error"])
    update = UpdateOutput(Nx, Nt, dataset_G, net, pre_layers, best_step, x_train, l2_errs_a, l2_errs_b, d_num)
    losshistory, train_state = model.train(epochs=iters,disregard_previous_best=True,  callbacks=[update, checker], model_save_path = "model2/model.ckpt".format(d_num))
    dde.saveplot(losshistory, train_state, issave=True, isplot = False, output_dir = "data{}/history_FPC".format(d_num))
    model.restore("model2/model.ckpt-".format(d_num) + "{}.ckpt".format(iters), verbose=1)

    #predict
    Ca_0 = np.loadtxt("data{}/Ca_0_grid.dat".format(d_num)).reshape((-1,1))
    Cb_0 = np.loadtxt("data{}/Cb_0_grid.dat".format(d_num)).reshape((-1,1))
    Ca_true = np.loadtxt("data{}/Ca_new_grid.dat".format(d_num)).reshape((-1,1))
    Cb_true = np.loadtxt("data{}/Cb_new_grid.dat".format(d_num)).reshape((-1,1))
    u_pred = model.predict(data.test_x)
    Ca_pred = u_pred[:][ :1] + Ca_0
    Cb_pred = u_pred[:][1:] + Cb_0
    Ca_pred, Cb_pred = Ca_pred.reshape((-1,1)), Cb_pred.reshape((-1,1))
    err_a = dde.metrics.l2_relative_error(Ca_pred, Ca_true)
    err_b = dde.metrics.l2_relative_error(Cb_pred, Cb_true)
    print("l2 relative error: ", err_a, " and ", err_b)
    np.savetxt("data{}/Ca_FPC.dat".format(d_num),Ca_pred)
    np.savetxt("data{}/Cb_FPC.dat".format(d_num),Cb_pred)

    l2_errs_a.append([iters,  err_a])
    l2_errs_b.append([iters,  err_b])
    l2_errs_a = np.array(l2_errs_a).reshape((-1,2))
    l2_errs_b = np.array(l2_errs_b).reshape((-1,2))
    np.savetxt("data{}/err_FPC_a.dat".format(d_num),l2_errs_a)
    np.savetxt("data{}/err_FPC_b.dat".format(d_num),l2_errs_b)
    # fig2 = plt.figure()
    # plt.rcParams.update({'font.size': 20})
    # plt.plot(l2_errs[:,0], l2_errs[:, 1])
    # plt.xlabel("# Epochs")
    # plt.ylabel("$L^2$ relative error")
    #plt.savefig("data{}/err_FPC.png".format(d_num))
    return err_a, err_b, train_state.best_step


def main():
    errs_a, errs_b = [], []
    b_steps = [[0] for i in range(100)]
    num = 1
    num_std = "0.10"
    std = 0.10  # std for delta f
    gen = False # generate new data or not
    M = 1001
    Nx, Nt = 101, 101
    N_f = 101*101
    N_b = 0

    for i in range(num):
        print("Dataset {}".format(i))
        os.makedirs("data_{}/data_{}".format(num_std, i), exist_ok = True)
        d_num = "_{}/data_{}".format(num_std, i)
        dataset_G, dataset = gen_all_data(M, Nx, Nt, N_f, N_b, 0.01, 0.1, 0.1, std, gen, d_num, correction = True)
        pre_layers = [7, 64, 2]
        best_step = "117616"
        err_a, err_b, b_step = apply(solve_nn, (Nx, Nt, dataset_G, dataset, pre_layers, best_step, d_num))
        errs_a.append(err_a)
        errs_b.append(err_b)
        b_steps[i][0] = b_step
        print(b_steps)
        np.savetxt("data_{}/b_steps_FPC.dat".format(num_std),b_steps)

    print(b_steps)
    #np.savetxt("data_{}/errs_a_FPC.dat".format(num_std),errs)
    #np.savetxt("data_{}/errs_b_FPC.dat".format(num_std),errs)
    print("The average l2 error is ", sum(errs_a)/num, " ", sum(errs_b)/num)


if __name__ == "__main__":
    main()


