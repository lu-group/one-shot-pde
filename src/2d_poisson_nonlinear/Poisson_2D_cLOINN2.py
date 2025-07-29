import os
os.environ["TF_XLA_FLAGS"] = '--tf_xla_cpu_global_jit'
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
    def __init__(self, Nx, Nt, dataset, net, pre_layers, best_step, x_train, l2_errs, d_num):
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
        self.d = np.loadtxt("data{}/f_new.dat".format(d_num))
        self.x_grid = np.loadtxt("data{}/x_grid.dat".format(d_num)).reshape((-1, 1))
        self.y_grid = np.loadtxt("data{}/y_grid.dat".format(d_num)).reshape((-1, 1))
        self.u_0_grid = np.loadtxt("data{}/u_0_grid.dat".format(d_num)).reshape((-1, 1))
        self.u_new_grid = np.loadtxt("data{}/u_new_grid.dat".format(d_num)).reshape((-1, 1))
        self.f_new = np.loadtxt("data{}/f_new.dat".format(d_num))[:, 2].reshape((-1, 1))
        self.u_new = np.loadtxt("data{}/u_new.dat".format(d_num))[:, 2].reshape((-1, 1))
        self.u_init = np.loadtxt("data{}/u_init.dat".format(d_num))[:, 2].reshape((-1, 1))
        self.net_outputs = net.outputs
        self.inputs = self.get_inputs()
        self.feed_dict = net.feed_dict(False, self.inputs)
        self.u_0 = self.get_u_0(d_num)
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
            err = np.linalg.norm(outputs[8::9] - self.u_new)/np.linalg.norm(self.u_new)
            print(self.model.train_state.epoch, "l2 relative error: ",err)

    def get_u_0(self,d_num):
        x = np.linspace(0,1,201)
        t = np.linspace(0,1,201)
        u_0 = np.loadtxt("data{}/u_0.dat".format(d_num))
        interp = interpolate.interp2d(t, x, u_0, kind='cubic')
        u_0_new = np.array([interp(i[1], i[0]) for i in self.inputs]).reshape((-1, 1))
        return u_0_new

    def get_inputs(self):
        # u[i-1, j], u[i, j-1], u[i+1, j],u[i-1, j-1],  u[i-1, j+1], u[i+1, j-1], u[i+1, j+1], f[i, j]
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
        return inputs

    def construct_local_domain(self, f, outputs):
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
        return np.array(inputs), outputs_u

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

    l2_errs = []
    net = dde.nn.FNN([2] + [16] + [1], "tanh", "LeCun normal")

    def output_transform(x, y):
        x0, y0 = x[:, 0:1], x[:, 1:2]
        return y0 * (y0 - 1)  * x0 * (x0 - 1) * y
        
    net.apply_output_transform(output_transform)
    model = dde.Model(data, net)

    iters = 20000
    checker = dde.callbacks.ModelCheckpoint("model2/model.ckpt", save_better_only=True, period=1000)
    model.compile("adam", lr=1e-3, decay = ("inverse time", 5000, 1), metrics=["l2 relative error"])
    update = UpdateOutput(Nx, Nt, dataset_G, net, pre_layers, best_step, x_train, l2_errs, d_num)
    losshistory, train_state = model.train(epochs=iters,disregard_previous_best=True,  callbacks=[update, checker], model_save_path = "model2/model.ckpt".format(d_num))
    dde.saveplot(losshistory, train_state, issave=True, isplot = False, output_dir = "data{}/history_FPC".format(d_num))
    model.restore("model2/model.ckpt-".format(d_num) + "{}.ckpt".format(iters), verbose=1)

    #predict
    u_0 = np.loadtxt("data{}/u_0_grid.dat".format(d_num)).reshape((-1,1))
    u_true = np.loadtxt("data{}/u_new_grid.dat".format(d_num)).reshape((-1,1))
    u_pred = model.predict(data.test_x)+u_0
    u_pred = u_pred.reshape((-1,1))
    err = dde.metrics.l2_relative_error(u_pred, u_true)
    print("l2 relative error: ", err)
    np.savetxt("data{}/u_FPC.dat".format(d_num),u_pred)

    l2_errs.append([iters,  err])
    l2_errs = np.array(l2_errs).reshape((-1,2))
    print(l2_errs)
    np.savetxt("data{}/err_FPC.dat".format(d_num),l2_errs)
    # fig2 = plt.figure()
    # plt.rcParams.update({'font.size': 20})
    # plt.plot(l2_errs[:,0], l2_errs[:, 1])
    # plt.xlabel("# Epochs")
    # plt.ylabel("$L^2$ relative error")
    # #plt.savefig("data{}/err_FPC.png".format(d_num))
    return err, train_state.best_step

def main():
    errs = []
    b_steps = [[0] for i in range(100)]
    M = 201
    Nx, Ny = 101, 101
    N_f = 101*101
    N_b = 0
    std = 0.05  # std for delta f
    num_std = "0.05"
    gen = False  # generate new data or not
    num = 100

    for i in range(num):
        print("Dataset {}".format(i))
        os.makedirs("data2_{}/data_{}".format(num_std, i), exist_ok = True)
        d_num = "2_{}/data_{}".format(num_std, i)
        dataset_G, dataset = gen_all_data(M, Nx, Ny, N_f, N_b, 0.01, 0.1, 0.1, std, gen, d_num, correction = True)
        pre_layers = [9, 64, 1]
        best_step = "101228"
        err,b_step = apply(solve_nn, (Nx, Ny, dataset_G, dataset, pre_layers, best_step, d_num))
        errs.append(err)
        b_steps[i][0] = b_step
        print(b_steps)
        np.savetxt("data2_{}/b_steps_FPC.dat".format(num_std),b_steps)
    print(errs)
    print(b_steps)
    #np.savetxt("data_{}/b_steps_FPC.dat".format(num_std),b_steps)
    np.savetxt("data2_{}/errs_FPC.dat".format(num_std),errs)
    print("The average l2 error is ", sum(errs)/num)


if __name__ == "__main__":
    main()
