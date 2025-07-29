import os
import deepxde as dde
from deepxde.callbacks import Callback
from multiprocessing import Pool
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from utils import gen_all_data_grid, construct_data
dde.config.set_default_float("float64")
dde.config.disable_xla_jit()

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
    def __init__(self, Nx, Nt, dataset, net, pre_layers, best_step, x_train, d_num):
        super(UpdateOutput, self).__init__()
        # load the pre-trained model
        self.graph = tf.Graph()
        self.dataset = dataset
        self.Nx = Nx
        self.Nt = Nt
        self.x_train = x_train
        self.f_new = np.loadtxt("data{}/f_new_grid.dat".format(d_num))
        self.u_new = np.loadtxt("data{}/u_new_grid.dat".format(d_num))
        self.net_outputs = net.outputs
        self.feed_dict = net.feed_dict(False, self.x_train)
        with self.graph.as_default():
            self.trained_model = self.load_trained_model(dataset, pre_layers, best_step)

    def on_batch_begin(self):
        outputs = self.model.sess.run(self.net_outputs, feed_dict = self.feed_dict)
        outputs = outputs.reshape((self.Nx, self.Nt))
        
        with self.graph.as_default():
            train_u,_ = construct_data(self.f_new, outputs)
            pred_u = self.trained_model.predict(train_u)
        
        pred_u = pred_u.reshape((self.Nx-2, self.Nt-1))
        pred_u = np.concatenate((np.zeros((self.Nx-2, 1)), pred_u), axis = 1)
        pred_u = np.concatenate((np.zeros((1,self.Nt)), pred_u, np.zeros((1, self.Nt))), axis = 0)
        self.model.data.train_y = pred_u.reshape((-1, 1))

    def load_trained_model(self, data, layers, best_step):
        net = dde.nn.FNN(layers, "tanh", "LeCun normal", use_bias = True)
        model = dde.Model(data, net)
        # restore model
        model.compile("adam", lr=1e-3, decay = ("inverse time", 20000, 0.5), metrics=["l2 relative error"])
        model.compile("L-BFGS-B",  metrics=["l2 relative error"])
        model.restore("model/model.ckpt-" + best_step + ".ckpt", verbose=1)
        return model

def solve_nn(Nx, Nt, dataset_G, data, pre_layers, best_step, d_num):
    os.makedirs("data{}/history_FP_grid".format(d_num), exist_ok = True)
    x_train = data.train_x
    
    # define the new model
    net = dde.nn.FNN([2] + [128]*2 + [1], "relu", "LeCun normal")

    model = dde.Model(data, net)

    checker = dde.callbacks.ModelCheckpoint("model1/model.ckpt", save_better_only=True, period=1000)
    model.compile("adam", lr=1e-3, decay = ("inverse time", 20000, 0.8), metrics=["l2 relative error"])
    update = UpdateOutput(Nx, Nt, dataset_G, net, pre_layers, best_step, x_train, d_num)
    losshistory, train_state = model.train(epochs=2000)
    losshistory, train_state = model.train(epochs=1000,  callbacks=[update])
    losshistory, train_state = model.train(epochs=97000,disregard_previous_best=True,  callbacks=[update, checker], model_save_path = "model1/model.ckpt".format(d_num))
    dde.saveplot(losshistory, train_state, issave=True, isplot=False, output_dir = "data{}/history_FP_grid".format(d_num))
    model.restore("model1/model.ckpt-".format(d_num) + str(train_state.best_step)+".ckpt", verbose=1)
    
    #predict
    u_pred = model.predict(x_train)
    u_new = data.test_y
    err = dde.metrics.l2_relative_error(u_pred, u_new)
    np.savetxt("data{}/u_FP_grid.dat".format(d_num),u_pred)
    print("l2 relative error: ", err)
    u_pred = u_pred.reshape((Nx, Nt))
    u_new = u_new.reshape((Nx, Nt))
    
    plt.imshow(np.rot90(abs(u_pred - u_new)), cmap = "gnuplot", extent=(0,1,0,1), aspect='auto')
    plt.colorbar()
    plt.xlabel("x")
    plt.ylabel("t")
    plt.savefig("data{}/res_FP_grid.png".format(d_num))
    plt.show()
    return err, train_state.best_step

def main():
    errs = []
    b_steps = [[0] for i in range(100)]
    num = 100
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

        dataset_G, dataset = gen_all_data_grid(M, Nx, Nt, N_f, N_b, 0.01, 0.1, 0.1, std, gen, d_num)
        pre_layers = [4, 64, 1]

        best_step = "157037"
        err,b_step = apply(solve_nn, (Nx, Nt, dataset_G, dataset, pre_layers, best_step, d_num))
        errs.append(err)
        b_steps[i][0] = b_step
        #print(b_steps)
        np.savetxt("data_{}/b_steps_FP_grid.dat".format(num_std),b_steps)

    print(errs)
    #print(b_steps)
    np.savetxt("data_{}/errs_FP_grid.dat".format(num_std),errs)
    print("The average l2 error is ", sum(errs)/num)

if __name__ == "__main__":
    main()
    
