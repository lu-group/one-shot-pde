import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import time
import argparse
import deepxde as dde
import numpy as np
import matplotlib.pyplot as plt
dde.config.set_default_float("float64")
dde.config.disable_xla_jit()
from utils import *

def S_0_func(x):
    return (1.0 - 0.5*np.cos(4 * np.pi * x))

def I_0_func(x):
    return 0.3*np.exp(-((x - 2/3)**2) / (2 * 0.15**2))

def load_trained_model(data, layers, best_step):
    net = dde.nn.FNN(layers, "silu", "Glorot uniform", use_bias = True)
    model = dde.Model(data, net)
    # restore model
    model.compile("adam", lr=1e-3, decay = ("inverse time", 20000, 0.5), metrics=["l2 relative error"])
    model.compile("L-BFGS-B",  metrics=["l2 relative error"])
    model.restore(f"model/{best_step}.ckpt", verbose=1)
    return model

def fixed_point_iteration(model, dataset, dname, PATH, isplot=False):
    # new source function
    f_new = np.loadtxt(f"{dname}/f_new_grid.dat")[:, 1]
    S_new = np.loadtxt(f"{dname}/S_new_grid.dat")
    I_new = np.loadtxt(f"{dname}/I_new_grid.dat")
    x_0 = np.loadtxt(f"{PATH}/data_0.10/x_grid.dat")[:, 0]
    print(x_0.shape)
    
    # initial guess
    f_0 = np.loadtxt(f"{PATH}/data_0.10/data_0/f_0_grid.dat")[:, 1]
    S = np.loadtxt(f"{PATH}/data_0.10/data_0/S_0_grid.dat")
    I = np.loadtxt(f"{PATH}/data_0.10/data_0/I_0_grid.dat")
    S_0 = np.loadtxt(f"{PATH}/data_0.10/data_0/S_0_grid.dat")
    I_0 = np.loadtxt(f"{PATH}/data_0.10/data_0/I_0_grid.dat")
    
    errors_S = [dde.metrics.l2_relative_error(S_new, S)]
    errors_I = [dde.metrics.l2_relative_error(I_new, I)]
    Nx, Nt = I_new.shape
    print(errors_S, errors_I)

    ts = time.time()
    count = 0
    min_err_S = errors_S[-1]
    min_err_I = errors_I[-1]
    while errors_S[-1] > 1e-4 and errors_I[-1] > 1e-4 and count < 2000:
        inputs, So, Io = construct_more_data(Nx, Nt, f_new, S, I)
        u_outputs = model.predict(inputs)
        u_outputs = np.clip(u_outputs, 1e-3, None)
        S_outputs = u_outputs[:, 0:3]
        I_outputs = u_outputs[:, 3:6]
        
        S[2:Nx-2, 1:Nt] = S_outputs[:, 1].reshape(Nx-4, Nt-1)
        S[:,0] = S_0_func(x_0)
        S[1, 1:Nt] = S_outputs[:, 0].reshape(Nx-4, Nt-1)[0,:]
        S[-2, 1:Nt] = S_outputs[:, 2].reshape(Nx-4, Nt-1)[-1,:]
        S[0, 1:Nt] = S[1, 1:Nt]
        S[-1, 1:Nt] = S[-2, 1:Nt]  
        
        I[2:Nx-2, 1:Nt] = I_outputs[:, 1].reshape(Nx-4, Nt-1)
        I[:,0] = I_0_func(x_0)
        I[1, 1:Nt] = I_outputs[:, 0].reshape(Nx-4, Nt-1)[0,:]
        I[-2, 1:Nt] = I_outputs[:, 2].reshape(Nx-4, Nt-1)[-1,:]
        I[0, 1:Nt]   = I[1, 1:Nt]
        I[-1, 1:Nt]  = I[-2, 1:Nt]
        
        errors_S.append(dde.metrics.l2_relative_error(S_new,S))
        errors_I.append(dde.metrics.l2_relative_error(I_new,I))
        if count < 20000 and count % 100 == 0:
            print(count, errors_S[-1], errors_I[-1])
        count += 1
        min_err_S = min(errors_S[-1],min_err_S)
        min_err_I = min(errors_I[-1],min_err_I)
    print("Error of S after {} iterations:".format(count), errors_S[-1])
    print("Minimum error is ", min_err_S)
    print("Error of I after {} iterations:".format(count), errors_I[-1])
    print("Minimum error is ", min_err_I)
    print("One-shot took {} s.".format(time.time()-ts))
    # save the results
    np.savetxt(f"{dname}/S_FPI.dat", S)
    np.savetxt(f"{dname}/I_FPI.dat", I)
    np.savetxt(f"{dname}/err_FPI_S.dat",errors_S)
    np.savetxt(f"{dname}/err_FPI_I.dat",errors_I)
    # plot
    if isplot:
        plt.figure()
        plt.rcParams.update({'font.size': 20,"savefig.dpi": 200, "figure.figsize": (8, 6)})
        plt.subplots_adjust(left=0.20, right=0.9, top=0.9, bottom=0.15)
        plt.imshow(np.rot90(abs(S-S_new)), cmap = "rainbow", extent=(0,1,0,1), aspect='auto')
        plt.colorbar()
        plt.xlabel("x")
        plt.ylabel("t")
        plt.savefig(f"{dname}/res_FPI_S.png")
        plt.close()
        
        plt.figure()
        plt.rcParams.update({'font.size': 20,"savefig.dpi": 200, "figure.figsize": (8, 6)})
        plt.subplots_adjust(left=0.20, right=0.9, top=0.9, bottom=0.15)
        plt.imshow(np.rot90(abs(I-I_new)), cmap = "rainbow", extent=(0,1,0,1), aspect='auto')
        plt.colorbar()
        plt.xlabel("x")
        plt.ylabel("t")
        plt.savefig(f"{dname}/res_FPI_I.png")
        plt.close()
        plot_errs(errors_S, dname, "S")
        plot_errs(errors_I, dname, "I")
    return errors_S[-1], errors_I[-1]

def plot_errs(errors, d_num, vn):
    plt.figure()
    plt.rcParams.update({'font.size': 20,"savefig.dpi": 200, "figure.figsize": (8, 6)})
    plt.subplots_adjust(left=0.20, right=0.9, top=0.9, bottom=0.15)
    plt.semilogy(errors)
    plt.xlabel("#iterations")
    plt.ylabel("l2 error")
    plt.savefig(f"{d_num}/errors_{vn}.png")
    plt.close()
    return

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
    dataset_G, dataset = load_all_data(Mx, Mt, Nx, Nt, N_f, N_b, l, a, l_new, a_new, 
                                 f"{PATH}/data_{sigma}/data_0", gen, 
                                 correction = False, grid = True, isplot = False)
    model = load_trained_model(dataset_G, pre_layers, best_step)
    
    errs_S = []
    errs_I = []

    for i in range(num_func):
        print("Dataset {}".format(i))
        dname = f"{PATH}/data_{sigma}/data_{i}"
        err_S, err_I = fixed_point_iteration(model, dataset, dname, PATH, isplot=False)
        errs_S.append(err_S)
        errs_I.append(err_I)

    
    np.savetxt(os.path.join(f"{PATH}/data_{sigma}/", f"errs_FPI_S.dat"), errs_S)
    np.savetxt(os.path.join(f"{PATH}/data_{sigma}/", f"errs_FPI_I.dat"), errs_I)
    print("The average l2 error is ", sum(errs_S)/num_func, sum(errs_I)/num_func)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num", type=int, default=2) # Number of functions
    parser.add_argument("--sigma", type=str, default="0.10") # Amplitude in the GRF
    args = parser.parse_args()

    main(args.sigma, args.num)

