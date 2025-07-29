import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import time
import argparse
import deepxde as dde
import numpy as np
import matplotlib.pyplot as plt
dde.config.set_default_float("float64")
from utils import load_all_data

def construct_data(Nx, Ny, f, u, x, y):
    f = f.reshape((Nx, Ny))
    u = u.reshape((Nx, Ny))
    # u[i-1, j], u[i, j-1], u[i+1, j], u[i, j+1], f[i, j]
    N = len(f)
    xstep = 1
    ystep = 1
    outputs = u[ystep: -ystep, xstep:-xstep].reshape((-1, 1))
    x = x.reshape((Nx, Ny))[ystep: -ystep, xstep:-xstep].reshape((-1, 1))
    y = y.reshape((Nx, Ny))[ystep: -ystep, xstep:-xstep].reshape((-1, 1))
    naninds_out = np.argwhere(np.isnan(outputs))[:, 0]
    inputs = np.hstack((u[ystep:-ystep, :-2*xstep].reshape((-1, 1)), u[:-2*ystep, xstep:-xstep].reshape((-1, 1)),u[ystep: -ystep, 2*xstep:].reshape((-1, 1)), u[2*ystep: , xstep:-xstep].reshape((-1, 1)),f[ystep: -ystep, xstep:-xstep].reshape((-1, 1))))
    naninds_in = np.argwhere(np.isnan(inputs))[:, 0]
    inds = [i for i in range(len(outputs)) if i not in naninds_out and i not in naninds_in]
    inputs = inputs[inds]
    outputs = outputs[inds]
    return inputs, outputs

def load_trained_model(data, layers, best_step):
    net = dde.nn.FNN(layers, "tanh", "LeCun normal", use_bias = True, regularization=['l2', 1e-8])
    model = dde.Model(data, net)
    model.compile("adam", lr=1e-3, decay = ("inverse time", 20000, 0.5), metrics=["l2 relative error"])
    model.compile("L-BFGS-B",  metrics=["l2 relative error"])
    model.restore("model/model.ckpt-" + best_step + ".ckpt", verbose=1)
    return model

def plot_errs(errors, dname):
    plt.figure()
    plt.semilogy(errors)
    plt.xlabel("#iterations")
    plt.ylabel("l2 error")
    plt.savefig(f"{dname}/errors.png")
    plt.show()
    return

def fixed_point_iteration(Nx, Ny, model, dname, PATH, isplot=False):
    data_new_grid = np.loadtxt(f"{dname}/f_new_grid.txt",skiprows=9)
    u_newg = data_new_grid[:,4].reshape((-1, 1))
    f_newg = data_new_grid[:,5].reshape((-1, 1))
    x_newg = data_new_grid[:,2].reshape((-1, 1))
    y_newg = data_new_grid[:,3].reshape((-1, 1))

    u = np.loadtxt(f"{PATH}/data_0.05/data_0/f_0_grid.txt",skiprows=9)[:,4].reshape((-1, 1))
    hx, hy = 1/Nx, 1/Ny
    inds_nan = [i for i in range(len(data_new_grid)) if i in np.argwhere(np.isnan(data_new_grid))[:, 0]] # nan 1256
    inds_all = [i for i in range(len(data_new_grid)) if i not in inds_nan]
    inds_c = [i for i in range(len(data_new_grid)) if ((data_new_grid[i, 2] - 0.5)**2 + (data_new_grid[i, 3] - 0.5)**2 <= 0.2**2) and i not in np.argwhere(np.isnan(data_new_grid))[:, 0]] # points on the circle 8
    inds_bc = [i for i in range(len(data_new_grid)) if ((data_new_grid[i, 0] == 0.995) or (data_new_grid[i, 0] == 0.005) or (data_new_grid[i, 1] == 0.005) or (data_new_grid[i, 1] == 0.995))] #10000-9604 = 396
    inds_bcc = [i for i in inds_all if i not in inds_c and (
              not (((data_new_grid[i, 2] - hx - 0.5)**2 + (data_new_grid[i, 3] - 0.5)**2 > 0.1995**2)
              and ((data_new_grid[i, 2] - 0.5)**2 + (data_new_grid[i, 3] - hy - 0.5)**2 > 0.1995**2)
              and ((data_new_grid[i, 2] + hx - 0.5)**2 + (data_new_grid[i, 3] - 0.5)**2 > 0.1995**2)
              and ((data_new_grid[i, 2] - 0.5)**2 + (data_new_grid[i, 3] + hy - 0.5)**2 > 0.1995**2)))] # 108
    # 10000 - 396 = 9604; 9604 - 1256 = 8348; 8348 - 108 - 8 = 8232; 8232
    inds = [i for i in range(len(data_new_grid)) if i not in inds_nan and (i not in inds_bc) and (i not in inds_bcc) and (i not in inds_c)]  #8232
    print(len(inds_nan), len(inds_all), len(inds_c), len(inds_bc), len(inds_bcc)) #1256 8744 8 396 108
    errors = [dde.metrics.l2_relative_error(u_newg[inds_all], u[inds_all])]
    if isplot:
        plt.rcParams.update({'font.size': 20,"savefig.dpi": 200, "figure.figsize": (8, 6)})
        plt.subplots_adjust(left=0.20, right=0.9, top=0.9, bottom=0.15)
        plt.imshow(np.rot90(u.reshape((100,100))), cmap = "gnuplot", extent=(0,1,0,1), aspect='auto')
        plt.colorbar()
        plt.scatter(x_newg[inds_nan], y_newg[inds_nan], s = 8, c = "red")
        plt.scatter(x_newg[inds_c], y_newg[inds_c], s = 8, c = "green")
        plt.scatter(x_newg[inds_bcc], y_newg[inds_bcc], s = 8, c = "blue")
        plt.scatter(x_newg[inds_bc], y_newg[inds_bc], s = 8, c = "purple")
        plt.scatter(x_newg[inds], y_newg[inds], s = 2, c = "yellow")
        plt.scatter(x_newg[inds_all], y_newg[inds_all], s = 0.1, c = "white")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.show()
    

    ts = time.time()
    count = 0
    min_err = errors[-1]
    while errors[-1] > 1e-4 and count < 10000:
        inputs,_ = construct_data(Nx, Ny, f_newg, u, x_newg, y_newg)
        u_outputs = model.predict(inputs)
        k = 0
        for i in inds:
            u[i] = u_outputs[k]
            k +=1

        for i in inds_c:
            u[i] = 0
        
        errors.append(dde.metrics.l2_relative_error(u_newg[inds_all],u[inds_all]))
        count += 1
        if count % 200 == 0:
            print(count,errors[-1])
        min_err = min(errors[-1],min_err)

    print("Error after {} iterations:".format(count), errors[-1])
    print("Minimum error is ", min_err)
    print("One-shot took {} s.".format(time.time()-ts))
    np.savetxt(f"{dname}/u_FPI.dat",u[inds_all])
    np.savetxt(f"{dname}/err_FPI.dat",errors)
    plot_errs(errors, dname)
    
    if isplot:
        plt.rcParams.update({'font.size': 20,"savefig.dpi": 200, "figure.figsize": (8, 6)})
        plt.subplots_adjust(left=0.20, right=0.9, top=0.9, bottom=0.15)
        plt.imshow(np.rot90(abs(u.reshape((Nx,Ny)) - u_newg.reshape((Nx,Ny)))), cmap = "gnuplot", extent=(0,1,0,1), aspect='auto')
        plt.colorbar()
        plt.xlabel("x")
        plt.ylabel("y")
        plt.show()
    return errors[-1]


def main(sigma, num_func, parent_dir = "../../data/", gen = False):
    Nx = Ny = 100
    l, a = 0.05, 0.10
    l_new, a_new = 0.1, float(sigma)

    # Create folders for the datasets
    new_dir = "data_nonlinear_poisson_circle"
    PATH = os.path.join(parent_dir, new_dir)
    
    # Load model
    best_step = "150994"
    pre_layers = [5, 64, 1]
    dataset_G, dataset = load_all_data(Nx, Ny, f"{PATH}/data_{sigma}/data_0", PATH,
                                       correction=False, isplot=False)
    model = load_trained_model(dataset_G, pre_layers, best_step)
    
    errs = []
    for i in range(num_func):
        print("Dataset {}".format(i))
        dname = f"{PATH}/data_{sigma}/data_{i}"
        err = fixed_point_iteration(Nx, Ny, model, dname, PATH, isplot=True)
        errs.append(err)

    print(errs)
    np.savetxt(os.path.join(f"{dname}", "errs_FPI.dat"), errs)
    print("The average l2 error is ", sum(errs)/num_func)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num", type=int, default=3) # Number of functions
    parser.add_argument("--sigma", type=str, default="0.05") # Amplitude in the GRF
    args = parser.parse_args()
    main(args.sigma, args.num)

