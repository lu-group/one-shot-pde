import deepxde as dde
import numpy as np
import matplotlib.pyplot as plt


def plot_data(f_name, u_name, d_num):
    f = np.loadtxt("data{}/{}.dat".format(d_num,f_name))
    u = np.loadtxt("data{}/{}.dat".format(d_num, u_name))
    f = np.rot90(f)
    u = np.rot90(u)
    print("({}, {})".format(f_name, u_name))
    plt.figure()
    plt.rcParams.update({'font.size': 20,"savefig.dpi": 200, "figure.figsize": (8, 6)})
    plt.subplots_adjust(left=0.15, right=0.9, top=0.9, bottom=0.15)
    plt.imshow(f, cmap = "gnuplot",  origin='upper', extent=(0,1,0,1), aspect='auto')
    plt.colorbar()
    plt.xlabel("x")
    plt.ylabel("t")
    #plt.savefig("data{}/{}.png".format(d_num,f_name))
    plt.show()

    plt.figure()
    plt.subplots_adjust(left=0.15, right=0.9, top=0.9, bottom=0.15)
    plt.imshow(u,  origin='upper', cmap = "gnuplot", extent=(0,1,0,1), aspect='auto')
    plt.colorbar()
    plt.xlabel("x")
    plt.ylabel("t")
    #plt.savefig("data{}/{}.png".format(d_num,u_name))
    plt.show()

def construct_more_data(Nx, Ny, f, u):
    # u[i-1, j], u[i, j-1], u[i+1, j], u[i, j+1], f[i, j]
    M = len(f) # 1001
    xstep = int((M-1)/(Nx-1))
    tstep = int((M-1)/(Ny-1))
    outputs = u[xstep: -xstep, tstep:-tstep].reshape((-1, 1))
    inputs = np.hstack((u[:-2*xstep, tstep:-tstep].reshape((-1, 1)), u[xstep: -xstep, :-2*tstep].reshape((-1, 1)), u[2*xstep: , tstep:-tstep].reshape((-1, 1)), u[xstep: -xstep, 2*tstep:].reshape((-1, 1)),f[xstep: -xstep, tstep:-tstep].reshape((-1, 1))))
    return inputs, outputs

def construct_more_data1(Nx, Ny, f, u):
    M = len(f) # 1001
    xstep = int((M-1)/(Nx-1))
    tstep = int((M-1)/(Ny-1))
    outputs = u[xstep: -xstep, tstep:-tstep].reshape((-1, 1))
    inputs = np.hstack((u[:-2*xstep, tstep:-tstep].reshape((-1, 1)), u[xstep: -xstep, :-2*tstep].reshape((-1, 1)), u[2*xstep: , tstep:-tstep].reshape((-1, 1)), u[xstep: -xstep, 2*tstep:].reshape((-1, 1)),u[:-2*xstep, :-2*tstep].reshape((-1, 1)),u[:-2*xstep, 2*tstep:].reshape((-1, 1)),u[2*xstep: , :-2*tstep].reshape((-1, 1)), u[2*xstep: , 2*tstep:].reshape((-1, 1)), f[xstep: -xstep, tstep:-tstep].reshape((-1, 1))))
    return inputs, outputs

def gen_all_data(M, Nx, Ny, N_f, N_b, l, a, l_new, a_new, gen, d_num, correction = False):
    if gen:
        print("Generate new dataset ... ")
    
    #plot_data("f_T", "u_T", d_num)
    #plot_data("f_0", "u_0", d_num)
    #plot_data("f_new_grid", "u_new_grid", d_num)

    # training data
    f_T = np.loadtxt("data{}/f_T.dat".format(d_num))
    u_T = np.loadtxt("data{}/u_T.dat".format(d_num))
    d_T = construct_more_data(Nx, Ny, f_T, u_T)
    # test data
    f_0 = np.loadtxt("data{}/f_0.dat".format(d_num))
    u_0 = np.loadtxt("data{}/u_0.dat".format(d_num))
    d_0 = construct_more_data(Nx, Ny, f_0, u_0)

    # For the 2nd stage
    x_train = np.loadtxt("data{}/u_new.dat".format(d_num))[:, 0:2]
    y_train = np.concatenate(([[0] * len(x_train)])).reshape((-1, 1))
    x = np.loadtxt("data{}/x_grid.dat".format(d_num)).reshape((-1, 1))
    t = np.loadtxt("data{}/y_grid.dat".format(d_num)).reshape((-1, 1))
    x_test = np.concatenate((x, t), axis = 1)

    if correction:
        u_new = np.loadtxt("data{}/u_new_grid.dat".format(d_num)).reshape((-1, 1))
        u_init = np.loadtxt("data{}/u_0_grid.dat".format(d_num)).reshape((-1, 1))
        y_test = u_new - u_init
    else:
        y_test = np.loadtxt("data{}/u_new_grid.dat".format(d_num)).reshape((-1, 1))
    data_G = dde.data.DataSet(X_train=d_T[0], y_train=d_T[1], X_test=d_0[0], y_test=d_0[1])
    data = dde.data.DataSet(X_train=x_train, y_train=y_train, X_test=x_test, y_test=y_test)
    return data_G, data

