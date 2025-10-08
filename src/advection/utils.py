import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from CVC_solver import solve_CVC
from spaces import GRF

import argparse
import deepxde as dde
import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt
dde.backend.set_default_backend('tensorflow.compat.v1')
dde.config.set_default_float("float64")

def compute_numerical_solution(V, Nx, Nt):
    xmin, xmax = 0, 1
    tmin, tmax = 0, 1
    f = lambda x: x ** 2
    g = lambda t: np.sin(np.pi * t)

    x, t, u = solve_CVC(xmin, xmax, tmin, tmax, f, g, V, Nx, Nt)
    return x, t, u

def gen_data_GRF(M, Nx, Nt, l, a, dname, isplot):
    space = GRF(1, length_scale=l, N=M, interp="cubic")
    features = space.random(1)
    v_T = lambda x: a * np.ravel(space.eval_u(features, x))
    x, t, u_T =  compute_numerical_solution(v_T, M, M)
    f_T = np.tile(v_T(x)[:, None], (1, len(t)))

    f_T_grid = f_T[::round(M/Nx), ::round(M/Nt)]
    u_T_grid = u_T[::round(M/Nx), ::round(M/Nt)]
    x_grid = x[::round(M/Nx)]
    t_grid = t[::round(M/Nx)]

    np.savetxt(f"{dname}/x_grid.dat", np.rot90(np.tile(x_grid, (Nx, 1)), k = 3))
    np.savetxt(f"{dname}/t_grid.dat", np.tile(t_grid, (Nt, 1)))
    np.savetxt(f"{dname}/f_T_grid.dat", f_T_grid)
    np.savetxt(f"{dname}/u_T_grid.dat", u_T_grid)
    print("Generated f_T_grid and u_T_grid.")
    if isplot:
        plot_data("f_T_grid", "u_T_grid", dname)

    np.savetxt(f"{dname}/x.dat", np.rot90(np.tile(x, (M, 1)), k = 3))
    np.savetxt(f"{dname}/t.dat", np.tile(t, (M, 1)))
    np.savetxt(f"{dname}/f_T.dat", f_T)
    np.savetxt(f"{dname}/u_T.dat", u_T)

    print("Generated f_T and u_T.")
    if isplot:
        plot_data("f_T", "u_T", dname)
    return


def gen_test_data(M, Nx, Nt, dname, isplot):
    v_0 = lambda x: 0*x
    x, t, u_0 =  compute_numerical_solution(v_0, M, M)
    f_0 = np.tile(v_0(x)[:, None], (1, len(t)))

    np.savetxt(f"{dname}/f_0.dat", f_0)
    np.savetxt(f"{dname}/u_0.dat", u_0)
    if isplot:
        plot_data("f_0", "u_0", dname)

    interp = interpolate.RegularGridInterpolator((x, t), u_0, method='cubic')
    f_0_grid = f_0[::round(M/Nx), ::round(M/Nt)]
    u_0_grid = u_0[::round(M/Nx), ::round(M/Nt)]

    np.savetxt(f"{dname}/f_0_grid.dat", f_0_grid)
    np.savetxt(f"{dname}/u_0_grid.dat", u_0_grid)
    if isplot:
        plot_data("f_0_grid", "u_0_grid", dname)

    print("Generated f_0, u_0, f_0_grid, and u_0_grid.")
    return interp


def sample_points(geomtime, Nx, Nt, N_f, N_b):
    hx = 1/(Nx-1)
    ht = 1/(Nt-1)
    N_f = N_f - Nx - Nt + 1
    x_random = geomtime.random_points(N_f, random = "Hammersley")
    x_f = []
    for i in x_random:
        if 0 + hx <= i[0] and 0 + ht <= i[1]:
            x_f.append(i)
    print("Removed {} point(s).".format(len(x_random) - len(x_f)))
    if N_b == 0:
        x_random  = np.array(x_f)
    else:
        x_b = geomtime.random_boundary_points(N_b, random = "Hammersley")
        x_random = np.concatenate((x_f, x_b), axis = 0)
    return x_random

def gen_new_data_exact(M, Nx, Nt, a_new, l_new, dname, isplot):
    space = GRF(1, length_scale=l_new, N=1001, interp="cubic")
    features = space.random(1)
    vi = lambda x: a_new * space.eval_u(features, x[:, None])[0]
    xi, ti, ui =  compute_numerical_solution(vi, 1001, 1001)
    fi = np.tile(vi(xi)[:, None], (1, len(ti)))
    np.savetxt(f"{dname}/f_new_1001.dat", fi)
    np.savetxt(f"{dname}/u_new_1001.dat", ui)
    interp = interpolate.RegularGridInterpolator((xi, ti), ui, method="cubic")

    fi_grid = fi[::round(M/Nx), ::round(M/Nt)]
    ui_grid = ui[::round(M/Nx), ::round(M/Nt)]
    np.savetxt(f"{dname}/f_new_grid.dat", fi_grid)
    np.savetxt(f"{dname}/u_new_grid.dat", ui_grid)

    print("Generated f_new_grid and u_new_grid.")
    if isplot:
        plot_data("f_new_grid", "u_new_grid", dname)
    return interp, features, space

def gen_new_data_GRF(M, Nx, Nt, N_f, N_b, dname, a_new, l_new, isplot):
    hx = 1/(Nx-1)
    ht = 1/(Nt-1)
    geom = dde.geometry.Interval(0 + hx, 1)
    timedomain = dde.geometry.TimeDomain(0 + ht, 1)
    geomtime = dde.geometry.GeometryXTime(geom, timedomain)
    x_random = sample_points(geomtime, Nx, Nt, N_f, N_b)
    interp, features, space = gen_new_data_exact(M, Nx, Nt, a_new, l_new, dname, isplot)
    v_new = lambda x: a_new * space.eval_u(features, x_random[:, None])[0]
    f_new = v_new(x_random[:, 0]).reshape((-1, 1))
    u_new =  np.array([interp((i[0], i[1])) for i in x_random]).reshape((-1, 1))

    np.savetxt(f"{dname}/f_new.dat", np.concatenate((x_random, f_new), axis = 1))
    np.savetxt(f"{dname}/u_new.dat", np.concatenate((x_random, u_new), axis = 1))

    print("Generated f_new and u_new.")
    if isplot:
        plot_data("f_new", "u_new", dname)
    return

def gen_data_correction(interp, dname, isplot):
    x_random = np.loadtxt(f"{dname}/u_new.dat")[:, 0:2]
    v_0 = lambda x: 0*x
    f_0 = v_0(x_random[:, 0]).reshape((-1, 1))
    u_0 =  np.array([interp((i[1], i[0])) for i in x_random]).reshape((-1, 1))

    np.savetxt(f"{dname}/f_init.dat", np.concatenate((x_random, f_0), axis = 1))
    np.savetxt(f"{dname}/u_init.dat", np.concatenate((x_random, u_0), axis = 1))

    print("Generated f_init and u_init.")
    if isplot:
        plot_data("f_init", "u_init", dname)
    return

def construct_data(f, u):
    Nx, Nt = f.shape
    outputs = u[1:, 1:].reshape((-1, 1))
    inputs = np.hstack(( f[1:, 1:].reshape((-1, 1)), u[:-1, 1:].reshape((-1, 1)), u[:-1, :-1].reshape((-1, 1)), u[1:, :-1].reshape((-1, 1))))
    return np.array(inputs), np.array(outputs)

def construct_more_data(Nx, Nt, f, u):
    # f[i, j], u[i-1, j], u[i-1, j-1], u[i, j-1]
    M = len(f)
    xstep = int((M-1)/(Nx-1))
    tstep = int((M-1)/(Nt-1))
    outputs = u[xstep:, tstep:].reshape((-1, 1))
    inputs = np.hstack((f[xstep:, tstep:].reshape((-1, 1)), u[:-xstep, tstep:].reshape((-1, 1)), u[: -xstep, :-tstep].reshape((-1, 1)), u[xstep: , :-tstep].reshape((-1, 1))))
    return inputs, outputs

def plot_data(f_name, u_name, dname):
    f = np.loadtxt(f"{dname}/{f_name}.dat")
    u = np.loadtxt(f"{dname}/{u_name}.dat")
    f = np.rot90(f)
    u = np.rot90(u)
    print("Plot ({}, {})".format(f_name, u_name))
    plt.figure()
    plt.plot(f[0])
    plt.show()
    plt.figure()
    plt.rcParams.update({'font.size': 20,"savefig.dpi": 200, "figure.figsize": (8, 6)})
    plt.subplots_adjust(left=0.15, right=0.9, top=0.9, bottom=0.15)
    plt.imshow(f, cmap = "gnuplot",  origin='upper', extent=(0,1,0,1), aspect='auto')
    plt.colorbar()
    plt.xlabel("x")
    plt.ylabel("t")
    plt.savefig(f"{dname}/{f_name}.png")
    plt.show()

    plt.figure()
    plt.subplots_adjust(left=0.15, right=0.9, top=0.9, bottom=0.15)
    plt.imshow(u,  origin='upper', cmap = "gnuplot", extent=(0,1,0,1), aspect='auto')
    plt.colorbar()
    plt.xlabel("x")
    plt.ylabel("t")
    plt.savefig(f"{dname}/{u_name}.png")
    plt.show()

def load_all_data(M, Nx, Nt, N_f, N_b, l, a, l_new, a_new, dname, gen = False, correction = False, grid = False, isplot = False):
    if gen:
        print("Generate new dataset ... ")
        gen_data_GRF(M, Nx, Nt, l, a, dname, isplot)
        interp_u0 = gen_test_data(M, Nx, Nt, dname, isplot)
        gen_new_data_GRF(M, Nx, Nt, N_f, N_b, dname, a_new, l_new, isplot)
        gen_data_correction(interp_u0, dname, isplot)

    f_T = np.loadtxt(f"{dname}/f_T.dat")
    u_T = np.loadtxt(f"{dname}/u_T.dat")
    print(f"Loaded f_T {f_T.shape} and u_T {u_T.shape} for training the local solution operator.")
    d_T = construct_more_data(Nx, Nt, f_T, u_T)

    f_0 = np.loadtxt(f"{dname}/f_0_grid.dat")
    u_0 = np.loadtxt(f"{dname}/u_0_grid.dat")
    print(f"Loaded f_0_grid {f_0.shape} and u_0_grid {u_0.shape} for testing the local solution operator.")
    d_0 = construct_more_data(Nx, Nt, f_0, u_0)

    data_G = dde.data.DataSet(X_train=d_T[0], y_train=d_T[1], X_test=d_0[0], y_test=d_0[1])
    
    if not grid:
        x_train = np.loadtxt(f"{dname}/u_new.dat")[:, 0:2]
        x = np.loadtxt(f"{dname}/x_grid.dat").reshape((-1, 1))
        t = np.loadtxt(f"{dname}/t_grid.dat").reshape((-1, 1))
        x_test = np.concatenate((x, t), axis = 1)
        print(f"Loaded u_new {x_train.shape} and u_new_grid {x_test.shape} for x_train and x_test.")
        y_train = np.concatenate(([[0] * len(x_train)])).reshape((-1, 1))

        if correction:
            # For cLOINN-random
            u_new = np.loadtxt(f"{dname}/u_new_grid.dat").reshape((-1, 1))
            u_init = np.loadtxt(f"{dname}/u_0_grid.dat").reshape((-1, 1))
            y_test = u_new - u_init
            print("Dataset generated for cLOINN-random (x_train, y_train, x_test, y_test).")
        else:
            # For FPI and LOINN-random
            y_test = np.loadtxt(f"{dname}/u_new_grid.dat").reshape((-1, 1))
            print("Dataset generated for LOINN-random (x_train, y_train, x_test, y_test).")
    
    else:
        x = np.loadtxt(f"{dname}/x_grid.dat").reshape((-1, 1))
        t = np.loadtxt(f"{dname}/t_grid.dat").reshape((-1, 1))
        x_train = np.concatenate((x, t), axis = 1)
        x_test = x_train
        print(f"Loaded u_new_grid {x_train.shape} for x_train and x_test.")
        y_train = np.concatenate(([[0] * len(x_train)])).reshape((-1, 1))

        if correction:
            # For cLOINN-grid
            u_new = np.loadtxt(f"{dname}/u_new_grid.dat").reshape((-1, 1))
            u_init = np.loadtxt(f"{dname}/u_0_grid.dat").reshape((-1, 1))
            y_test = u_new - u_init
            print("Dataset generated for cLOINN-grid (x_train, y_train, x_test, y_test).")
        else:
            # For FPI and LOINN-grid
            y_test = np.loadtxt(f"{dname}/u_new_grid.dat").reshape((-1, 1))
            print("Dataset generated for FPI/LOINN-grid (x_train, y_train, x_test, y_test).")
 
    data = dde.data.DataSet(X_train=x_train, y_train=y_train, X_test=x_test, y_test=y_test)
    return data_G, data