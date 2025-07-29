import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from SIR_solver import solver
from spaces import GRF

import argparse
import deepxde as dde
import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt
dde.backend.set_default_backend('tensorflow.compat.v1')
dde.config.set_default_float("float64")

T = 10.0
f0 = lambda x: np.exp(-((x - 0.5)**2) / (2 * 0.2**2)) #1 + 0.5*np.sin(x * np.pi *2)

def compute_numerical_solution(f, Mx, Mt):
    L = 1.0
    dt = T/(Mt-1)
    #assert dt == 0.15, f"dt={dt} is not 0.03"
    gamma = 0.2
    beta = 0.8
    dx_2 = 0.001
    x, t, S, I  = solver(Mx, dt, L, T, beta, gamma, dx_2, f)
    return x, t, S, I

def gen_data_GRF(Mx, Mt, Nx, Nt, l, a, dname, isplot = False):
    # generate data for training the local solution operator
    space = GRF(1, length_scale=l, N=Mx, interp="cubic")
    features = space.random(1)
    f_T = lambda x: f0(x) + np.ravel(a * np.log(1+np.exp(space.eval_u(features, x))))
    x, t, S_T, I_T =  compute_numerical_solution(f_T, Mx, Mt)
    assert t.shape[0] == Mt, "Incorrect t dimension."
    print(x.shape, t.shape, S_T.shape, I_T.shape, x[:, None].shape)
    f_T = f_T(x)[:, None]
    print(np.mean(f_T), np.min(f_T), np.max(f_T))
    assert min(f_T) > 0, "Negative f values."
    
    f_T_grid = f_T[::round(Mx/Nx)]
    S_T_grid = S_T[::round(Mx/Nx), ::round(Mt/Nt)]
    I_T_grid = I_T[::round(Mx/Nx), ::round(Mt/Nt)]
    x_grid = x[::round(Mx/Nx)]
    t_grid = t[::round(Mt/Nt)]
    
    # Generate the single data pair for training local solution operators
    # Prepare data and save using consistent f-string formatting
    np.savetxt(f"{dname}/x_grid.dat", np.rot90(np.tile(x_grid, (Nt, 1)), k=3))
    np.savetxt(f"{dname}/t_grid.dat", np.tile(t_grid, (Nx, 1)))
    np.savetxt(f"{dname}/f_T_grid.dat", np.column_stack((x_grid, f_T_grid)))
    np.savetxt(f"{dname}/S_T_grid.dat", S_T_grid)
    np.savetxt(f"{dname}/I_T_grid.dat", I_T_grid)

    # np.savetxt(f"{dname}/x.dat", np.rot90(np.tile(x, (M, 1)), k = 3))
    # np.savetxt(f"{dname}/t.dat", np.tile(t, (M, 1)))
    np.savetxt(f"{dname}/f_T.dat", np.column_stack((x, f_T)))
    np.savetxt(f"{dname}/S_T.dat", S_T)
    np.savetxt(f"{dname}/I_T.dat", I_T)

    print("Generated f_T, u_T, f_T_grid and S_T_grid.")
    if isplot:
        plot_data("f_T_grid", "S_T_grid", dname)
        plot_data("f_T_grid", "I_T_grid", dname)
        plot_data("f_T", "S_T", dname)
        plot_data("f_T", "I_T", dname)
    return

def gen_test_data(Mx, Mt, Nx, Nt, dname, isplot = False):
    x, t, S_0, I_0 =  compute_numerical_solution(f0, Mx, Mt)
    f_0 = f0(x)[:, None]

    interp_S = interpolate.RegularGridInterpolator((x, t), S_0, method="cubic")
    interp_I = interpolate.RegularGridInterpolator((x, t), I_0, method="cubic")
    f_0_grid = f_0[::round(Mx/Nx)]
    S_0_grid = S_0[::round(Mx/Nx), ::round(Mt/Nt)]
    I_0_grid = I_0[::round(Mx/Nx), ::round(Mt/Nt)]
    x_grid = x[::round(Mx/Nx)]

    np.savetxt(f"{dname}/f_0_grid.dat", np.column_stack((x_grid, f_0_grid)))
    np.savetxt(f"{dname}/S_0_grid.dat", S_0_grid)
    np.savetxt(f"{dname}/I_0_grid.dat", I_0_grid)
    if isplot:
        plot_data("f_0_grid", "S_0_grid", dname)
        plot_data("f_0_grid", "I_0_grid", dname)
    # np.savetxt(f"{dname}/f_0.dat", f_0)
    # np.savetxt(f"{dname}/S_0.dat", S_0)
    # np.savetxt(f"{dname}/I_0.dat", I_0)
    # xyz
    print("Generated f_0_grid, S_0_grid and I_0_grid.")
    return interp_S, interp_I

def sample_points(geomtime, Nx, Nt, N_f, N_b):
    hx = 1/(Nx-1)
    ht = T/(Nt-1)
    N_f = N_f - N_b
    x_random = geomtime.random_points(N_f, random = "Hammersley")
    x_f = []
    for i in x_random:
        if 0 + 2*hx <= i[0] <= 1 - 2*hx and 0 + ht <= i[1]:
            x_f.append(i)
    print("Removed {} point(s).".format(len(x_random) - len(x_f)))
    if N_b == 0:
        x_random  = np.array(x_f)
    else:
        geom = dde.geometry.Interval(0, 1)
        timedomain = dde.geometry.TimeDomain(0, T)
        geomtime = dde.geometry.GeometryXTime(geom, timedomain)
        x_b = geomtime.random_boundary_points(N_b, random = "Hammersley")
        x_random = np.concatenate((x_f, x_b), axis = 0)
    return x_random

def gen_new_data_exact(Mx, Mt, Nx, Nt, a_new, l_new, dname, isplot):
    space = GRF(1, length_scale=l_new, N=Mx, interp="cubic")
    features = space.random(1)
    #shift = max(np.ravel(a_new * space.eval_u(features, np.linspace(0, 1, Mx))))
    fi = lambda x: f0(x) + np.ravel(a_new * np.log(1+np.exp(space.eval_u(features, x))))
    xi, ti, Si, Ii = compute_numerical_solution(fi, Mx, Mt)
    fi = fi(xi)[:, None]
    assert ti.shape[0] == Mt, f"Incorrect t dimension ({ti.shape[0]})."
    assert min(fi) > 0, "Negative f values."

    interp_S = interpolate.RegularGridInterpolator((xi, ti), Si, method="cubic")
    interp_I = interpolate.RegularGridInterpolator((xi, ti), Ii, method="cubic")

    xi_grid = xi[::round(Mx/Nx)]
    fi_grid = fi[::round(Mx/Nx)]
    Si_grid = Si[::round(Mx/Nx), ::round(Mt/Nt)]
    Ii_grid = Ii[::round(Mx/Nx), ::round(Mt/Nt)]

    np.savetxt(f"{dname}/f_new_grid.dat", np.column_stack((xi_grid, fi_grid)))
    np.savetxt(f"{dname}/S_new_grid.dat", Si_grid)
    np.savetxt(f"{dname}/I_new_grid.dat", Ii_grid)
    if isplot:
        plot_data("f_new_grid", "S_new_grid", dname)
        plot_data("f_new_grid", "I_new_grid", dname)
    print("Generated f_new_grid, S_new_grid and I_new_grid.")
    return interp_S, interp_I, features, space

def gen_new_data_GRF(Mx, Mt, Nx, Nt, N_f, N_b, dname, a_new, l_new, isplot):
    hx = 1/(Nx-1)
    ht = T/(Nt-1)
    geom = dde.geometry.Interval(0 + 2*hx, 1 - 2*hx)
    timedomain = dde.geometry.TimeDomain(0 + ht, T)
    geomtime = dde.geometry.GeometryXTime(geom, timedomain)
    x_random = sample_points(geomtime, Nx, Nt, N_f, N_b)
    print(f"Sampled {x_random.shape} random points.")
    
    interp_S, interp_I, feature, space = gen_new_data_exact(Mx, Mt, Nx, Nt, a_new, l_new, dname, isplot)
    f_new = lambda x: f0(x) + np.ravel(a_new * np.log(1+np.exp(space.eval_u(feature, x))))
    S_new =  np.array([interp_S((i[0], i[1])) for i in x_random]).reshape((-1, 1))
    I_new =  np.array([interp_I((i[0], i[1])) for i in x_random]).reshape((-1, 1))
    f_new = f_new(x_random[:, 0])[:, None]
    assert min(f_new) > 0, "Negative f values."
    np.savetxt(f"{dname}/f_new.dat", np.column_stack((x_random, f_new)))
    np.savetxt(f"{dname}/S_new.dat", np.column_stack((x_random, S_new)))
    np.savetxt(f"{dname}/I_new.dat", np.column_stack((x_random, I_new)))
    if isplot:
        scatter_plot_data("f_new", "S_new", dname)
        scatter_plot_data("f_new", "I_new", dname)
    print("Generated f_new, S_new and I_new.")
    return

def gen_data_correction(interp_S, interp_I, dname, isplot):
    x_random = np.loadtxt(f"{dname}/S_new.dat")[:, 0:2]
    f_0 = f0(x_random[:,0])[:, None]
    S_0 =  np.array([interp_S((i[0], i[1])) for i in x_random]).reshape((-1, 1))
    I_0 =  np.array([interp_I((i[0], i[1])) for i in x_random]).reshape((-1, 1))

    np.savetxt(f"{dname}/f_init.dat", np.column_stack((x_random, f_0)))
    np.savetxt(f"{dname}/S_init.dat", np.column_stack((x_random, S_0)))
    np.savetxt(f"{dname}/I_init.dat", np.column_stack((x_random, I_0)))
    if isplot:
        scatter_plot_data("f_init", "S_init", dname)
        scatter_plot_data("f_init", "I_init", dname)
        print("Generated f_init, S_init and I_init.")
    return

def scatter_plot_data(fname, uname, dname):
    x = np.loadtxt(f"{dname}/{fname}.dat")[:,0]
    t = np.loadtxt(f"{dname}/{fname}.dat")[:,1]
    f = np.loadtxt(f"{dname}/{fname}.dat")[:,-1]
    u = np.loadtxt(f"{dname}/{uname}.dat")[:,-1]
    print("({}, {})".format(fname, uname))
    # print(x, f)

    plt.figure()
    plt.subplots_adjust(left=0.15, right=0.9, top=0.9, bottom=0.15)
    plt.scatter(x, f)
    plt.xlabel("x")
    plt.ylabel(fname)
    plt.savefig(f"{dname}/{fname}.png")
    plt.close()
    
    plt.figure()
    plt.rcParams.update({'font.size': 20,"savefig.dpi": 200, "figure.figsize": (8, 6)})
    plt.subplots_adjust(left=0.15, right=0.9, top=0.9, bottom=0.15)
    plt.scatter(x, t, c = u, cmap = "rainbow", s=0.5)
    plt.colorbar()
    plt.xlabel("x")
    plt.ylabel("t")
    plt.savefig(f"{dname}/{uname}.png")
    plt.close()

def plot_data(fname, uname, dname):
    x = np.loadtxt(f"{dname}/{fname}.dat")[:, 0]
    f = np.loadtxt(f"{dname}/{fname}.dat")[:, 1]
    u = np.loadtxt(f"{dname}/{uname}.dat")
    u = np.rot90(u)
    print("({}, {})".format(fname, uname))
    plt.figure()
    plt.subplots_adjust(left=0.15, right=0.9, top=0.9, bottom=0.15)
    plt.plot(x, f)
    plt.xlabel("x")
    plt.ylabel(fname)
    plt.savefig(f"{dname}/{fname}.png")
    plt.close()

    plt.figure()
    plt.subplots_adjust(left=0.15, right=0.9, top=0.9, bottom=0.15)
    plt.imshow(u,  origin='upper', cmap = "rainbow", extent=(0,1,0,1), aspect='auto')
    plt.colorbar()
    plt.xlabel("x")
    plt.ylabel("t")
    plt.title(uname)
    plt.savefig(f"{dname}/{uname}.png")
    plt.close()

def construct_more_data(Nx, Nt, f, S, I):
    # u[i-1, j], u[i, j-1], u[i+1, j], f[i, j]
    Mx, Mt = S.shape
    x = np.linspace(0, 1, Mx)
    t = np.linspace(0, 30, Mt)
    x = np.rot90(np.tile(x, (Mt, 1)), k=3)
    t = np.tile(t, (Mx, 1))
    xstep = int((Mx-1)/(Nx-1))
    tstep = int((Mt-1)/(Nt-1))
    outputs_S =  np.hstack((S[xstep:-3*xstep, tstep:].reshape((-1, 1)),
                            S[2*xstep: -2*xstep, tstep:].reshape((-1, 1)),
                            S[3*xstep:-xstep, tstep:].reshape((-1, 1))))
    outputs_I = np.hstack((I[xstep:-3*xstep, tstep:].reshape((-1, 1)),
                            I[2*xstep: -2*xstep, tstep:].reshape((-1, 1)),
                            I[3*xstep:-xstep, tstep:].reshape((-1, 1))))
    f_values = np.tile(f[2*xstep: -2*xstep][:, None], (1, Mt-tstep)).reshape((-1, 1))
    inputs = np.hstack((
                        S[xstep:-3*xstep, tstep:].reshape((-1, 1)), 
                        S[:-4*xstep, tstep:].reshape((-1, 1)), 
                        S[3*xstep:-xstep, tstep:].reshape((-1, 1)), 
                        S[4*xstep:, tstep:].reshape((-1, 1)), 
                        
                        S[xstep:-3*xstep, :-tstep].reshape((-1, 1)), 
                        S[:-4*xstep, :-tstep].reshape((-1, 1)), 
                        S[2*xstep: -2*xstep, :-tstep].reshape((-1, 1)), 
                        S[3*xstep:-xstep, :-tstep].reshape((-1, 1)), 
                        S[4*xstep:, :-tstep].reshape((-1, 1)), 
                        
                        I[xstep:-3*xstep, tstep:].reshape((-1, 1)), 
                        I[:-4*xstep, tstep:].reshape((-1, 1)), 
                        I[3*xstep:-xstep, tstep:].reshape((-1, 1)), 
                        I[4*xstep:, tstep:].reshape((-1, 1)), 
                        
                        I[xstep:-3*xstep, :-tstep].reshape((-1, 1)), 
                        I[:-4*xstep, :-tstep].reshape((-1, 1)), 
                        I[2*xstep: -2*xstep, :-tstep].reshape((-1, 1)), 
                        I[3*xstep:-xstep, :-tstep].reshape((-1, 1)), 
                        I[4*xstep:, :-tstep].reshape((-1, 1)), 
                        np.tile(f[xstep:-3*xstep][:, None], (1, Mt-tstep)).reshape((-1, 1)),
                        f_values,
                        np.tile(f[3*xstep:-xstep][:, None], (1, Mt-tstep)).reshape((-1, 1)),
                        ))
    return inputs, outputs_S, outputs_I

def load_all_data(Mx, Mt, Nx, Nt, N_f, N_b, l, a, l_new, a_new, dname, 
                  gen = False, correction = False, grid = False, isplot = False):
    if gen:
        print("Generate new datasets ... ")
        gen_data_GRF(Mx, Mt, Nx, Nt, l, a, dname, isplot)
        interp_S0, interp_I0 = gen_test_data(Mx, Mt, Nx, Nt, dname, isplot)
        gen_new_data_GRF(Mx, Mt, Nx, Nt, N_f, N_b, dname, a_new, l_new, isplot)
        gen_data_correction(interp_S0, interp_I0, dname, isplot)

    parent_dir = "../../data/"
    new_dir = "data_SIR"
    PATH = os.path.join(parent_dir, new_dir)
    f_T = np.loadtxt(f"{PATH}/data_{a}_{l}/data_0/f_T.dat")[:,1]
    S_T = np.loadtxt(f"{PATH}/data_{a}_{l}/data_0/S_T.dat")
    I_T = np.loadtxt(f"{PATH}/data_{a}_{l}/data_0/I_T.dat")
    print(f"Loaded f_T {f_T.shape}, S_T {S_T.shape}, and I_T {I_T.shape} for training the local solution operator.")
    d_T = construct_more_data(Nx, Nt, f_T, S_T, I_T)

    f_0 = np.loadtxt(f"{PATH}/data_0.10/data_0/f_0_grid.dat")[:,1]
    S_0 = np.loadtxt(f"{PATH}/data_0.10/data_0/S_0_grid.dat")
    I_0 = np.loadtxt(f"{PATH}/data_0.10/data_0/I_0_grid.dat")
    print(f"Loaded f_0_grid {f_0.shape} S_0_grid {S_0.shape}, and I_0_grid {I_0.shape} for testing the local solution operator.")
    d_0 = construct_more_data(Nx, Nt, f_0, S_0, I_0)

    data_G = dde.data.DataSet(
        X_train=d_T[0],
        y_train=np.hstack((d_T[1], d_T[2])),
        X_test=d_0[0],
        y_test=np.hstack((d_0[1], d_0[2])),
    )

    if not grid:
        x_train = np.loadtxt(f"{dname}/S_new.dat")[:, 0:2]
        x = np.loadtxt(f"{PATH}/data_0.10/x_grid.dat").reshape((-1, 1))
        t = np.loadtxt(f"{PATH}/data_0.10/t_grid.dat").reshape((-1, 1))
        x_test = np.concatenate((x, t), axis = 1)
        print(f"Loaded x_new {x_train.shape} and x_new_grid {x_test.shape} for x_train and x_test.")
        y_train = np.concatenate(([[0] * len(x_train)])).reshape((-1, 1))

        if correction:
            # For cLOINN-random
            S_new = np.loadtxt(f"{dname}/S_new_grid.dat").reshape((-1, 1))
            I_new = np.loadtxt(f"{dname}/I_new_grid.dat").reshape((-1, 1))
            S_init = np.loadtxt(f"{PATH}/data_0.10/data_0/S_0_grid.dat").reshape((-1, 1))
            I_init = np.loadtxt(f"{PATH}/data_0.10/data_0/I_0_grid.dat").reshape((-1, 1))
            S_test = S_new - S_init
            I_test = I_new - I_init
            print("Dataset generated for cLOINN-random (x_train, y_train, x_test, y_test).")
        else:
            print("Dataset generated for LOINN-random (x_train, y_train, x_test, y_test).")
            pass
    
    else:
        x = np.loadtxt(f"{PATH}/data_0.10/x_grid.dat").reshape((-1, 1))
        t = np.loadtxt(f"{PATH}/data_0.10/t_grid.dat").reshape((-1, 1))
        x_train = np.concatenate((x, t), axis = 1)
        x_test = x_train
        print(f"Loaded x_grid {x_train.shape} for x_train and x_test.")
        y_train = np.concatenate(([[0] * len(x_train)])).reshape((-1, 1))

        if correction:
            # For cLOINN-grid
            print("Dataset generated for cLOINN-grid (x_train, y_train, x_test, y_test).")
            pass
        else:
            # For FPI and LOINN-grid
            S_test = np.loadtxt(f"{dname}/S_new_grid.dat").reshape((-1, 1))
            I_test = np.loadtxt(f"{dname}/I_new_grid.dat").reshape((-1, 1))
            print("Dataset generated for FPI/LOINN-grid (x_train, y_train, x_test, y_test).")

    data = dde.data.DataSet(
        X_train=x_train, y_train=np.hstack((y_train, y_train)), 
        X_test=x_test, y_test=np.hstack((S_test, I_test)))
    return data_G, data