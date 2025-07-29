import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from ADR_solver import solve_ADR
from spaces import GRF

import argparse
import deepxde as dde
import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt
dde.backend.set_default_backend('tensorflow.compat.v1')
dde.config.set_default_float("float64")

def compute_numerical_solution(f, Nx, Nt):
    xmin, xmax = 0, 1
    tmin, tmax = 0, 1
    k = lambda x: 0.01 * np.ones_like(x)
    v = lambda x: np.zeros_like(x)
    g = lambda u: np.zeros_like(u)
    dg = lambda u: np.zeros_like(u)
    u0 = lambda x: np.zeros_like(x)

    x, t, u = solve_ADR(xmin, xmax, tmin, tmax, k, v, g, dg, f, u0, Nx, Nt)
    return x, t, u


def gen_data_GRF(M, Nx, Nt, l, a, dname, isplot = False):
    # generate data for training the local solution operator
    f0 = lambda x: 0.9 * np.sin(2 * np.pi * x)
    space = GRF(1, length_scale=l, N=M, interp="cubic")
    features = space.random(1)
    f_T = lambda x, t: f0(x) + a * space.eval_u(features, x).T + 0 * t
    x, t, u_T =  compute_numerical_solution(f_T,M, M)
    f_T = f_T(x[:, None], t)
    
    f_T_grid = f_T[::round(M/Nx), ::round(M/Nt)]
    u_T_grid = u_T[::round(M/Nx), ::round(M/Nt)]
    x_grid = x[::round(M/Nx)]
    t_grid = t[::round(M/Nx)]
    
    # Generate the single data pair for training local solution operators
    # Prepare data and save using consistent f-string formatting
    np.savetxt(f"{dname}/x_grid.dat", np.rot90(np.tile(x_grid, (Nx, 1)), k=3))
    np.savetxt(f"{dname}/t_grid.dat", np.tile(t_grid, (Nt, 1)))
    np.savetxt(f"{dname}/f_T_grid.dat", f_T_grid)
    np.savetxt(f"{dname}/u_T_grid.dat", u_T_grid)

    # np.savetxt(f"{dname}/x.dat", np.rot90(np.tile(x, (M, 1)), k = 3))
    # np.savetxt(f"{dname}/t.dat", np.tile(t, (M, 1)))
    np.savetxt(f"{dname}/f_T.dat", f_T)
    np.savetxt(f"{dname}/u_T.dat", u_T)

    print("Generated f_T, u_T, f_T_grid and u_T_grid.")
    if isplot:
        plot_data("f_T_grid", "u_T_grid", dname)
    return

def gen_test_data(M, Nx, Nt, dname, isplot = False):
    f0 = lambda x: 0.9*np.sin(2 * np.pi * x)
    f_0 = lambda x, t: f0(x) + 0 * t
    x, t, u_0 =  compute_numerical_solution(f_0, M, M)
    f_0 = f_0(x[:, None], t)

    interp = interpolate.interp2d(t, x, u_0, kind='cubic')
    f_0_grid = f_0[::round(M/Nx), ::round(M/Nt)]
    u_0_grid = u_0[::round(M/Nx), ::round(M/Nt)]

    np.savetxt(f"{dname}/f_0_grid.dat", f_0_grid)
    np.savetxt(f"{dname}/u_0_grid.dat", u_0_grid)
    if isplot:
        plot_data("f_0_grid", "u_0_grid", dname)
    np.savetxt(f"{dname}/f_0.dat", f_0)
    np.savetxt(f"{dname}/u_0.dat", u_0)
    print("Generated f_0, u_0, f_0_grid, and u_0_grid.")
    return interp

def gen_new_data_GRF(M, Nx, Nt, N_f, N_b, dname, a_new, l_new, isplot):
    hx = 1/(Nx-1)
    ht = 1/(Nt-1)
    geom = dde.geometry.Interval(0 + hx, 1 - hx)
    timedomain = dde.geometry.TimeDomain(0 + ht, 1)
    geomtime = dde.geometry.GeometryXTime(geom, timedomain)

    x_random = sample_points(geomtime, Nx, Nt, N_f, N_b)
    print(f"Sampled {x_random.shape} random points.")

    interp, feature, space = gen_new_data_exact(M, Nx, Nt, a_new, l_new, dname, isplot)
    f0 = lambda x: 0.9*np.sin(2 * np.pi * x)
    f_new = lambda x, t: f0(x) + 0 * t + a_new * space.eval_u(feature, x).T
    f_new = np.array([f_new(i[0], i[1]) for i in x_random]).reshape((-1, 1))
    u_new =  np.array([interp(i[1], i[0]) for i in x_random]).reshape((-1, 1))

    np.savetxt(f"{dname}/f_new.dat", np.concatenate((x_random, f_new), axis = 1))
    np.savetxt(f"{dname}/u_new.dat", np.concatenate((x_random, u_new), axis = 1))

    print("Generated f_new and u_new.")
    if isplot:
        scatter_plot_data("f_new", "u_new", dname)
    return

def sample_points(geomtime, Nx, Nt, N_f, N_b):
    hx = 1/(Nx-1)
    ht = 1/(Nt-1)
    N_f = N_f - N_b
    x_random = geomtime.random_points(N_f, random = "Hammersley")
    x_f = []
    for i in x_random:
        if 0 + hx <= i[0] <= 1 - hx and 0 + ht <= i[1]:
            x_f.append(i)
    print("Removed {} point(s).".format(len(x_random) - len(x_f)))
    if N_b == 0:
        x_random  = np.array(x_f)
    else:
        geom = dde.geometry.Interval(0, 1)
        timedomain = dde.geometry.TimeDomain(0, 1)
        geomtime = dde.geometry.GeometryXTime(geom, timedomain)
        x_b = geomtime.random_boundary_points(N_b, random = "Hammersley")
        x_random = np.concatenate((x_f, x_b), axis = 0)
    return x_random

def gen_new_data_exact(M, Nx, Nt, a_new, l_new, dname, isplot):
    space = GRF(1, length_scale=l_new, N=M, interp="cubic")
    feature = space.random(1)
    f0 = lambda x: 0.9 * np.sin(2 * np.pi * x)
    fi = lambda x, t: f0(x) + 0 * t + a_new * space.eval_u(feature, x).T
    xi, ti, ui = compute_numerical_solution(fi, M, M)
    fi = fi(xi[:, None], ti)
    interp = interpolate.interp2d(ti, xi, ui, kind = "cubic")

    fi_grid = fi[::round(M/Nx), ::round(M/Nt)]
    ui_grid = ui[::round(M/Nx), ::round(M/Nt)]

    # np.savetxt(f"{dname}/fnew_1001.dat", fi)
    # np.savetxt(f"{dname}/unew_1001.dat", ui)

    np.savetxt(f"{dname}/f_new_grid.dat".format(dname), fi_grid)
    np.savetxt(f"{dname}/u_new_grid.dat".format(dname), ui_grid)
    if isplot:
        plot_data("f_new_grid", "u_new_grid", dname)
    print("Generated f_new_grid and u_new_grid.")
    return interp, feature, space

def gen_data_correction(interp, dname, isplot):
    x_random = np.loadtxt(f"{dname}/u_new.dat")[:, 0:2]
    f_0 = lambda x, t: 0.9*np.sin(2 * np.pi * x) + 0 * t
    f_0 = np.array([f_0(i[0], i[1]) for i in x_random]).reshape((-1, 1))
    u_0 =  np.array([interp(i[1], i[0]) for i in x_random]).reshape((-1, 1))

    np.savetxt(f"{dname}/f_init.dat", np.concatenate((x_random, f_0), axis = 1))
    np.savetxt(f"{dname}/u_init.dat", np.concatenate((x_random, u_0), axis = 1))
    if isplot:
        scatter_plot_data("f_init", "u_init", dname)
    print("Generated f_init and u_init.")
    return

def scatter_plot_data(fname, uname, dname):
    x = np.loadtxt(f"{dname}/{fname}.dat")[:,0]
    t = np.loadtxt(f"{dname}/{fname}.dat")[:,1]
    f = np.loadtxt(f"{dname}/{fname}.dat")[:,-1]
    u = np.loadtxt(f"{dname}/{uname}.dat")[:,-1]
    print("({}, {})".format(fname, uname))

    plt.figure()
    plt.rcParams.update({'font.size': 20,"savefig.dpi": 200, "figure.figsize": (8, 6)})
    plt.subplots_adjust(left=0.15, right=0.9, top=0.9, bottom=0.15)
    plt.scatter(x, t, c = f, cmap = "rainbow", s=0.5)
    plt.colorbar()
    plt.xlabel("x")
    plt.ylabel("t")
    plt.savefig(f"{dname}/{fname}.png")
    plt.show()

    plt.figure()
    plt.rcParams.update({'font.size': 20,"savefig.dpi": 200, "figure.figsize": (8, 6)})
    plt.subplots_adjust(left=0.15, right=0.9, top=0.9, bottom=0.15)
    plt.scatter(x, t, c = u, cmap = "rainbow", s=0.5)
    plt.colorbar()
    plt.xlabel("x")
    plt.ylabel("t")
    plt.savefig(f"{dname}/{uname}.png")
    plt.show()

def plot_data(fname, uname, dname):
    f = np.loadtxt(f"{dname}/{fname}.dat")
    u = np.loadtxt(f"{dname}/{uname}.dat")
    f = np.rot90(f)
    u = np.rot90(u)
    print("({}, {})".format(fname, uname))
    plt.figure()
    plt.plot(f[0])
    plt.show()
    plt.figure()
    plt.rcParams.update({'font.size': 20,"savefig.dpi": 200, "figure.figsize": (8, 6)})
    plt.subplots_adjust(left=0.15, right=0.9, top=0.9, bottom=0.15)
    plt.imshow(f, cmap = "rainbow",  origin='upper', extent=(0,1,0,1), aspect='auto')
    plt.colorbar()
    plt.xlabel("x")
    plt.ylabel("t")
    plt.savefig(f"{dname}/{fname}.png")
    plt.show()

    plt.figure()
    plt.subplots_adjust(left=0.15, right=0.9, top=0.9, bottom=0.15)
    plt.imshow(u,  origin='upper', cmap = "rainbow", extent=(0,1,0,1), aspect='auto')
    plt.colorbar()
    plt.xlabel("x")
    plt.ylabel("t")
    plt.savefig(f"{dname}/{uname}.png")
    plt.show()

def gen_all_data(M, Nx, Nt, N_f, N_b, l, a, l_new, a_new, dname, gen = False, correction = False, isplot = False):
    if gen:
        print("Generate new datasets ... ")
        gen_data_GRF(M, Nx, Nt, l, a, dname, isplot)
        interp_u0 = gen_test_data(M, Nx, Nt, dname, isplot)
        gen_new_data_GRF(M, Nx, Nt, N_f, N_b, dname, a_new, l_new, isplot)
        gen_data_correction(interp_u0, dname, isplot)


def gen_datasets(sigma, num_func, parent_dir = "./", gen = True):
    M = 1001 # Number of points 
    Nx, Nt = 101, 101
    N_f = 101*101
    N_b = Nx*2 + Nt - 2
    l, a = 0.01, 0.1
    l_new, a_new = 0.1, float(sigma)

    # Create folders for the datasets
    new_dir = "data_linear_diffusion"
    PATH = os.path.join(parent_dir, new_dir)
    os.makedirs(f"{PATH}/data_{sigma}", exist_ok = True)

    for i in range(num_func):
        print("Dataset {}".format(i))
        dname = f"{PATH}/data_{sigma}/data_{i}"
        os.makedirs(dname, exist_ok = True)

        # Generate datasets
        gen_all_data(M, Nx, Nt, N_f, N_b, l, a, l_new, a_new, dname, gen, 
                                          correction = False, isplot = False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num", type=int, default=3) # Number of functions
    parser.add_argument("--sigma", type=str, default="0.10") # Amplitude in the GRF
    args = parser.parse_args()
    gen_datasets(args.sigma, args.num)