from math import e
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
    g = lambda u: 0.01 * u ** 2
    dg = lambda u: 0.02 * u
    u0 = lambda x: np.zeros_like(x)

    x, t, u = solve_ADR(xmin, xmax, tmin, tmax, k, v, g, dg, f, u0, Nx, Nt)
    return x, t, u


def gen_new_data_exact(M, Nx, Nt, a_new, l_new, dname, ftype, isplot):
    space = GRF(1, length_scale=l_new, N=M, interp="cubic")
    feature = space.random(1)
    if ftype == "orig":
        f0 = lambda x: np.sin(2 * np.pi * x)
    elif ftype == "sin":
        f0 = lambda x: np.sin(2 * np.pi * x + np.pi)
    elif ftype == "cos":
        f0 = lambda x: np.cos(2 * np.pi * x)
    elif ftype == "freq":
        f0 = lambda x: np.sin(2 * np.pi * 1.5 * x)
    elif ftype == "hfreq":
        f0 = lambda x: 2*np.sin(2 * np.pi * 4.5 * x)
    elif ftype == "twofreqs":
        f0 = lambda x: np.sin(2 * np.pi * 3 * x) + np.sin(2 * np.pi * 7 * x)
    elif ftype == "morefreqs":
        f0 = lambda x: (
            0.5 * np.sin(2 * np.pi * 1 * x) +
            0.5 * np.sin(2 * np.pi * 3 * x) +
            0.5 * np.sin(2 * np.pi * 7 * x) +
            0.5 * np.sin(2 * np.pi * 13 * x)
        )
    elif ftype == "piecewise":
        f0 = lambda x: np.where(x < 0.5,
                        2*x * np.sin(2*np.pi*4*x),
                        (1 - x) * np.cos(2*np.pi*10*(x - 0.5)))
    elif ftype == "chirp":
        # 2. Chirp / frequency sweep (non-stationary)
        f0 = lambda x: np.sin(2*np.pi*(5*x + 5*x**2)) 
    elif ftype == "modulated":
        f0 = lambda x: (0.6 + 0.4*np.cos(2*np.pi*2*x)) * np.sin(2*np.pi*15*x)
    elif ftype == "noise1":
        f0 = lambda x: 0.01*np.random.normal(0, 1, size=x.shape)*np.cos(2 * np.pi * x) + np.cos(2 * np.pi * x)
        fi_denoise = lambda x, t: np.cos(2 * np.pi * x) + 0 * t + 0 * x 
    elif ftype == "noise5":
        f0 = lambda x: 0.05*np.random.normal(0, 1, size=x.shape)*np.cos(2 * np.pi * x) + np.cos(2 * np.pi * x)
        fi_denoise = lambda x, t: np.cos(2 * np.pi * x) + 0 * t + 0 * x 
    elif ftype == "noise10":
        f0 = lambda x: 0.10*np.random.normal(0, 1, size=x.shape)*np.cos(2 * np.pi * x) + np.cos(2 * np.pi * x)
        fi_denoise = lambda x, t: np.cos(2 * np.pi * x) + 0 * t + 0 * x 
    elif ftype == "noise15":
        # f0 = lambda x: np.random.normal(0, 0.15, size=x.shape) + np.cos(2 * np.pi * x)
        f0 = lambda x: 0.15*np.random.normal(0, 1, size=x.shape)*np.cos(2 * np.pi * x) + np.cos(2 * np.pi * x)
        fi_denoise = lambda x, t: np.cos(2 * np.pi * x) + 0 * t + 0 * x 
    elif ftype == "noise20":
        f0 = lambda x: 0.20*np.random.normal(0, 1, size=x.shape)*np.cos(2 * np.pi * x) + np.cos(2 * np.pi * x)
        fi_denoise = lambda x, t: np.cos(2 * np.pi * x) + 0 * t + 0 * x 
    elif ftype == "noise30":
        f0 = lambda x: 0.30*np.random.normal(0, 1, size=x.shape)*np.cos(2 * np.pi * x) + np.cos(2 * np.pi * x)
        fi_denoise = lambda x, t: np.cos(2 * np.pi * x) + 0 * t + 0 * x 
    fi = lambda x, t: f0(x) + 0 * t + 0 * x #a_new * space.eval_u(feature, x).T
    xi, ti, ui = compute_numerical_solution(fi,M, M)
    _, _, ui = compute_numerical_solution(fi_denoise, M, M) if ftype == "noise" else (xi, ti, ui)
    fi = fi(xi[:, None], ti)
    interp = interpolate.RegularGridInterpolator((xi, ti), ui, method="cubic")
    
    np.savetxt(f"{dname}/f_new.dat".format(dname), fi)
    np.savetxt(f"{dname}/u_new.dat".format(dname), ui)
    if isplot:
        plot_data("f_new", "u_new", dname)
    print("Generated f_new and u_new.")

    fi_grid = fi[::round(M/Nx), ::round(M/Nt)]
    ui_grid = ui[::round(M/Nx), ::round(M/Nt)]

    np.savetxt(f"{dname}/f_new_grid.dat".format(dname), fi_grid)
    np.savetxt(f"{dname}/u_new_grid.dat".format(dname), ui_grid)
    if isplot:
        plot_data("f_new_grid", "u_new_grid", dname)
    print("Generated f_new_grid and u_new_grid.")
    return interp, feature, space


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
    # x = np.loadtxt(f"{dname}/x_grid.dat")
    f = np.loadtxt(f"{dname}/{fname}.dat")
    u = np.loadtxt(f"{dname}/{uname}.dat")
    f = np.rot90(f)
    u = np.rot90(u)
    print("({}, {})".format(fname, uname))
    plt.figure()
    plt.plot(f[0])
    plt.xlabel("x")
    plt.ylabel("{}".format(fname))
    plt.savefig(f"{dname}/{fname}.png")
    plt.show()
    
    plt.figure()
    plt.rcParams.update({'font.size': 20,"savefig.dpi": 200, "figure.figsize": (8, 6)})
    plt.subplots_adjust(left=0.15, right=0.9, top=0.9, bottom=0.15)
    plt.imshow(f, cmap = "rainbow",  origin='upper', extent=(0,1,0,1), aspect='auto')
    plt.colorbar()
    plt.xlabel("x")
    plt.ylabel("t")
    plt.savefig(f"{dname}/{fname}_2D.png")
    plt.show()

    plt.figure()
    plt.subplots_adjust(left=0.15, right=0.9, top=0.9, bottom=0.15)
    plt.imshow(u,  origin='upper', cmap = "rainbow", extent=(0,1,0,1), aspect='auto')
    plt.colorbar()
    plt.xlabel("x")
    plt.ylabel("t")
    plt.savefig(f"{dname}/{uname}.png")
    plt.show()


def gen_datasets(sigma, num_func, ftype, parent_dir = "./", gen = True):
    M = 1001 # Number of points 
    Nx, Nt = 101, 101
    N_f = 101*101
    N_b = Nx*2 + Nt - 2
    l, a = 0.01, 0.1
    l_new, a_new = 0.1, float(sigma)

    # Create folders for the datasets
    new_dir = "data_nonlinear_diffusion"
    PATH = os.path.join(parent_dir, new_dir)
    os.makedirs(f"{PATH}/data_{sigma}_{ftype}", exist_ok = True)

    for i in range(num_func):
        print("Dataset {}".format(i))
        dname = f"{PATH}/data_{sigma}_{ftype}/data_{i}"
        os.makedirs(dname, exist_ok = True)
        # Generate datasets
        gen_new_data_exact(M, Nx, Nt, a_new, l_new, dname, ftype, isplot=False)
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num", type=int, default=1) # Number of functions
    parser.add_argument("--sigma", type=str, default="0.80") # Amplitude in the GRF
    parser.add_argument("--ftype", type=str, default="noise1")
    args = parser.parse_args()
    gen_datasets(args.sigma, args.num, args.ftype)