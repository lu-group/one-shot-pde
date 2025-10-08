import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from porous_media_solver import solver
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
    D = 0.01
    k = 1
    Ca0 = lambda x: np.exp(-20 * x)
    Cb0 = lambda x: np.exp(-20 * x)
    x, t, Ca, Cb = solver(xmin, xmax, tmin, tmax, D, k, f, Ca0, Cb0, Nx, Nt)
    return x, t, Ca, Cb

def gen_data_GRF(M, Nx, Nt, l, a, dname, isplot=False):
    f0 = lambda x: np.exp(-((x - 0.5) ** 2) / 0.05)
    space = GRF(1, length_scale=l, N=M, interp="cubic")
    features = space.random(1)
    f_T = lambda x: f0(x) + a * np.ravel(space.eval_u(features, x))
    x, t, Ca_T, Cb_T =  compute_numerical_solution(f_T,M, M)
    f_T = np.tile(f_T(x)[:, None], (1, len(t)))

    f_T_grid = f_T[::round(M/Nx), ::round(M/Nt)]
    Ca_T_grid = Ca_T[::round(M/Nx), ::round(M/Nt)]
    Cb_T_grid = Cb_T[::round(M/Nx), ::round(M/Nt)]
    x_grid = x[::round(M/Nx)]
    t_grid = t[::round(M/Nx)]

    np.savetxt(f"{dname}/x_grid.dat", np.rot90(np.tile(x_grid, (Nx, 1)), k=3))
    np.savetxt(f"{dname}/t_grid.dat", np.tile(t_grid, (Nt, 1)))
    np.savetxt(f"{dname}/f_T_grid.dat", f_T_grid)
    np.savetxt(f"{dname}/Ca_T_grid.dat", Ca_T_grid)
    np.savetxt(f"{dname}/Cb_T_grid.dat", Cb_T_grid)
    print("Generated f_T_grid, Ca_T_grid and Cb_T_grid.")
    if isplot:
        plot_data("f_T_grid", "Ca_T_grid", dname)
        plot_data("f_T_grid", "Cb_T_grid", dname)

    np.savetxt(f"{dname}/x.dat", np.rot90(np.tile(x, (M, 1)), k=3))
    np.savetxt(f"{dname}/t.dat", np.tile(t, (M, 1)))
    np.savetxt(f"{dname}/f_T.dat", f_T)
    np.savetxt(f"{dname}/Ca_T.dat", Ca_T)
    np.savetxt(f"{dname}/Cb_T.dat", Cb_T)

    print("Generated f_T, Ca_T and Cb_T.")
    if isplot:
        plot_data("f_T", "Ca_T", dname)
        plot_data("f_T", "Cb_T", dname)
    return

def gen_test_data(M, Nx, Nt, dname, isplot=False):
    f0 = lambda x: np.exp(-((x - 0.5) ** 2) / 0.05)
    f_0 = lambda x: f0(x)
    x, t, Ca_0, Cb_0 =  compute_numerical_solution(f_0, M, M)
    f_0 = np.tile(f_0(x)[:, None], (1, len(t)))

    np.savetxt(f"{dname}/f_0.dat", f_0)
    np.savetxt(f"{dname}/Ca_0.dat", Ca_0)
    np.savetxt(f"{dname}/Cb_0.dat", Cb_0)

    if isplot:
        plot_data("f_0", "Ca_0", dname)
        plot_data("f_0", "Cb_0", dname)

    interp_a = interpolate.RegularGridInterpolator((x, t), Ca_0, method='cubic', bounds_error=False, fill_value=0 )
    interp_b = interpolate.RegularGridInterpolator((x, t), Cb_0, method='cubic', bounds_error=False, fill_value=0 )

    f_0_grid = f_0[::round(M/Nx), ::round(M/Nt)]
    Ca_0_grid = Ca_0[::round(M/Nx), ::round(M/Nt)]
    Cb_0_grid = Cb_0[::round(M/Nx), ::round(M/Nt)]

    np.savetxt(f"{dname}/f_0_grid.dat", f_0_grid)
    np.savetxt(f"{dname}/Ca_0_grid.dat", Ca_0_grid)
    np.savetxt(f"{dname}/Cb_0_grid.dat", Cb_0_grid)
    if isplot:
        plot_data("f_0_grid", "Ca_0_grid", dname)
        plot_data("f_0_grid", "Cb_0_grid", dname)

    print("Generated f_0, Ca_0, Cb_0, f_0_grid, Ca_0_grid, and Cb_0_grid.")
    return interp_a, interp_b

def sample_points(geomtime, Nx, Nt, N_f, N_b):
    hx = 1/(Nx-1)
    ht = 1/(Nt-1)
    N_f = N_f - Nx*2 - Nt + 2
    x_random = geomtime.random_points(N_f, random = "Hammersley")
    x_f = []
    for i in x_random:
        if 0 + hx <= i[0] <= 1 - hx and 0 + ht <= i[1] <= 1 :
            #print(i)
            x_f.append(i)
    print("Removed {} point(s).".format(len(x_random) - len(x_f)))
    if N_b == 0:
        x_random  = np.array(x_f)
    else:
        x_b = geomtime.random_boundary_points(N_b, random = "Hammersley")
        x_random = np.concatenate((x_f, x_b), axis = 0)
    return x_random

def gen_new_data_exact(M, Nx, Nt, a_new, l_new, dname, isplot=False):
    space = GRF(1, length_scale=l_new, N=1001, interp="cubic")
    feature = space.random(1)
    f0 = lambda x: np.exp(-((x - 0.5) ** 2) / 0.05)
    fi = lambda x: f0(x)+ a_new * np.ravel(space.eval_u(feature, x))
    xi, ti, Cai, Cbi = compute_numerical_solution(fi,1001, 1001)
    fi = np.tile(fi(xi)[:, None], (1, len(ti)))
    # np.savetxt(f"{dname}/fnew_1001.dat", fi)
    # np.savetxt(f"{dname}/Canew_1001.dat", Cai)
    # np.savetxt(f"{dname}/Cbnew_1001.dat", Cbi)
    interp_a = interpolate.RegularGridInterpolator((xi, ti), Cai, method='cubic', bounds_error=False, fill_value=0)
    interp_b = interpolate.RegularGridInterpolator((xi, ti), Cbi, method='cubic', bounds_error=False, fill_value=0)


    fi_grid = fi[::round(M/Nx), ::round(M/Nt)]
    Cai_grid = Cai[::round(M/Nx), ::round(M/Nt)]
    Cbi_grid = Cbi[::round(M/Nx), ::round(M/Nt)]
    np.savetxt(f"{dname}/f_new_grid.dat", fi_grid)
    np.savetxt(f"{dname}/Ca_new_grid.dat", Cai_grid)
    np.savetxt(f"{dname}/Cb_new_grid.dat", Cbi_grid)

    print("Generated f_new_grid, Ca_new_grid and Cb_new_grid.")
    if isplot:
        plot_data("f_new_grid", "Ca_new_grid", dname)
        plot_data("f_new_grid", "Cb_new_grid", dname)
    return interp_a, interp_b, feature, space

def gen_new_data_GRF(M, Nx, Nt, N_f, N_b, dname, a_new, l_new, isplot=False):
    hx = 1/(Nx-1)
    ht = 1/(Nt-1)
    geom = dde.geometry.Interval(0 + hx, 1 - hx)
    timedomain = dde.geometry.TimeDomain(0 + ht, 1)
    geomtime = dde.geometry.GeometryXTime(geom, timedomain)
    x_random = sample_points(geomtime, Nx, Nt, N_f, N_b)
    interp_a, interp_b, feature, space = gen_new_data_exact(M, Nx, Nt, a_new, l_new, dname, isplot)
    f0 = lambda x: np.exp(-((x - 0.5) ** 2) / 0.05)
    f_new = lambda x: f0(x) + a_new * np.ravel(space.eval_u(feature, x))
    f_new = f_new(x_random[:, 0]).reshape((-1, 1))
    Ca_new =  np.array([interp_a((i[0], i[1])) for i in x_random]).reshape((-1, 1))
    Cb_new =  np.array([interp_b((i[0], i[1])) for i in x_random]).reshape((-1, 1))

    np.savetxt(f"{dname}/f_new.dat", np.concatenate((x_random, f_new), axis = 1))
    np.savetxt(f"{dname}/Ca_new.dat", np.concatenate((x_random, Ca_new), axis = 1))
    np.savetxt(f"{dname}/Cb_new.dat", np.concatenate((x_random, Cb_new), axis = 1))
    print("Generated f_new, Ca_new and Cb_new.")
    return

def gen_data_correction(interp_a, interp_b, dname, isplot=False):
    f0 = lambda x: np.exp(-((x - 0.5) ** 2) / 0.05)
    x_random = np.loadtxt(f"{dname}/Ca_new.dat")[:, 0:2]
    f_0 = f0(x_random[:, 0]).reshape((-1, 1))
    Ca_0 =  np.array([interp_a((i[0], i[1])) for i in x_random]).reshape((-1, 1))
    Cb_0 =  np.array([interp_b((i[0], i[1])) for i in x_random]).reshape((-1, 1))

    np.savetxt(f"{dname}/f_init.dat", np.concatenate((x_random, f_0), axis = 1))
    np.savetxt(f"{dname}/Ca_init.dat", np.concatenate((x_random, Ca_0), axis = 1))
    np.savetxt(f"{dname}/Cb_init.dat", np.concatenate((x_random, Cb_0), axis = 1))

    print("Generated f_init, Ca_init and Cb_init.")
    return

def plot_data(f_name, u_name, dname):
    f = np.loadtxt(f"{dname}/{f_name}.dat")
    u = np.loadtxt(f"{dname}/{u_name}.dat")
    f = np.rot90(f)
    u = np.rot90(u)
    print("({}, {})".format(f_name, u_name))
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
    # plt.savefig(f"{dname}/{f_name}.png")
    plt.show()

    plt.figure()
    plt.subplots_adjust(left=0.15, right=0.9, top=0.9, bottom=0.15)
    plt.imshow(u,  origin='upper', cmap = "gnuplot", extent=(0,1,0,1), aspect='auto')
    plt.colorbar()
    plt.xlabel("x")
    plt.ylabel("t")
    # plt.savefig(f"{dname}/{u_name}.png")
    plt.show()

def construct_data(f, u1, u2):
    Nx, Nt = f.shape
    inputs = []
    outputs1 = []
    outputs2 = []
    for i in range(1, Nx - 1):
        for j in range(1, Nt):
            inputs.append(
                [
                    f[i, j],
                    u1[i - 1, j],
                    # u1[i - 1, j - 1],
                    u1[i, j - 1],
                    # u1[i + 1, j - 1],
                    u1[i + 1, j],
                    # u2[i - 1, j - 1],
                    u2[i - 1, j],
                    u2[i, j - 1],
                    # u2[i + 1, j - 1],
                    u2[i + 1, j],
                    # u1[i, j] * u2[i, j] ** 2,
                    # np.log(max(u1[i, j], 1e-16)),
                    # np.log(max(u2[i, j], 1e-16)),
                ]
            )
            outputs1.append([u1[i, j]])
            outputs2.append([u2[i, j]])
    return np.array(inputs), np.array(outputs1), np.array(outputs2)

def construct_more_data(Nx, Nt, f, Ca, Cb):
    # f[i, j], u[i-1, j], u[i, j-1], u[i+1, j]
    M = len(f)
    xstep = int((M-1)/(Nx-1))
    tstep = int((M-1)/(Nt-1))
    outputs_a = Ca[xstep: -xstep, tstep:].reshape((-1, 1))
    outputs_b = Cb[xstep: -xstep, tstep:].reshape((-1, 1))
    inputs = np.hstack((f[xstep: -xstep, tstep:].reshape((-1, 1)), \
        Ca[:-2*xstep, tstep:].reshape((-1, 1)), Ca[xstep: -xstep, :-tstep].reshape((-1, 1)), Ca[2*xstep: , tstep:].reshape((-1, 1)), \
        Cb[:-2*xstep, tstep:].reshape((-1, 1)), Cb[xstep: -xstep, :-tstep].reshape((-1, 1)), Cb[2*xstep: , tstep:].reshape((-1, 1))))
    return inputs, outputs_a, outputs_b

def gen_all_data(M, Nx, Nt, N_f, N_b, l, a, l_new, a_new, dname, gen=False, isplot=False):
    if gen:
        print("Generate new dataset ...")
        gen_data_GRF(M, Nx, Nt, l, a, dname, isplot)
        interp_Ca0, interp_Cb0 = gen_test_data(M, Nx, Nt, dname, isplot)
        gen_new_data_GRF(M, Nx, Nt, N_f, N_b, dname, a_new, l_new, isplot)
        gen_data_correction(interp_Ca0, interp_Cb0, dname, isplot)

def gen_datasets(sigma, num_func, parent_dir="./", gen=True):
    M = 1001
    Nx, Nt = 101, 101
    N_f = 101 * 101
    N_b = 0
    l, a = 0.01, 0.1
    l_new, a_new = 0.1, float(sigma)

    new_dir = "data_porous_media"
    PATH = os.path.join(parent_dir, new_dir)
    os.makedirs(f"{PATH}/data_{sigma}", exist_ok=True)

    for i in range(num_func):
        print(f"Dataset {i}")
        dname = f"{PATH}/data_{sigma}/data_{i}"
        os.makedirs(dname, exist_ok=True)
        gen_all_data(M, Nx, Nt, N_f, N_b, l, a, l_new, a_new, dname, gen=gen, isplot=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num", type=int, default=3) # Number of functions
    parser.add_argument("--sigma", type=str, default="0.10") # Amplitude in the GRF
    args = parser.parse_args()
    gen_datasets(args.sigma, args.num)