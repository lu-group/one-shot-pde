"""Solve a nonlinear Poisson's equation,

    div (1 + u^2) grad u(x, y) = f(x, y)

and boundary conditions given by

    u(x, y) = 1

This is equivalent to solving the variational problem

    F(u) = ((1 + u^2)*grad(u), grad(v)) + (f, v) = 0

Ref:
- https://fenicsproject.org/olddocs/dolfin/1.3.0/python/demo/documented/poisson/python/documentation.html
- https://fenicsproject.org/olddocs/dolfin/1.3.0/python/demo/documented/nonlinear-poisson/python/documentation.html
- https://github.com/nw2190/ConvPDE/blob/master/Nonlinear_Poisson/Setup/solver.py
"""

import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import argparse
import matplotlib.pyplot as plt
import os
import meshio
import numpy as np
from dolfin import *
from scipy import interpolate, ndimage

from spaces import GRF2D


class MyExpression(UserExpression):
    def __init__(self, f, x, y, **kwargs):
        super().__init__(**kwargs)
        self.interp = interpolate.RegularGridInterpolator((x, y), f)

    def _eval(self, x):
        # return x[0] * np.sin(x[1])  # f0
        return self.interp(x)[0]

    def eval(self, value, x):
        value[0] = self._eval(x)

    def eval_batch(self, x):
        return np.array(list(map(self._eval, x)))


def convert(f, d_num, fname, Ny, Nx):
    mesh = meshio.read("data{}/u{}000000.vtu".format(d_num, fname))
    x = mesh.points[:, :2]
    #u = mesh.point_data[key][:, None]
    u = list(mesh.point_data.values())[0][:, None]
    # np.savetxt("data/u.dat", np.hstack((x, u)))
    np.savetxt("data{}/u{}.dat".format(d_num, fname), u.reshape(Ny, Nx))  # Ny x Nx
    f = f.eval_batch(x)[:, None]
    # np.savetxt("data/f.dat", np.hstack((x, f)))
    np.savetxt("data{}/f{}.dat".format(d_num, fname), f.reshape(Ny, Nx))
    return f.reshape(Ny, Nx), u.reshape(Ny, Nx)

def solver(f, d_num, fname, M):
    # Create mesh and define function space
    mesh = UnitSquareMesh(M-1, M-1)
    V = FunctionSpace(mesh, "CG", 1)

    # Define boundary condition
    g = Constant(0.0)
    bc = DirichletBC(V, g, lambda _, on_boundary: on_boundary)

    # Nonlinear solver
    u = Function(V)
    v = TestFunction(V)
    F = inner((1 + u ** 2) * grad(u), grad(v)) * dx + 10 * f * v * dx
    solve(F == 0, u, bc)

    # Save solution in VTK format
    file = File("data{}/u{}.pvd".format(d_num, fname))
    file << u


def gen_data_GRF(M, Nx, Ny, l, a, d_num):
    '''
    Generate training data (f_T, u_T) from GRF.
    '''
    x = np.linspace(0, 1, num=M)
    y = np.linspace(0, 1, num=M)
    np.savetxt("data{}/y.dat".format(d_num), np.rot90(np.tile(y, (M, 1)), k = 3))
    np.savetxt("data{}/x.dat".format(d_num), np.tile(x, (M, 1)))
    # define f0
    f0 = x[:, None] * np.sin(y)

    space = GRF2D(length_scale=l, N=101)
    feature = space.random(1)
    xv, yv = np.meshgrid(x, y)
    f = space.eval_u(feature, np.vstack((np.ravel(xv), np.ravel(yv))).T)[0]
    f = np.reshape(f, (len(y), len(x))).T
    f = f0 + a * f
    f = MyExpression(f, x, y, degree=2)
    solver(f, d_num, "_T", M)
    f_T, u_T = convert(f, d_num, "_T", M, M)
    f_T_grid = f_T[::round(M/Nx), ::round(M/Ny)]
    u_T_grid = u_T[::round(M/Nx), ::round(M/Ny)]
    x_grid = x[::round(M/Nx)]
    y_grid = y[::round(M/Ny)]

    np.savetxt("data{}/x_grid.dat".format(d_num), np.rot90(np.tile(x_grid, (Nx, 1)), k = 3))
    np.savetxt("data{}/y_grid.dat".format(d_num), np.tile(y_grid, (Ny, 1)))
    np.savetxt("data{}/f_T_grid.dat".format(d_num), f_T_grid)
    np.savetxt("data{}/u_T_grid.dat".format(d_num), u_T_grid)

    # plot data
    print("Generated f_T and u_T.")
    plot_data("f_T", "u_T", d_num)
    # plot data
    print("Generated f_T_grid and u_T_grid.")
    plot_data("f_T_grid", "u_T_grid", d_num)


def gen_test_data(M, Nx, Ny, d_num):
    x = np.linspace(0, 1, num=M)
    y = np.linspace(0, 1, num=M)
    # define f0
    f0 = x[:, None] * np.sin(y)
    f = MyExpression(f0, x, y, degree=2)
    solver(f, d_num, "_0", M)
    f_0, u_0 = convert(f, d_num, "_0", M, M)
    interp = interpolate.interp2d(y, x, u_0, kind='cubic')

    # grid data for FPI
    f_0_grid = f_0[::round(M/Nx), ::round(M/Ny)]
    u_0_grid = u_0[::round(M/Nx), ::round(M/Ny)]

    np.savetxt("data{}/f_0_grid.dat".format(d_num), f_0_grid)
    np.savetxt("data{}/u_0_grid.dat".format(d_num), u_0_grid)

    # plot data
    print("Generated f_0, u_0, f_0_grid, and u_0_grid.")
    plot_data("f_0", "u_0", d_num)
    plot_data("f_0_grid", "u_0_grid", d_num)
    return interp


def gen_new_data_exact(M, Nx, Ny, a_new, l_new, d_num):
    x = np.linspace(0, 1, num=M)
    y = np.linspace(0, 1, num=M)
    f0 = x[:, None] * np.sin(y)

    space = GRF2D(length_scale=l_new, N=101)
    feature = space.random(1)
    xv, yv = np.meshgrid(x, y)
    f = space.eval_u(feature, np.vstack((np.ravel(xv), np.ravel(yv))).T)[0]
    f = np.reshape(f, (len(y), len(x))).T
    f = f0 + a_new * f
    f = MyExpression(f, x, y, degree=2)
    solver(f, d_num, "new_M", M)
    fi, ui = convert(f, d_num, "new_M", M, M)

    interp = interpolate.interp2d(y, x, ui, kind = "cubic")

    fi_grid = fi[::round(M/Nx), ::round(M/Ny)]
    ui_grid = ui[::round(M/Nx), ::round(M/Ny)]
    np.savetxt("data{}/f_new_grid.dat".format(d_num), fi_grid)
    np.savetxt("data{}/u_new_grid.dat".format(d_num), ui_grid)

    print("Generated f_new_grid and u_new_grid.")
    plot_data("f_new_grid", "u_new_grid", d_num)
    return interp, feature, space

def gen_new_data_GRF(M, Nx, Ny, N_f, N_b, d_num, a_new, l_new):
    # points in the domain
    x_random = np.loadtxt("data{}/x_random.dat".format(d_num))

    interp, feature, space = gen_new_data_exact(M, Nx, Ny, a_new, l_new, d_num)

    f_new = lambda x, y: x * np.sin(y) + a_new * space.eval_u(feature, np.vstack((np.ravel(x), np.ravel(y))).T)[0]
    f_new = np.array([f_new(i[0], i[1]) for i in x_random]).reshape((-1, 1))
    u_new =  np.array([interp(i[1], i[0]) for i in x_random]).reshape((-1, 1))
    print(u_new.shape) #(10197,1)
    # save data into files
    np.savetxt("data{}/f_new.dat".format(d_num), np.concatenate((x_random, f_new), axis = 1))
    np.savetxt("data{}/u_new.dat".format(d_num), np.concatenate((x_random, u_new), axis = 1))

    # plot data
    print("Generated f_new and u_new.")
    #plot_data("f_new", "u_new", d_num)
    return

def gen_data_correction(interp, d_num):
    f_0 = lambda x, y: x * np.sin(y)
    x_random = np.loadtxt("data{}/x_random.dat".format(d_num))
    f_0 = np.array([f_0(i[0], i[1]) for i in x_random]).reshape((-1, 1))
    u_0 =  np.array([interp(i[1], i[0]) for i in x_random]).reshape((-1, 1))

    np.savetxt("data{}/f_init.dat".format(d_num), np.concatenate((x_random, f_0), axis = 1))
    np.savetxt("data{}/u_init.dat".format(d_num), np.concatenate((x_random, u_0), axis = 1))

    # plot data
    print("Generated f_init and u_init.")
    #plot_data("f_init", "u_init", d_num)
    return

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
    plt.savefig("data{}/{}.png".format(d_num,f_name))
    #plt.show()

    plt.figure()
    plt.subplots_adjust(left=0.15, right=0.9, top=0.9, bottom=0.15)
    plt.imshow(u,  origin='upper', cmap = "gnuplot", extent=(0,1,0,1), aspect='auto')
    plt.colorbar()
    plt.xlabel("x")
    plt.ylabel("t")
    plt.savefig("data{}/{}.png".format(d_num,u_name))
    #plt.show()                           

def construct_more_data(Nx, Ny, f, u):
    # u[i-1, j], u[i, j-1], u[i+1, j], u[i, j+1], f[i, j]
    M = len(f) # 1001
    xstep = int((M-1)/(Nx-1))
    tstep = int((M-1)/(Ny-1))
    #print(xstep, tstep) #10, 10
    outputs = u[xstep: -xstep, tstep:-tstep].reshape((-1, 1))
    #print(outputs.shape) #(972171, 1)
    #print(u[:-2*xstep, tstep:].shape, u[xstep: -xstep, :-tstep].shape, u[2*xstep: , tstep:].shape, f[xstep: -xstep, tstep].shape)
    inputs = np.hstack((u[:-2*xstep, tstep:-tstep].reshape((-1, 1)), u[xstep: -xstep, :-2*tstep].reshape((-1, 1)), u[2*xstep: , tstep:-tstep].reshape((-1, 1)), u[xstep: -xstep, 2*tstep:].reshape((-1, 1)),f[xstep: -xstep, tstep:].reshape((-1, 1))))
    print(inputs.shape, outputs.shape)
    return inputs, outputs


def main(sigma, num_func, parent_dir = "../../data/", gen = True):
    M = 201
    Nx, Ny = 101, 101
    N_f = 101*101
    N_b = 0
    l, a = 0.01, 0.10
    l_new, a_new = 0.1, float(sigma)
    # Create folders for the datasets
    new_dir = "data_nonlinear_poisson"
    PATH = os.path.join(parent_dir, new_dir)
    os.makedirs(f"{PATH}/data_{sigma}", exist_ok = True)
    
    for i in range(num_func):
        d_num = "_{}/data_{}".format(sigma, i) # dataset number
        os.makedirs("data_{}/data_{}".format(sigma, i), exist_ok = True)
        
        gen_data_GRF(M, Nx, Ny, l, a, d_num)
        interp_u0 = gen_test_data(M, Nx, Ny, d_num)
        gen_data_correction(interp_u0, d_num)
        gen_new_data_GRF(M, Nx, Ny, N_f, N_b, d_num, a_new, l_new)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num", type=int, default=3) # Number of functions
    parser.add_argument("--sigma", type=str, default="0.05") # Amplitude in the GRF
    args = parser.parse_args()
    main(args.sigma, args.num)
