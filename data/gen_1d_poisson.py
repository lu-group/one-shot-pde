import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.environ["TF_XLA_FLAGS"] = '--tf_xla_cpu_global_jit'
from spaces import GRF
import argparse
import deepxde as dde
import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt
dde.backend.set_default_backend('tensorflow.compat.v1')
dde.config.set_default_float("float64")

def compute_numerical_solution(x, f, N):
    # compute the reference solution
    h = x[1]
    K = -2 * np.eye(N - 2) + np.eye(N - 2, k=1) + np.eye(N - 2, k=-1)
    b = h ** 2 * 100 * f[1:-1]
    u = np.linalg.solve(K,b)
    return np.concatenate(([0], u, [0]))

def gen_data_GRF(M, N, l, a, geom, dname, isplot = False):
    # generate data for training the local solution operator
    space = GRF(1, length_scale=l, N=M, interp="cubic")
    x = np.ravel(geom.uniform_points(M))
    
    # f_T = f_random + f0, f0 = np.sin(2 * np.pi * x)
    f_T = np.ravel(a * space.eval_u(space.random(1), x[:, None])) + np.sin(2 * np.pi * x)
    u_T =  compute_numerical_solution(x,f_T,M) #(M,)
    
    index = [round(M/N) * i for i in range(N)]
    x_grid = x[index]
    f_T_grid = f_T[index]
    u_T_grid = u_T[index]
    
    # Generate the single data pair for training local solution operators
    np.savetxt(f"{dname}/f_T_grid.dat", np.column_stack((x_grid, f_T_grid)))
    np.savetxt(f"{dname}/u_T_grid.dat", np.column_stack((x_grid, u_T_grid)))
    np.savetxt(f"{dname}/f_T.dat", np.column_stack((x, f_T)))
    np.savetxt(f"{dname}/u_T.dat", np.column_stack((x, u_T)))

    print("Generated f_T_grid and u_T_grid.")
    if isplot:
        plot_data("f_T_grid", "u_T_grid", dname)
    return

def gen_test_data(M, N, geom, dname, isplot = False):
    # f_0 = sin(2*pi*x)
    x = np.ravel(geom.uniform_points(M))
    f_0  = np.sin(2 * np.pi * x)
    u_0 =  compute_numerical_solution(x,f_0,M)
    interp = interpolate.interp1d(x, u_0, kind = "cubic")

    x_grid = np.ravel(geom.uniform_points(N))
    f_0_grid  = np.sin(2 * np.pi * x_grid)
    u_0_grid =  compute_numerical_solution(x_grid,f_0_grid,N)
    
    np.savetxt(f"{dname}/f_0_grid.dat", np.column_stack((x_grid, f_0_grid)))
    np.savetxt(f"{dname}/u_0_grid.dat", np.column_stack((x_grid, u_0_grid)))
    if isplot:
        plot_data("f_0_grid", "u_0_grid", dname)
    print("Generated f_0_grid and u_0_grid.")
    # np.savetxt(f"{dname}/f_0.dat", np.column_stack((x, f_0)))
    np.savetxt(f"{dname}/u_0.dat", np.column_stack((x, u_0)))
    return interp

def gen_new_data_GRF(M, N, N_f, N_b, geom, dname, a_new, l_new, isplot = False):
    # sampler https://github.com/lululxvi/deepxde/blob/c8699216ccf7de18415e6c98bfab13f1d902bc06/deepxde/geometry/sampler.py#L35
    # a new f = f0 + delta f
    x_random = sample_points(N, N_f, N_b)
    interp, feature, space = gen_new_data_exact(M, N, geom, a_new, l_new, dname, isplot)
    delta_f = np.ravel(a_new * space.eval_u(feature, x_random[:, None]))
    f_new  = np.sin(2 * np.pi * x_random) + delta_f
    u_new =  interp(x_random)

    np.savetxt(f"{dname}/f_new.dat", np.column_stack((x_random, f_new)))
    np.savetxt(f"{dname}/u_new.dat", np.column_stack((x_random, u_new)))

    print("Generated f_new and u_new.")
    if isplot:
        plot_data("f_new", "u_new", dname)
    return

def sample_points(N, N_f, N_b):
    h = 1/(N-1)
    geom = dde.geometry.Interval(0+h, 1-h)
    x_random = geom.random_points(N_f, random = "Hammersley")
    x_f = [i for i in x_random if 0 + h <= i <= 1 - h]
    print("Removed {} point(s).".format(len(x_random) - len(x_f)))
    x_f = np.sort(np.ravel(x_f))
    if N_b == 2:
        x_b = [0, 1]
        x_random = np.ravel(np.concatenate((x_f, x_b), axis = 0))
    else:
        x_random = np.ravel(x_f)
    return x_random

def gen_new_data_exact(M, N, geom, a_new, l_new, dname, isplot):
    space = GRF(1, length_scale=l_new, N=M, interp="cubic")
    xi = np.ravel(geom.uniform_points(M))
    feature = space.random(1)
    fi  = np.sin(2 * np.pi * xi) + np.ravel(a_new * space.eval_u(feature, xi[:, None]))
    ui = compute_numerical_solution(xi,fi,M)
    interp = interpolate.interp1d(xi, ui, kind = "cubic")
    
    # Generate f_new and u_new data for testing
    x_grid = np.ravel(geom.uniform_points(N))
    delta_f = np.ravel(a_new * space.eval_u(feature, x_grid[:, None]))
    f_new_grid  = np.sin(2 * np.pi * x_grid) + delta_f
    u_new_grid =  compute_numerical_solution(x_grid,f_new_grid,N)
    np.savetxt(f"{dname}/f_new_grid.dat".format(dname), np.vstack((x_grid, f_new_grid)).T)
    np.savetxt(f"{dname}/u_new_grid.dat".format(dname), np.vstack((x_grid, u_new_grid)).T)
    if isplot:
        plot_data("f_new_grid", "u_new_grid", dname)
    print("Generated f_new_grid and u_new_grid.")
    return interp, feature, space

def gen_data_correction(interp, dname, isplot):
    x_random = np.loadtxt(f"{dname}/u_new.dat")[:, 0]
    f_0  = np.sin(2 * np.pi * x_random)
    u_0 =  interp(x_random)
    np.savetxt(f"{dname}/f_init.dat", np.column_stack((x_random, f_0)))
    np.savetxt(f"{dname}/u_init.dat", np.column_stack((x_random, u_0)))
    if isplot:
        plot_data("f_init", "u_init", dname)
    return

def plot_data(fname, uname, dname):
    x = np.loadtxt(f"{dname}/{fname}.dat")[:, 0]
    f = np.loadtxt(f"{dname}/{fname}.dat")[:, 1]
    u = np.loadtxt(f"{dname}/{uname}.dat")[:, 1]
    plt.figure()
    plt.plot(x, f)
    plt.xlabel("x")
    plt.ylabel("{}".format(fname))
    plt.savefig(f"{dname}/{fname}.png")
    plt.show()

    plt.figure()
    plt.plot(x, u)
    plt.xlabel("x")
    plt.ylabel("{}".format(uname))
    plt.savefig(f"{dname}/{uname}.png")
    plt.show()

def construct_data(f, u):
    N = len(f)
    inputs = []
    outputs = []
    for i in range(1, N - 1):
        inputs.append([u[i - 1], u[i + 1], f[i]])
        outputs.append([u[i]])
    return np.array(inputs), np.array(outputs)

def construct_more_data(N, f, u):
    M = len(f) 
    step = int((M-1)/(N-1))
    outputs = u[step: -step].reshape((-1, 1)) 
    inputs = np.hstack((u[:-step*2].reshape((-1, 1)), u[2*step:].reshape((-1, 1)), f[step:-step].reshape((-1, 1))))
    return inputs, outputs

def gen_all_data(M, N, N_f, N_b, l, a, l_new, a_new, dname, gen = False, correction = False, grid = False, isplot = False):
    # Define the domain
    geom = dde.geometry.Interval(0, 1)
    if gen:
        print("Generate new datasets ... ")
        gen_data_GRF(M, N, l, a, geom, dname, isplot)
        interp_u0 = gen_test_data(M, N, geom, dname, isplot)
        gen_new_data_GRF(M, N, N_f, N_b, geom, dname, a_new, l_new, isplot)
        gen_data_correction(interp_u0, dname, isplot)

def gen_datasets(sigma, num_func, parent_dir = "./", gen = True):
    M = 1001 # Number of points 
    N = 101
    N_f = 101
    N_b = 0
    l, a = 0.01, 0.5
    l_new, a_new = 0.1, float(sigma)

    # Create folders for the datasets
    new_dir = "data_1d_poisson"
    PATH = os.path.join(parent_dir, new_dir)
    os.makedirs(f"{PATH}/data_{sigma}", exist_ok = True)

    for i in range(num_func):
        print("Dataset {}".format(i))
        dname = f"{PATH}/data_{sigma}/data_{i}"
        os.makedirs(dname, exist_ok = True)

        # Generate datasets
        gen_all_data(M, N, N_f, N_b, l, a, l_new, a_new, dname, gen, 
                                        correction = False, grid = True, isplot = False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num", type=int, default=3) # Number of functions
    parser.add_argument("--sigma", type=str, default="0.10") # Amplitude in the GRF
    args = parser.parse_args()
    gen_datasets(args.sigma, args.num)