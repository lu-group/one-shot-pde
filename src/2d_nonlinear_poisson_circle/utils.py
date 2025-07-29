import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from spaces import GRF2D
import deepxde as dde
import numpy as np
import matplotlib.pyplot as plt
dde.backend.set_default_backend('tensorflow.compat.v1')
dde.config.set_default_float("float64")

def construct_data(Nx, Ny, f, u):
    f = f.reshape((Nx, Ny))
    u = u.reshape((Nx, Ny))
    # u[i-1, j], u[i, j-1], u[i+1, j], u[i, j+1], f[i, j]
    N = len(f)
    xstep = 1
    ystep = 1
    outputs = u[ystep: -ystep, xstep:-xstep].reshape((-1, 1))
    naninds_out = np.argwhere(np.isnan(outputs))[:, 0]
    inputs = np.hstack((u[ystep:-ystep, :-2*xstep].reshape((-1, 1)), u[:-2*ystep, xstep:-xstep].reshape((-1, 1)),u[ystep: -ystep, 2*xstep:].reshape((-1, 1)), u[2*ystep: , xstep:-xstep].reshape((-1, 1)),f[ystep: -ystep, xstep:-xstep].reshape((-1, 1))))
    naninds_in = np.argwhere(np.isnan(inputs))[:, 0]
    inds = [i for i in range(len(outputs)) if i not in naninds_out and i not in naninds_in]
    inputs = inputs[inds]
    outputs = outputs[inds]
    return inputs, outputs

def load_all_data(Nx, Ny, dname, PATH, correction, isplot=False, issave=False):
    data_T_grid = np.loadtxt(f"{dname}/f_T_grid.txt",skiprows=9)
    x_T = data_T_grid[:,2].reshape((-1, 1))
    y_T = data_T_grid[:,3].reshape((-1, 1))
    u_T = data_T_grid[:,4].reshape((-1, 1))
    f_T = data_T_grid[:,5].reshape((-1, 1))
    d_T = construct_data(Nx, Ny, f_T, u_T)
    if isplot:
        scatter_plot_data(x_T, y_T, f_T, u_T)

    # load test data
    data_0_grid = np.loadtxt(f"{PATH}/data_0.05/data_0/f_0_grid.txt",skiprows=9)
    x_0 = data_0_grid[:,2].reshape((-1, 1))
    y_0 = data_0_grid[:,3].reshape((-1, 1))
    u_0 = data_0_grid[:,4].reshape((-1, 1))
    f_0 = data_0_grid[:,5].reshape((-1, 1))
    d_0 = construct_data(Nx, Ny, f_0, u_0)
    if isplot:
        scatter_plot_data(x_0, y_0, f_0, u_0)
    
    data_G = dde.data.DataSet(X_train=d_T[0], y_train=d_T[1], X_test=d_0[0], y_test=d_0[1])

    # For the 2nd stage
    data_new_grid = np.loadtxt(f"{dname}/f_new_grid.txt",skiprows=9)
    inds = [i for i in range(len(data_new_grid)) if i not in np.argwhere(np.isnan(data_new_grid))[:, 0]]
    data_new_grid = data_new_grid[inds]
    hx, hy = 1/Nx, 1/Ny
    if issave:
        np.savetxt(f"{dname}/data_new_grid.txt", data_new_grid)
        np.savetxt(f"{PATH}/data_0.05/data_0/data_0_grid.txt", data_0_grid[inds])
        np.savetxt(f"{dname}/data_T_grid.txt", data_T_grid[inds])
    x_newg = data_new_grid[:,2].reshape((-1, 1))
    y_newg = data_new_grid[:,3].reshape((-1, 1))
    u_newg = data_new_grid[:,4].reshape((-1, 1))
    f_newg = data_new_grid[:,5].reshape((-1, 1))
    if isplot:
        scatter_plot_data(x_newg, y_newg, f_newg, u_newg)

    data_new = np.loadtxt(f"{dname}/f_new.txt",skiprows=9)
    inds_b = [i for i in range(len(data_new)) 
              if ((data_new[i, 2] - hx - 0.5)**2 + (data_new[i, 3] - 0.5)**2 > 0.21**2)
              and ((data_new[i, 2] - 0.5)**2 + (data_new[i, 3] - hy - 0.5)**2 > 0.21**2)
              and ((data_new[i, 2] + hx - 0.5)**2 + (data_new[i, 3] - 0.5)**2 > 0.21**2)
              and ((data_new[i, 2] - 0.5)**2 + (data_new[i, 3] + hy - 0.5)**2 > 0.21**2)
              and ((data_new[i, 2] - 0.5)**2 + (data_new[i, 3] - 0.5)**2 > 0.21**2)
              and (0<data_new[i, 2] - hx<1) and (0<data_new[i, 2] + hx<1) and (0<data_new[i, 3] - hy<1) and (0<data_new[i, 3] + hy<1)]
    data_new = data_new[inds_b]
    data_0 = np.loadtxt(f"{PATH}/data_0.05/data_0/f_0.txt",skiprows=9)[inds_b]
    if issave:
        np.savetxt(f"{dname}/data_new.txt", data_new)
        np.savetxt(f"{PATH}/data_0.05/data_0/data_0.txt", data_0)
    x_new = data_new[:,2].reshape((-1, 1))
    y_new = data_new[:,3].reshape((-1, 1))
    u_new = data_new[:,4].reshape((-1, 1))
    f_new = data_new[:,5].reshape((-1, 1))
    if isplot:
        scatter_plot_data(x_new, y_new, f_new, u_new)

    x_train = np.concatenate((x_new, y_new), axis = 1)
    y_train = np.concatenate(([[0] * (len(x_train))])).reshape((-1, 1))
    x_test = np.concatenate((x_newg, y_newg), axis = 1)

    if correction:
        y_test = u_newg - u_0[inds]
    else:
        y_test = u_newg

    data = dde.data.DataSet(X_train=x_train, y_train=y_train, X_test=x_test, y_test=y_test)
    return data_G, data

def scatter_plot_data(x, y, f, u):
    plt.figure()
    plt.scatter(x, y, c=f, cmap = "rainbow")
    plt.colorbar()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()

    plt.figure()
    plt.scatter(x, y, c=u, cmap = "rainbow")
    plt.colorbar()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()


def gen_f_GRF(Nx, Ny, l, a, dname):
    '''
    Generate training data (f_T, u_T) from GRF.
    '''
    x = np.linspace(0, 1, num=Nx)
    y = np.linspace(0, 1, num=Ny)
    # define f0
    f0 = x[:, None] * np.sin(y)
    space = GRF2D(length_scale=l, N=101)
    feature = space.random(1)
    xv, yv = np.meshgrid(x, y)
    f = space.eval_u(feature, np.vstack((np.ravel(xv), np.ravel(yv))).T)[0]
    f = np.reshape(f, (len(y), len(x))).T
    f = f0 + a * f
    x_grid = np.rot90(np.tile(x, (Nx, 1)), k = 3).reshape((-1, 1))
    y_grid = np.tile(y, (Ny, 1)).reshape((-1, 1))
    f_T_grid  = f.reshape((-1, 1))
    np.savetxt(f"{dname}/data_T_grid.dat", np.hstack((x_grid, y_grid, f_T_grid)))
    print("Generated f_T_grid.")

    plt.figure()
    plt.rcParams.update({'font.size': 20,"savefig.dpi": 200, "figure.figsize": (8, 6)})
    plt.subplots_adjust(left=0.15, right=0.9, top=0.9, bottom=0.15)
    plt.imshow(np.rot90(f), cmap = "rainbow",  origin='upper', extent=(0,1,0,1), aspect='auto')
    plt.colorbar()
    plt.xlabel("x")
    plt.ylabel("t")
    plt.savefig(f"{dname}/f_T_grid.png")
    plt.show()

def gen_new_f_GRF(Nx, Ny, a_new, l_new, dname):
    # points in the domain
    x = np.linspace(0, 1, num=Nx)
    y = np.linspace(0, 1, num=Ny)
    xv, yv = np.meshgrid(x, y)
    space = GRF2D(length_scale=l_new, N=101)
    feature = space.random(1)
    f0 = x[:, None] * np.sin(y)
    f_new = space.eval_u(feature, np.vstack((np.ravel(xv), np.ravel(yv))).T)[0]
    f_new = np.reshape(f_new, (len(y), len(x))).T
    f_new = f0 + a_new * f_new
    
    x_grid = np.rot90(np.tile(x, (Nx, 1)), k = 3).reshape((-1, 1))
    y_grid = np.tile(y, (Ny, 1)).reshape((-1, 1))
    f_new_grid  = f_new.reshape((-1, 1))

    np.savetxt(f"{dname}/data_new_grid.dat", np.hstack((x_grid, y_grid, f_new_grid)))
    print("Generated f_new_grid.")

    plt.figure()
    plt.rcParams.update({'font.size': 20,"savefig.dpi": 200, "figure.figsize": (8, 6)})
    plt.subplots_adjust(left=0.15, right=0.9, top=0.9, bottom=0.15)
    plt.imshow(np.rot90(f_new), cmap = "rainbow",  origin='upper', extent=(0,1,0,1), aspect='auto')
    plt.colorbar()
    plt.xlabel("x")
    plt.ylabel("t")
    plt.savefig(f"{dname}/f_new_grid.png")
    plt.show()
    return    

def generate_random_points(N_f, dname):
    import deepxde as dde
    geom1 = dde.geometry.Rectangle((0,0),(1,1))
    geom2 = dde.geometry.Disk((0.5,0.5), 0.2)
    geom = dde.geometry.CSGDifference(geom1, geom2)
    x_random = geom.random_points(N_f, random = "Hammersley")
    np.savetxt(f"{dname}/x_random.txt", x_random)


def gen_datasets(sigma, num_func, parent_dir = "./", gen = True):
    Nx = Ny = 100
    l, a = 0.05, 0.10
    l_new, a_new = 0.1, float(sigma)

    # Create folders for the datasets
    new_dir = "data_nonlinear_poisson_circle"
    PATH = os.path.join(parent_dir, new_dir)
    os.makedirs(f"{PATH}/data_{sigma}", exist_ok = True)
    
    # Generate x,y grid and f
    if gen:
        for i in range(num_func):
            print("Dataset {}".format(i))
            dname = f"{PATH}/data_{sigma}/data_{i}"
            os.makedirs(dname, exist_ok = True)

            #generate_random_points(100*100, dname)
            gen_f_GRF(1000, 1000, l, a, dname)
            gen_new_f_GRF(1000, 1000, a_new, l_new, dname)
    
    # Then load the data files to FEM solver
    else:
        for i in range(num_func):
            print("Dataset {}".format(i))
            dname = f"{PATH}/data_{sigma}/data_{i}"
            os.makedirs(dname, exist_ok = True)

            # Load and Generate datasets
            load_all_data(Nx, Ny, dname, PATH, correction=False, isplot=True, issave=True)
    return


