import matplotlib.pyplot as plt
import os
import numpy as np

def sample_points(Nx, Ny, N_f, N_b):
    import deepxde as dde
    hx = 1/(Nx-1)
    hy = 1/(Ny-1)
    geom = dde.geometry.Rectangle([0 + hx, 0 + hy], [1 - hx, 1 - hy])
    N_f = N_f - Nx*2 - Ny*2 + 4
    x_random = geom.random_points(N_f, random = "Hammersley")

    x_f = []
    for i in x_random:
        if 0 + hx <= i[0] <= 1 - hx and 0 + hy <= i[1] <= 1 - hy:
            #print(i)
            x_f.append(i)
    print("Removed {} point(s).".format(len(x_random) - len(x_f)))
    if N_b == 0:
        x_random  = np.array(x_f)
    else:
        x_b = geom.random_boundary_points(N_b, random = "Hammersley")
        x_random = np.concatenate((x_f, x_b), axis = 0)

    return x_random

M = 201
Nx, Ny = 101, 101
l, a = 0.1, 0.1
l_new = 0.10
N_f = 101*101 
N_b = 0
a_new = 0.05  # std for delta f
std_num= "0.05"
for i in range(100):
    d_num = "_{}/data_{}".format(std_num, i) # dataset number
    os.makedirs("data_{}/data_{}".format(std_num, i), exist_ok = True)
    x_random = sample_points(Nx, Ny, N_f, N_b)
    np.savetxt("data{}/x_random.dat".format(d_num), x_random)