import deepxde as dde
import numpy as np
import matplotlib.pyplot as plt
from spaces import GRF
from scipy import interpolate
from porous_media_solver import solver

def compute_numerical_solution(f, Nx, Nt):
    xmin, xmax = 0, 1
    tmin, tmax = 0, 1
    D = 0.01
    k = 1
    Ca0 = lambda x: np.exp(-20 * x)
    Cb0 = lambda x: np.exp(-20 * x)
    x, t, Ca, Cb = solver(xmin, xmax, tmin, tmax, D, k, f, Ca0, Cb0, Nx, Nt)
    return x, t, Ca, Cb

def gen_data_GRF(M, Nx, Nt, l, a, d_num):
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

    np.savetxt("data{}/x_grid.dat".format(d_num), np.rot90(np.tile(x_grid, (Nx, 1)), k = 3))
    np.savetxt("data{}/t_grid.dat".format(d_num), np.tile(t_grid, (Nt, 1)))
    np.savetxt("data{}/f_T_grid.dat".format(d_num), f_T_grid)
    np.savetxt("data{}/Ca_T_grid.dat".format(d_num), Ca_T_grid)
    np.savetxt("data{}/Cb_T_grid.dat".format(d_num), Cb_T_grid)
    print("Generated f_T_grid, Ca_T_grid and Cb_T_grid.")
    # plot_data("f_T_grid", "Ca_T_grid", d_num)
    # plot_data("f_T_grid", "Cb_T_grid", d_num)

    np.savetxt("data{}/x.dat".format(d_num), np.rot90(np.tile(x, (M, 1)), k = 3))
    np.savetxt("data{}/t.dat".format(d_num), np.tile(t, (M, 1)))
    np.savetxt("data{}/f_T.dat".format(d_num), f_T)
    np.savetxt("data{}/Ca_T.dat".format(d_num), Ca_T)
    np.savetxt("data{}/Cb_T.dat".format(d_num), Cb_T)

    print("Generated f_T, Ca_T and Cb_T.")
    # plot_data("f_T", "Ca_T", d_num)
    # plot_data("f_T", "Cb_T", d_num)
    return

def gen_test_data(M, Nx, Nt, d_num):
    f0 = lambda x: np.exp(-((x - 0.5) ** 2) / 0.05)
    f_0 = lambda x: f0(x)
    x, t, Ca_0, Cb_0 =  compute_numerical_solution(f_0, M, M)
    f_0 = np.tile(f_0(x)[:, None], (1, len(t)))
    
    np.savetxt("data{}/f_0.dat".format(d_num), f_0)
    np.savetxt("data{}/Ca_0.dat".format(d_num), Ca_0)
    np.savetxt("data{}/Cb_0.dat".format(d_num), Cb_0)

    # plot_data("f_0", "Ca_0", d_num)
    # plot_data("f_0", "Cb_0", d_num)

    interp_a = interpolate.interp2d(t, x, Ca_0, kind='cubic')
    interp_b = interpolate.interp2d(t, x, Cb_0, kind='cubic')

    f_0_grid = f_0[::round(M/Nx), ::round(M/Nt)]
    Ca_0_grid = Ca_0[::round(M/Nx), ::round(M/Nt)]
    Cb_0_grid = Cb_0[::round(M/Nx), ::round(M/Nt)]

    np.savetxt("data{}/f_0_grid.dat".format(d_num), f_0_grid)
    np.savetxt("data{}/Ca_0_grid.dat".format(d_num), Ca_0_grid)
    np.savetxt("data{}/Cb_0_grid.dat".format(d_num), Cb_0_grid)
    # plot_data("f_0_grid", "Ca_0_grid", d_num)
    # plot_data("f_0_grid", "Cb_0_grid", d_num)

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

def gen_new_data_exact(M, Nx, Nt, a_new, l_new, d_num):
    space = GRF(1, length_scale=l_new, N=1001, interp="cubic")
    feature = space.random(1)
    f0 = lambda x: np.exp(-((x - 0.5) ** 2) / 0.05)
    fi = lambda x: f0(x)+ a_new * np.ravel(space.eval_u(feature, x))
    xi, ti, Cai, Cbi = compute_numerical_solution(fi,1001, 1001)
    fi = np.tile(fi(xi)[:, None], (1, len(ti)))
    np.savetxt("data{}/fnew_1001.dat".format(d_num), fi)
    np.savetxt("data{}/Canew_1001.dat".format(d_num), Cai)
    np.savetxt("data{}/Cbnew_1001.dat".format(d_num), Cbi)
    interp_a = interpolate.interp2d(ti, xi, Cai, kind = "cubic")
    interp_b = interpolate.interp2d(ti, xi, Cbi, kind = "cubic")


    fi_grid = fi[::round(M/Nx), ::round(M/Nt)]
    Cai_grid = Cai[::round(M/Nx), ::round(M/Nt)]
    Cbi_grid = Cbi[::round(M/Nx), ::round(M/Nt)]
    np.savetxt("data{}/f_new_grid.dat".format(d_num), fi_grid)
    np.savetxt("data{}/Ca_new_grid.dat".format(d_num), Cai_grid)
    np.savetxt("data{}/Cb_new_grid.dat".format(d_num), Cbi_grid)

    print("Generated f_new_grid, Ca_new_grid and Cb_new_grid.")
    plot_data("f_new_grid", "Ca_new_grid", d_num)
    plot_data("f_new_grid", "Cb_new_grid", d_num)
    return interp_a, interp_b, feature, space

def gen_new_data_GRF(M, Nx, Nt, N_f, N_b, d_num, a_new, l_new):
    hx = 1/(Nx-1)
    ht = 1/(Nt-1)
    geom = dde.geometry.Interval(0 + hx, 1 - hx)
    timedomain = dde.geometry.TimeDomain(0 + ht, 1)
    geomtime = dde.geometry.GeometryXTime(geom, timedomain)
    x_random = sample_points(geomtime, Nx, Nt, N_f, N_b)
    interp_a, interp_b, feature, space = gen_new_data_exact(M, Nx, Nt, a_new, l_new, d_num)
    f0 = lambda x: np.exp(-((x - 0.5) ** 2) / 0.05)
    f_new = lambda x: f0(x) + a_new * np.ravel(space.eval_u(feature, x))
    f_new = f_new(x_random[:, 0]).reshape((-1, 1))
    Ca_new =  np.array([interp_a(i[1], i[0]) for i in x_random]).reshape((-1, 1))
    Cb_new =  np.array([interp_b(i[1], i[0]) for i in x_random]).reshape((-1, 1))

    np.savetxt("data{}/f_new.dat".format(d_num), np.concatenate((x_random, f_new), axis = 1))
    np.savetxt("data{}/Ca_new.dat".format(d_num), np.concatenate((x_random, Ca_new), axis = 1))
    np.savetxt("data{}/Cb_new.dat".format(d_num), np.concatenate((x_random, Cb_new), axis = 1))
    print("Generated f_new, Ca_new and Cb_new.")
    return

def gen_data_correction(interp_a, interp_b, d_num):
    f0 = lambda x: np.exp(-((x - 0.5) ** 2) / 0.05)
    x_random = np.loadtxt("data{}/Ca_new.dat".format(d_num))[:, 0:2]
    f_0 = f0(x_random[:, 0]).reshape((-1, 1))
    Ca_0 =  np.array([interp_a(i[1], i[0]) for i in x_random]).reshape((-1, 1))
    Cb_0 =  np.array([interp_b(i[1], i[0]) for i in x_random]).reshape((-1, 1))

    np.savetxt("data{}/f_init.dat".format(d_num), np.concatenate((x_random, f_0), axis = 1))
    np.savetxt("data{}/Ca_init.dat".format(d_num), np.concatenate((x_random, Ca_0), axis = 1))
    np.savetxt("data{}/Cb_init.dat".format(d_num), np.concatenate((x_random, Cb_0), axis = 1))

    print("Generated f_init, Ca_init and Cb_init.")
    return

def plot_data(f_name, u_name, d_num):
    f = np.loadtxt("data{}/{}.dat".format(d_num,f_name))
    u = np.loadtxt("data{}/{}.dat".format(d_num, u_name))
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

def gen_all_data(M, Nx, Nt, N_f, N_b, l, a, l_new, a_new, gen, d_num, correction = False):
    if gen:
        print("Generate new dataset ... ")
        gen_data_GRF(M, Nx, Nt, l, a, d_num)
        interp_Ca0,interp_Cb0 = gen_test_data(M, Nx, Nt, d_num)
        gen_new_data_GRF(M, Nx, Nt, N_f, N_b, d_num, a_new, l_new)
        gen_data_correction(interp_Ca0, interp_Cb0, d_num)

    # training data
    f_T = np.loadtxt("data{}/f_T.dat".format(d_num))
    Ca_T = np.loadtxt("data{}/Ca_T.dat".format(d_num))
    Cb_T = np.loadtxt("data{}/Cb_T.dat".format(d_num))
    d_T = construct_more_data(Nx, Nt, f_T, Ca_T, Cb_T)
    # test data
    f_0 = np.loadtxt("data{}/f_0.dat".format(d_num))
    Ca_0 = np.loadtxt("data{}/Ca_0.dat".format(d_num))
    Cb_0 = np.loadtxt("data{}/Cb_0.dat".format(d_num))
    d_0 = construct_more_data(Nx, Nt, f_0, Ca_0, Cb_0)

    # For the 2nd stage
    x_train = np.loadtxt("data{}/Ca_new.dat".format(d_num))[:, 0:2]
    y_train = np.concatenate(([[0] * len(x_train)])).reshape((-1, 1))
    x = np.loadtxt("data{}/x_grid.dat".format(d_num)).reshape((-1, 1))
    t = np.loadtxt("data{}/t_grid.dat".format(d_num)).reshape((-1, 1))
    x_test = np.concatenate((x, t), axis = 1)

    if correction:
        Ca_new = np.loadtxt("data{}/Ca_new_grid.dat".format(d_num)).reshape((-1, 1))
        Cb_new = np.loadtxt("data{}/Cb_new_grid.dat".format(d_num)).reshape((-1, 1))
        Ca_init = np.loadtxt("data{}/Ca_0_grid.dat".format(d_num)).reshape((-1, 1))
        Cb_init = np.loadtxt("data{}/Cb_0_grid.dat".format(d_num)).reshape((-1, 1))
        ya_test = Ca_new - Ca_init
        yb_test = Cb_new - Cb_init
    else:
        ya_test = np.loadtxt("data{}/Ca_new_grid.dat".format(d_num)).reshape((-1, 1))
        yb_test = np.loadtxt("data{}/Cb_new_grid.dat".format(d_num)).reshape((-1, 1))

    data_G = dde.data.DataSet(
        X_train=d_T[0],
        y_train=np.hstack((d_T[1], d_T[2])),
        X_test=d_0[0],
        y_test=np.hstack((d_0[1], d_0[2])),
    )
    data = dde.data.DataSet(
        X_train=x_train, y_train=np.hstack((y_train, y_train)), 
        X_test=x_test, y_test=np.hstack((ya_test, yb_test)))
    return data_G, data
