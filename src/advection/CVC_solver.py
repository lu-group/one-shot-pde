from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt

from spaces import GRF


def solve_CVC(xmin, xmax, tmin, tmax, f, g, V, Nx, Nt):
    """Solve
    u_t + a(x) * u_x = 0
    """
    # Case III: Wendroff for a(x)=1+0.1*V(x), u(x,0)=f(x), u(0,t)=g(t)    (f(0)=g(0))
    x = np.linspace(xmin, xmax, Nx)
    t = np.linspace(tmin, tmax, Nt)
    h = x[1] - x[0]
    dt = t[1] - t[0]
    lam = dt / h
    v = 1 + 0.1 * V(x)
    u = np.zeros([Nx, Nt])
    u[:, 0] = f(x)
    u[0, :] = g(t)
    a = (v[:-1] + v[1:]) / 2
    k = (1 - a * lam) / (1 + a * lam)
    K = np.eye(Nx - 1, k=0)
    K_temp = np.eye(Nx - 1, k=0)
    Trans = np.eye(Nx - 1, k=-1)
    for _ in range(Nx - 2):
        K_temp = (-k[:, None]) * (Trans @ K_temp)
        K += K_temp
    D = np.diag(k) + np.eye(Nx - 1, k=-1)

    for n in range(Nt - 1):
        b = np.zeros(Nx - 1)
        b[0] = g(n * dt) - k[0] * g((n + 1) * dt)
        u[1:, n + 1] = K @ (D @ u[1:, n] + b)

    return x, t, u


def main():
    # Case III: Wendroff for a(x)=1+0.1*V(x), u(x,0)=f(x), u(0,t)=g(t)    (f(0)=g(0))
    # Random f
    rn = 2 * np.random.rand(101) - 1
    V = lambda x: rn
    # f0
    # V = lambda x: 0 * x
    # f
    #V = lambda x: 0 * x + 0.5
    # New random f
    #space = GRF(1, length_scale=0.1, N=1000, interp="cubic")
    #features = space.random(1)
    #V = lambda x: 2 * space.eval_u(features, x[:, None])[0]

    xmin, xmax = 0, 1
    tmin, tmax = 0, 1
    f = lambda x: x ** 2
    g = lambda t: np.sin(np.pi * t)

    Nx, Nt = 101, 101
    x, t, u = solve_CVC(xmin, xmax, tmin, tmax, f, g, V, Nx, Nt)

    f = np.tile(V(x)[:, None], (1, len(t)))
    np.savetxt("f.dat", f)
    np.savetxt("u.dat", u)

    plt.imshow(f, cmap = "gnuplot", extent=(0,1,0,1), aspect='auto') 
    plt.colorbar()
    plt.xlabel("x")
    plt.ylabel("t")
    plt.show()

    plt.imshow(u, cmap = "gnuplot", extent=(0,1,0,1), aspect='auto') 
    plt.colorbar()
    plt.xlabel("x")
    plt.ylabel("t")
    plt.show()

    inputs = []
    outputs = []
    for i in range(1, Nx - 1):
        for j in range(1, Nt):
            # inputs.append([f[i, j], u[i, j - 1], u[i - 1, j], u[i + 1, j]])
            inputs.append(
                [
                    f[i, j],
                    u[i - 1, j],
                    u[i - 1, j - 1],
                    u[i, j - 1],
                    # u[i + 1, j - 1],
                    # u[i + 1, j],
                ]
            )
            outputs.append([u[i, j]])
    # np.savetxt("data_f0.dat", np.hstack((inputs, outputs)))


if __name__ == "__main__":
    main()


