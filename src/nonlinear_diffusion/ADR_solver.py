from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np

from spaces import GRF


def solve_ADR(xmin, xmax, tmin, tmax, k, v, g, dg, f, u0, Nx, Nt):
    """Solve 1D
    u_t = (k(x) u_x)_x - v(x) u_x + g(u) + f(x, t)
    with zero boundary condition.
    """

    x = np.linspace(xmin, xmax, Nx)
    t = np.linspace(tmin, tmax, Nt)
    h = x[1] - x[0]
    dt = t[1] - t[0]
    h2 = h ** 2

    D1 = np.eye(Nx, k=1) - np.eye(Nx, k=-1)
    D2 = -2 * np.eye(Nx) + np.eye(Nx, k=-1) + np.eye(Nx, k=1)
    D3 = np.eye(Nx - 2)
    k = k(x)
    M = -np.diag(D1 @ k) @ D1 - 4 * np.diag(k) @ D2
    m_bond = 8 * h2 / dt * D3 + M[1:-1, 1:-1]
    v = v(x)
    v_bond = 2 * h * np.diag(v[1:-1]) @ D1[1:-1, 1:-1] + 2 * h * np.diag(
        v[2:] - v[: Nx - 2]
    )
    mv_bond = m_bond + v_bond
    c = 8 * h2 / dt * D3 - M[1:-1, 1:-1] - v_bond
    f = f(x[:, None], t)

    u = np.zeros((Nx, Nt))
    u[:, 0] = u0(x)
    for i in range(Nt - 1):
        gi = g(u[1:-1, i])
        dgi = dg(u[1:-1, i])
        h2dgi = np.diag(4 * h2 * dgi)
        A = mv_bond - h2dgi
        b1 = 8 * h2 * (0.5 * f[1:-1, i] + 0.5 * f[1:-1, i + 1] + gi)
        b2 = (c - h2dgi) @ u[1:-1, i].T
        u[1:-1, i + 1] = np.linalg.solve(A, b1 + b2)
    return x, t, u


def dr():
    xmin, xmax = 0, 1
    tmin, tmax = 0, 1
    k = lambda x: 0.01 * np.ones_like(x)
    v = lambda x: np.zeros_like(x)
    # No reaction
    # g = lambda u: np.zeros_like(u)
    # dg = lambda u: np.zeros_like(u)
    # Reaction
    g = lambda u: 0.01 * u ** 2
    dg = lambda u: 0.02 * u
    u0 = lambda x: np.zeros_like(x)

    f0 = lambda x: 0.9 * np.sin(2 * np.pi * x)
    # Random f
    rn = 2 * np.random.rand(101, 1) - 1
    f = lambda x, t: np.tile(rn, (1, len(t))) + f0(x)
    # f0
    # f = lambda x, t: f0(x) + 0 * t
    # New f
    # f = lambda x, t: f0(x) + 0.05 + 0 * t
    # New random f
    # space = GRF(1, length_scale=0.1, N=1000, interp="cubic")
    # features = space.random(1)
    # f = lambda x, t: f0(x) + 0.05 * space.eval_u(features, x).T + 0 * t

    Nx, Nt = 101, 101
    x, t, u = solve_ADR(xmin, xmax, tmin, tmax, k, v, g, dg, f, u0, Nx, Nt)
    f = f(x[:, None], t)
    np.savetxt("f_random.dat", f)
    np.savetxt("u_random.dat", u)

    # plt.plot(x, f0(x), label="f0")
    # plt.plot(x, f[:, 0], label="f")
    # plt.legend()
    # plt.show()
    # x = np.repeat(x, Nt)
    # t = np.tile(t, Nx)
    # f = np.repeat(f, Nt)
    # u = np.ravel(u)
    # np.savetxt('data.dat', np.vstack((x, t, f, u)).T)


def diffusion_nonhomogeneous():
    xmin, xmax = 0, 1
    tmin, tmax = 0, 1
    v = lambda x: np.zeros_like(x)
    g = lambda u: np.zeros_like(u)
    dg = lambda u: np.zeros_like(u)
    # f = lambda x, t: np.sin(2 * np.pi * x) + 0 * t
    # u0 = lambda x: np.zeros_like(x)
    f = lambda x, t: 0 * x * t
    u0 = lambda x: np.sin(2 * np.pi * x)

    k0 = lambda x: np.ones_like(x)
    # Random f
    rn = 2 * np.random.rand(101) + 0.5
    k = lambda x: rn
    # f0
    # k = k0
    # New f
    # k = lambda x: k0(x) + 0.5
    # New random f
    # space = GRF(1, length_scale=0.1, N=1000, interp="cubic")
    # features = space.random(1)
    # k = lambda x: np.abs(k0(x) + 0.3 * np.ravel(space.eval_u(features, x)))

    k1 = lambda x: 0.01 * k(x)
    Nx, Nt = 101, 101
    x, t, u = solve_ADR(xmin, xmax, tmin, tmax, k1, v, g, dg, f, u0, Nx, Nt)
    k = k(x)[:, None] + 0 * t
    np.savetxt("f_random.dat", k)
    np.savetxt("u_random.dat", u)


if __name__ == "__main__":
    dr()
    # diffusion_nonhomogeneous()

