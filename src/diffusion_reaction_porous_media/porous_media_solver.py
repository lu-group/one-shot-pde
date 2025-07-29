import numpy as np


def solver(xmin, xmax, tmin, tmax, D, k, f, Ca0, Cb0, Nx, Nt):
    """Solve
    Ca_t = D * Ca_xx - k Ca Cb^2 + f(x)
    Cb_t = D * Cb_xx - 2k Ca Cb^2
    IC: Ca0, Cb0
    BC: Ca(xmin, t) = Cb(xmin, t) = 1
        Ca(xmax, t) = Cb(xmax, t) = 0

    Backward Euler, explicit in the reaction term
    """
    x = np.linspace(xmin, xmax, Nx)
    t = np.linspace(tmin, tmax, Nt)
    h = x[1] - x[0]
    dt = t[1] - t[0]
    h2 = h ** 2

    D2 = 2 * np.eye(Nx) - np.eye(Nx, k=-1) - np.eye(Nx, k=1)
    A = D * D2 + h2 / dt * np.eye(Nx)
    A[0, 0] = A[-1, -1] = 1
    A[0, 1] = A[-1, -2] = 0
    f = f(x)

    Ca = np.zeros((Nx, Nt))
    Cb = np.zeros((Nx, Nt))
    Ca[:, 0] = Ca0(x)
    Cb[:, 0] = Cb0(x)
    for i in range(Nt - 1):
        b1 = h2 / dt * Ca[:, i]
        b2 = -h2 * k * Ca[:, i] * Cb[:, i] ** 2
        b3 = h2 * f
        b = b1 + b2 + b3
        b[0] = 1
        b[-1] = 0
        Ca[:, i + 1] = np.linalg.solve(A, b)

        b1 = h2 / dt * Cb[:, i]
        b2 *= 2
        b3 *= 2
        b = b1 + b2 + b3
        b[0] = 1
        b[-1] = 0
        Cb[:, i + 1] = np.linalg.solve(A, b)
    return x, t, Ca, Cb


def main():
    xmin, xmax = 0, 1
    tmin, tmax = 0, 1
    D = 0.01
    k = 1
    Ca0 = lambda x: np.exp(-20 * x)
    Cb0 = lambda x: np.exp(-20 * x)

    f0 = lambda x: np.exp(-((x - 0.5) ** 2) / 0.05)
    # Random f
    rn = 2 * np.random.rand(101)
    f = lambda x: rn
    # f0
    # f = f0
    # New f
    # f = lambda x: f0(x) + 0.1

    Nx, Nt = 101, 101
    x, t, Ca, Cb = solver(xmin, xmax, tmin, tmax, D, k, f, Ca0, Cb0, Nx, Nt)
    # np.savetxt("data/f_random.dat", f(x)[:, None] + 0 * t)
    # np.savetxt("data/u1_random.dat", Ca)
    # np.savetxt("data/u2_random.dat", Cb)


if __name__ == "__main__":
    main()

