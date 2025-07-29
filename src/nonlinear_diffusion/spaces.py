from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np
# from pathos.pools import ProcessPool
from scipy import linalg, interpolate
from sklearn import gaussian_process as gp


class GRF(object):
    """Gaussian random field (GRF) in [0, T]."""

    def __init__(self, T, kernel="RBF", length_scale=1, N=1000, interp="cubic"):
        self.N = N
        self.interp = interp
        self.x = np.linspace(0, T, num=N)[:, None]
        if kernel == "RBF":
            K = gp.kernels.RBF(length_scale=length_scale)
        elif kernel == "AE":
            K = gp.kernels.Matern(length_scale=length_scale, nu=0.5)
        self.K = K(self.x)
        self.L = np.linalg.cholesky(self.K + 1e-13 * np.eye(self.N))

    def random(self, n):
        """Generate `n` random feature vectors."""
        u = np.random.randn(self.N, n)
        return np.dot(self.L, u).T

    def eval_u_one(self, y, x):
        """Compute the function value at `x` for the feature `y`."""
        if self.interp == "linear":
            return np.interp(x, np.ravel(self.x), y)
        f = interpolate.interp1d(
            np.ravel(self.x), y, kind=self.interp, copy=False, assume_sorted=True
        )
        return f(x)

    def eval_u(self, ys, sensors):
        """For a list of functions represented by `ys`,
        compute a list of a list of function values at a list `sensors`.
        """
        if self.interp == "linear":
            return np.vstack([np.interp(sensors, np.ravel(self.x), y).T for y in ys])
        # p = ProcessPool(nodes=config.processes)
        # res = p.map(
        res = map(
            lambda y: interpolate.interp1d(
                np.ravel(self.x), y, kind=self.interp, copy=False, assume_sorted=True
            )(sensors).T,
            ys,
        )
        return np.vstack(list(res))


class GRF2D(object):
    """Gaussian random field (GRF) in [0, 1]x[0, 1]."""

    def __init__(self, kernel="RBF", length_scale=1, N=1000, interp="splinef2d"):
        self.N = N
        self.interp = interp
        self.x = np.linspace(0, 1, num=N)
        self.y = np.linspace(0, 1, num=N)
        xv, yv = np.meshgrid(self.x, self.y)
        self.X = np.vstack((np.ravel(xv), np.ravel(yv))).T
        if kernel == "RBF":
            K = gp.kernels.RBF(length_scale=length_scale)
        elif kernel == "AE":
            K = gp.kernels.Matern(length_scale=length_scale, nu=0.5)
        self.K = K(self.X)
        self.L = np.linalg.cholesky(self.K + 1e-12 * np.eye(self.N ** 2))

    def random(self, n):
        """Generate `n` random feature vectors."""
        u = np.random.randn(self.N ** 2, n)
        return np.dot(self.L, u).T

    def eval_u_one(self, y, x):
        """Compute the function value at `x` for the feature `y`."""
        y = np.reshape(y, (self.N, self.N))
        return interpolate.interpn((self.x, self.y), y, x, method=self.interp)[0]

    def eval_u(self, ys, sensors):
        """For a list of functions represented by `ys`,
        compute a list of a list of function values at a list `sensors`.
        """
        points = (self.x, self.y)
        ys = np.reshape(ys, (-1, self.N, self.N))
        res = map(
            lambda y: interpolate.interpn(points, y, sensors, method=self.interp), ys
        )
        return np.vstack(list(res))


def space_samples(space, T):
    features = space.random(10)
    sensors = np.linspace(0, T, num=1000)
    u = 0.5 * space.eval_u(features, sensors[:, None])
    for ui in u:
        print(max(ui), min(ui))

    # plt.plot(sensors, np.mean(u, axis=0), "k")
    # plt.plot(sensors, np.std(u, axis=0), "k--")
    # plt.plot(sensors, np.cov(u.T)[0], "k--")
    # plt.plot(sensors, np.exp(-0.5 * sensors ** 2 / 0.2 ** 2))
    for ui in u[:5]:
        plt.plot(sensors, ui)
    plt.show()


def space_samples_2d(space):
    features = space.random(3)
    x = np.linspace(0, 1, num=500)
    y = np.linspace(0, 1, num=500)
    xv, yv = np.meshgrid(x, y)
    sensors = np.vstack((np.ravel(xv), np.ravel(yv))).T
    u = space.eval_u(features, sensors)
    for ui in u:
        plt.figure()
        plt.imshow(np.reshape(ui, (len(y), len(x))))
        plt.colorbar()
    plt.show()


def main():
    space = GRF(1, length_scale=0.5, N=1000, interp="cubic")
    space_samples(space, 1)
    #space = GRF2D(length_scale=0.1, N=100)
    #space_samples_2d(space)


if __name__ == "__main__":
    main()

