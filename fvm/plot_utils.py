import numpy
import matplotlib.pyplot as plt

from fvm.utils import create_state_mtx # noqa: F401

def plot_state(u, v, nx, ny, x=None, y=None):
    psi = numpy.zeros([nx, ny])

    for i in range(nx):
        for j in range(ny):
            psiu = u[i, j]
            if j > 0:
                psiu += u[i, j-1]
            psiv = v[i, j]
            if i > 0:
                psiv += v[i-1, j]
            psi[i, j] = numpy.linalg.norm([psiu, psiv])

    if x is None:
        x = numpy.arange(1/nx, 1+1/nx, 1/nx)
        y = numpy.arange(1/ny, 1+1/ny, 1/ny)

    x, y = numpy.meshgrid(x, y)

    fig1, ax1 = plt.subplots()
    cs = ax1.contourf(x, y, psi.transpose(), 15)
    fig1.colorbar(cs)

    ax1.vlines(x[0], *y[[0, -1], 0], colors='0.3', linewidths=0.5)
    ax1.hlines(y[:, 0], *x[0, [0, -1]], colors='0.3', linewidths=0.5)

    plt.show()
