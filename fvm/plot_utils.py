import numpy
import matplotlib.pyplot as plt

from fvm import utils
from fvm.utils import create_state_mtx # noqa: F401

def plot_velocity_magnitude(u, v, interface=None, x=None, y=None):
    nx = u.shape[0]
    ny = u.shape[1]

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
        x = interface.discretization.x[:-3]
    if y is None:
        y = interface.discretization.y[:-3]

    x, y = numpy.meshgrid(x, y)

    fig1, ax1 = plt.subplots()
    cs = ax1.contourf(x, y, psi.transpose(), 15)
    fig1.colorbar(cs)

    ax1.vlines(x[0, :], *y[[0, -1], 0], colors='0.3', linewidths=0.5)
    ax1.hlines(y[:, 0], *x[0, [0, -1]], colors='0.3', linewidths=0.5)

    plt.show()

def plot_streamfunction(u, v, interface=None, x=None, y=None):
    if x is None:
        x = interface.discretization.x[:-3]
    if y is None:
        y = interface.discretization.y[:-3]

    psi = utils.compute_streamfunction(u, v, x=x, y=y)

    x, y = numpy.meshgrid(x, y)

    fig1, ax1 = plt.subplots()
    cs = ax1.contourf(x, y, psi, 15)
    fig1.colorbar(cs)

    ax1.vlines(x[0, :], *y[[0, -1], 0], colors='0.3', linewidths=0.5)
    ax1.hlines(y[:, 0], *x[0, [0, -1]], colors='0.3', linewidths=0.5)

    plt.show()

def plot_value(t, interface=None, x=None, y=None):
    if x is None:
        x = interface.discretization.x[:-3]
    if y is None:
        y = interface.discretization.y[:-3]

    x, y = numpy.meshgrid(x, y)

    fig1, ax1 = plt.subplots()
    cs = ax1.contourf(x, y, t.transpose(), 15)
    fig1.colorbar(cs)

    ax1.vlines(x[0, :], *y[[0, -1], 0], colors='0.3', linewidths=0.5)
    ax1.hlines(y[:, 0], *x[0, [0, -1]], colors='0.3', linewidths=0.5)

    plt.show()
