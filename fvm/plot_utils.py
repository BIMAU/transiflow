import numpy
import matplotlib.pyplot as plt

from fvm import utils
from fvm.utils import create_state_mtx # noqa: F401

def plot_velocity_magnitude(state, interface):
    m = utils.compute_velocity_magnitude(state, interface)

    x = interface.discretization.x[:-3]
    y = interface.discretization.y[:-3]
    x, y = numpy.meshgrid(x, y)

    fig1, ax1 = plt.subplots()
    cs = ax1.contourf(x, y, m.transpose(), 15)
    fig1.colorbar(cs)

    ax1.vlines(x[0, :], *y[[0, -1], 0], colors='0.3', linewidths=0.5)
    ax1.hlines(y[:, 0], *x[0, [0, -1]], colors='0.3', linewidths=0.5)

    plt.show()

def plot_streamfunction(state, interface):
    psi = utils.compute_streamfunction(state, interface)

    x = interface.discretization.x[:-3]
    y = interface.discretization.y[:-3]
    x, y = numpy.meshgrid(x, y)

    fig1, ax1 = plt.subplots()
    cs = ax1.contourf(x, y, psi.transpose(), 15)
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
