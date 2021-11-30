import numpy
import matplotlib.pyplot as plt

from fvm import utils
from fvm.utils import create_state_mtx # noqa: F401

def get_meshgrid(interface, x=None, y=None):
    if x is None:
        x = interface.discretization.x[:-3]
    if y is None:
        y = interface.discretization.y[:-3]

    return numpy.meshgrid(x, y)

def plot_contour(x, y, value, legend=True, grid=True, show=True):
    fig, ax = plt.subplots()
    cs = ax.contourf(x, y, value.transpose(), 15)

    if legend:
        fig.colorbar(cs)

    if grid:
        ax.vlines(x[0, :], *y[[0, -1], 0], colors='0.3', linewidths=0.5)
        ax.hlines(y[:, 0], *x[0, [0, -1]], colors='0.3', linewidths=0.5)

    if show:
        plt.show()

    return fig

def plot_velocity_magnitude(state, interface, *args, **kwargs):
    m = utils.compute_velocity_magnitude(state, interface)

    x, y = get_meshgrid(interface)

    return plot_contour(x, y, m, *args, **kwargs)

def plot_streamfunction(state, interface, *args, **kwargs):
    psi = utils.compute_streamfunction(state, interface)

    x, y = get_meshgrid(interface)

    return plot_contour(x, y, psi, *args, **kwargs)

def plot_value(t, interface=None, x=None, y=None, *args, **kwargs):
    x, y = get_meshgrid(interface, x, y)

    return plot_contour(x, y, t, *args, **kwargs)
