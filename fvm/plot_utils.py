import numpy
import matplotlib.pyplot as plt

from fvm import utils
from fvm.utils import create_state_mtx # noqa: F401

def get_meshgrid(interface, x=None, y=None):
    if x is None:
        x = interface.discretization.x[:-3]
    if y is None:
        if interface.ny > 1:
            y = interface.discretization.y[:-3]
        else:
            y = interface.discretization.z[:-3]

    return numpy.meshgrid(x, y)

def plot_contour(x, y, value, axis=2, title=None, legend=True, grid=True,
                 show=True, color=True, labels=True, levels=15, inline=False):
    fig, ax = plt.subplots()

    if color:
        cs = ax.contourf(x, y, value.transpose(), levels)
    else:
        cs = ax.contour(x, y, value.transpose(), levels, colors=['k'])

    if inline:
        ax.clabel(cs, cs.levels, inline=True, fontsize=10)

    if legend:
        fig.colorbar(cs)

    if grid:
        ax.vlines(x[0, :], *y[[0, -1], 0], colors='0.3', linewidths=0.5)
        ax.hlines(y[:, 0], *x[0, [0, -1]], colors='0.3', linewidths=0.5)

    if labels:
        if axis == 0:
            ax.set_xlabel('y')
        else:
            ax.set_xlabel('x')

        if axis == 2:
            ax.set_ylabel('y')
        else:
            ax.set_ylabel('z')

    if title:
        plt.title(title)

    if show:
        plt.show()

    return fig

def plot_velocity_magnitude(state, interface, axis=2, position=None, title='Velocity magnitude', *args, **kwargs):
    m = utils.compute_velocity_magnitude(state, interface, axis, position)

    x, y = get_meshgrid(interface)

    return plot_contour(x, y, m, axis=axis, title=title, *args, **kwargs)

def plot_streamfunction(state, interface, axis=2, title='Streamfunction', *args, **kwargs):
    psi = utils.compute_streamfunction(state, interface, axis)

    x, y = get_meshgrid(interface)

    return plot_contour(x, y, psi, axis=axis, title=title, *args, **kwargs)

def plot_vorticity(state, interface, axis=2, title='Vorticity', *args, **kwargs):
    psi = utils.compute_vorticity(state, interface, axis)

    x, y = get_meshgrid(interface)

    return plot_contour(x, y, psi, axis=axis, title=title, *args, **kwargs)

def plot_value(t, interface=None, x=None, y=None, title=None, *args, **kwargs):
    x, y = get_meshgrid(interface, x, y)

    return plot_contour(x, y, t, title=title, *args, **kwargs)
