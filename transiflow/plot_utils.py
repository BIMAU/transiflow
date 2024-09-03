import numpy
import matplotlib.pyplot as plt

from transiflow import utils
from transiflow.utils import create_state_mtx # noqa: F401

def get_meshgrid(interface, x=None, y=None):
    '''Wrapper around `numpy.meshgrid(x, y)` that obtains necessary
    information from the interface.

    Parameters
    ----------
    interface : Interface
        Interface containing the coordinate vectors.
    x : array_like, optional
        Override the `x` meshgrid argument.
    y : array_like, optional
        Override the `y` meshgrid argument.

    Returns
    -------
    x : array_like
        2D array containing x coordinates.
    y : array_like
        2D array containing y coordinates.

    '''
    if x is None:
        x = interface.x[:-3]
    if y is None:
        if interface.ny > 1:
            y = interface.y[:-3]
        else:
            y = interface.z[:-3]

    return numpy.meshgrid(x, y)

def plot_contour(x, y, value, axis=2, title=None, legend=True, grid=True,
                 show=True, color=True, labels=True, levels=15, inline=False):
    '''Helper for plotting a contour plot.

    Parameters
    ----------
    x : array_like
        2D array containing x coordinates.
    y : array_like
        2D array containing y coordinates.
    value : array_like
        2D array of the value to plot.
    axis : int, optional
        Axis to ignore. Used for axis labels.
    title : str, optional
        Title of the plot.
    legend : bool, optional
        Whether to add a colorbar.
    grid : bool, optional
        Whether to show the mesh.
    show : bool, optional
        Whether to show the plot. This can be disabled when using
        `savefig()` manually.
    color : bool, optional
        Can be set to False to make plots suitable for black and white
        printing.
    labels : bool, optional
        Whether to add labels to the axis.
    levels : int, optional
        Number of levels used for the contours.
    inline : bool, optional
        Add inline labels to the contours. Useful for black and white
        plots.

    Returns
    -------
    fig : Figure
        Figure object that can be used to make manual modifications to
        the plot after calling this function.

    '''
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
    '''Create a plot of the velocity magnitude.

    See :meth:`plot_contour` and :func:`.compute_velocity_magnitude` for details.

    '''
    m = utils.compute_velocity_magnitude(state, interface, axis, position)

    x, y = get_meshgrid(interface)

    return plot_contour(x, y, m, axis=axis, title=title, *args, **kwargs)

def plot_streamfunction(state, interface, axis=2, title='Stream function', *args, **kwargs):
    '''Create a plot of the stream function.

    See :meth:`plot_contour` and :func:`.compute_streamfunction` for details.

    '''
    psi = utils.compute_streamfunction(state, interface, axis)

    x, y = get_meshgrid(interface)

    return plot_contour(x, y, psi, axis=axis, title=title, *args, **kwargs)

def plot_vorticity(state, interface, axis=2, title='Vorticity', *args, **kwargs):
    '''Create a plot of the vorticity.

    See :meth:`plot_contour` and :func:`.compute_vorticity` for details.

    '''
    psi = utils.compute_vorticity(state, interface, axis)

    x, y = get_meshgrid(interface)

    return plot_contour(x, y, psi, axis=axis, title=title, *args, **kwargs)

def plot_value(value, interface=None, x=None, y=None, title=None, *args, **kwargs):
    '''Create a plot of the velocity magnitude.

    See :meth:`plot_contour` for details and extra parameters.

    Parameters
    ----------
    value : array_like
        2D array of the value to plot.
    interface : Interface, optional
        Interface containing the coordinate vectors.
    x : array_like, optional
        First coordinate vector.
    y : array_like, optional
        Second coordinate vector.

    '''
    x, y = get_meshgrid(interface, x, y)

    return plot_contour(x, y, value, title=title, *args, **kwargs)
