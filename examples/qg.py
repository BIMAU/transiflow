import numpy

import matplotlib.pyplot as plt

from fvm import Continuation
from fvm import Interface
from fvm import plot_utils
from fvm import utils

def main():
    '''An example of performing a continuation for a double-gyre
    wind-driven ocean, plotting the streamfunction at different Reynolds
    numbers, and plotting the bifurcation diagram.'''

    dim = 2
    dof = 3
    nx = 32
    ny = nx
    nz = 1
    n = dof * nx * ny * nz

    # Define the problem
    parameters = {'Problem Type': 'Double Gyre',
                  # Problem parameters
                  'Reynolds Number': 16,
                  'Rossby Parameter': 1000,
                  'Wind Stress Parameter': 0,
                  # Give back extra output (this is also more expensive)
                  'Verbose': True}

    interface = Interface(parameters, nx, ny, nz, dim, dof)

    # Value describes the value that is traced in the continuation
    # and time integration methods
    parameters['Value'] = lambda x: numpy.max(utils.compute_streamfunction(
        utils.create_state_mtx(x, nx, ny, nz, dof)[:, :, 0, 0],
        utils.create_state_mtx(x, nx, ny, nz, dof)[:, :, 0, 1], interface))

    continuation = Continuation(interface, parameters)

    # First activate the wind stress
    x0 = numpy.zeros(n)

    ds = 100
    target = 1000
    x0 = continuation.continuation(x0, 'Wind Stress Parameter', 0, target, ds)[0]

    v = utils.create_state_mtx(x0, nx, ny, nz, dof)
    plot_utils.plot_streamfunction(v[:, :, 0, 0], v[:, :, 0, 1], interface)

    # Perform an initial continuation to Reynolds number 29 without detecting bifurcation points
    ds = 5
    target = 29
    x, mu, data1 = continuation.continuation(x0, 'Reynolds Number', 16, target, ds)

    v = utils.create_state_mtx(x, nx, ny, nz, dof)
    plot_utils.plot_streamfunction(v[:, :, 0, 0], v[:, :, 0, 1], interface)

    # Plot a bifurcation diagram
    plt.plot(data1.mu, data1.value)
    plt.show()


if __name__ == '__main__':
    main()
