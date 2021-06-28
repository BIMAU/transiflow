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
    parameters['Value'] = lambda x: numpy.max(utils.compute_streamfunction(x, interface))

    continuation = Continuation(interface, parameters)

    # First activate the wind stress
    x0 = numpy.zeros(n)

    ds = 100
    target = 1000
    x1 = continuation.continuation(x0, 'Wind Stress Parameter', 0, target, ds)[0]

    plot_utils.plot_streamfunction(x1, interface)

    # Perform a continuation to Reynolds number 40 without detecting bifurcation points
    ds = 5
    target = 40
    interface.set_parameter('Maximum Step Size', 10)
    x2, mu2, data2 = continuation.continuation(x1, 'Reynolds Number', 16, target, ds)

    plot_utils.plot_streamfunction(x2, interface)

    # Add asymmetry to the problem
    ds = 10
    target = 1
    interface.set_parameter('Maximum Iterations', 1)
    interface.set_parameter('Reynolds Number', 16)
    x3, mu3, data3 = continuation.continuation(x1, 'Asymmetry Parameter', 0, target, ds)

    # Perform a continuation to Reynolds number 40 with assymmetry added to the problem,
    # meaning we can't stay on the unstable branch
    ds = 5
    target = 40
    interface.set_parameter('Maximum Iterations', 1000)
    x4, mu4, data4 = continuation.continuation(x3, 'Reynolds Number', 16, target, ds)

    # Go back to the symmetric problem
    ds = -1
    target = 0
    x5, mu5, data5 = continuation.continuation(x4, 'Asymmetry Parameter', mu3, target, ds)

    # Now compute the stable branch after the pitchfork bifurcation by going backwards
    ds = -5
    target = 40
    x6, mu6, data6 = continuation.continuation(x5, 'Reynolds Number', mu4, target, ds)

    # Plot a bifurcation diagram
    plt.plot(data2.mu, data2.value)
    plt.plot(data6.mu, data6.value)
    plt.show()


if __name__ == '__main__':
    main()
