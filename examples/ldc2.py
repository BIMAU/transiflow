import numpy

import matplotlib.pyplot as plt

from fvm import Continuation
from fvm import Interface
from fvm import plot_utils
from fvm import utils

def main():
    ''' An example of performing a continuation for a 2D lid-driven cavity and detecting a bifurcation point'''
    dim = 2
    dof = 3
    nx = 32
    ny = nx
    nz = 1
    n = dof * nx * ny * nz

    # Define the problem
    parameters = {'Problem Type': 'Lid-driven cavity',
                  # Problem parameters
                  'Reynolds Number': 1,
                  'Lid Velocity': 0,
                  # Use a stretched grid
                  'Grid Stretching Factor': 1.5,
                  # Set a maximum step size ds
                  'Maximum Step Size': 500,
                  # Give back extra output (this is also more expensive)
                  'Verbose': True,
                  # Value describes the value that is traced in the continuation
                  # and time integration methods
                  'Value': lambda x: utils.create_state_mtx(x, nx, ny, nz, dof)[nx // 2, ny // 4, 0, 0]}
    interface = Interface(parameters, nx, ny, nz, dim, dof)

    continuation = Continuation(interface, parameters)

    # Compute an initial guess
    x0 = numpy.zeros(n)
    x0 = continuation.continuation(x0, 'Lid Velocity', 0, 1, 0.1)[0]

    # Perform an initial continuation to Reynolds number 7000 without detecting bifurcation points
    ds = 100
    target = 7000
    x, mu, data1 = continuation.continuation(x0, 'Reynolds Number', 0, target, ds)

    parameters['Newton Tolerance'] = 1e-12
    parameters['Destination Tolerance'] = 1e-4
    parameters['Detect Bifurcation Points'] = True
    parameters['Maximum Step Size'] = 100

    # parameters['Eigenvalue Solver'] = {}
    # parameters['Eigenvalue Solver']['Target'] = 3j
    # parameters['Eigenvalue Solver']['Tolerance'] = 1e-9
    # parameters['Eigenvalue Solver']['Number of Eigenvalues'] = 20

    # Now detect the bifurcation point
    target = 10000
    x2, mu2, data2 = continuation.continuation(x, 'Reynolds Number', mu, target, ds)

    # Compute the unstable branch after the bifurcation
    parameters['Detect Bifurcation Points'] = False
    parameters['Maximum Step Size'] = 2000

    target = 10000
    parameters['Newton Tolerance'] = 1e-4
    x3, mu3, data3 = continuation.continuation(x2, 'Reynolds Number', mu2, target, ds)

    # Plot a bifurcation diagram
    plt.plot(data1.mu, data1.value)
    plt.plot(data2.mu, data2.value)
    plt.plot(data3.mu, data3.value)
    plt.show()

    # Add a perturbation based on the eigenvector
    interface.set_parameter('Reynolds Number', mu2)
    _, v = interface.eigs(x2, True)
    v = v[:, 0].real

    v = plot_utils.create_state_mtx(v, nx, ny, nz, dof)

    # Plot the velocity magnutide
    plot_utils.plot_velocity_magnitude(v[:, :, 0, 0], v[:, :, 0, 1], interface)

    # Plot the pressure
    plot_utils.plot_value(v[:, :, 0, 2], interface)


if __name__ == '__main__':
    main()
