'''An example of performing a "poor man's continuation" for a 2D
lid-driven cavity using time integration.'''

import numpy

import matplotlib.pyplot as plt

from transiflow import Interface
from transiflow import TimeIntegration
from transiflow import utils


def postprocess(data, interface, x, t):
    data['t'].append(t)
    data['Volume Averaged Kinetic Energy'].append(
        utils.compute_volume_averaged_kinetic_energy(x, interface))


def main(nx=16):
    ny = nx

    # Define the problem
    parameters = {'Problem Type': 'Lid-driven Cavity',
                  # Problem parameters
                  'Reynolds Number': 0,
                  'Lid Velocity': 1,
                  # Use a stretched grid
                  'Grid Stretching Factor': 1.5,
                  # Give back extra output (this is also more expensive)
                  'Verbose': False,
                  'Theta': 1}

    interface = Interface(parameters, nx, ny)

    # Store data for computing the bifurcation diagram using postprocessing
    data = {'t': [], 'Volume Averaged Kinetic Energy': []}
    callback = lambda interface, x, t: postprocess(data, interface, x, t)

    n = interface.discretization.dof * nx * ny
    x = numpy.random.random(n)

    mu_list = []
    value_list = []

    for mu in range(0, 100, 10):
        interface.set_parameter('Reynolds Number', mu)
        time_integration = TimeIntegration(interface)
        x, t = time_integration.integration(x, 1, 10, callback)

        # Plot the traced value during the time integration
        # plt.plot(data.t, data.value)
        # plt.show()

        mu_list.append(mu)
        value_list.append(data['Volume Averaged Kinetic Energy'][-1])

    # Plot a bifurcation diagram
    plt.plot(mu_list, value_list)

    plt.title('Bifurcation diagram for the lid-driven cavity with $n_x=n_z={}$'.format(nx))
    plt.xlabel('Reynolds number')
    plt.ylabel('Volume averaged kinetic energy')
    plt.show()


if __name__ == '__main__':
    main()
