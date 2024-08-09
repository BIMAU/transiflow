import numpy

import matplotlib.pyplot as plt

from transiflow import Interface
from transiflow import TimeIntegration
from transiflow import utils


class Data:
    def __init__(self):
        self.t = []
        self.value = []

    def append(self, t, value):
        self.t.append(t)
        self.value.append(value)

    def callback(self, interface, x, t):
        self.append(t, utils.compute_volume_averaged_kinetic_energy(x, interface))


def main():
    ''' An example of performing a "poor man's continuation" for a 2D lid-driven cavity using time integration'''
    nx = 16
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
    data = Data()

    n = interface.discretization.dof * nx * ny
    x = numpy.random.random(n)

    mu_list = []
    value_list = []

    for mu in range(0, 100, 10):
        interface.set_parameter('Reynolds Number', mu)
        time_integration = TimeIntegration(interface)
        x, t = time_integration.integration(x, 1, 10, data.callback)

        # Plot the traced value during the time integration
        # plt.plot(data.t, data.value)
        # plt.show()

        mu_list.append(mu)
        value_list.append(data.value[-1])

    # Plot a bifurcation diagram
    plt.plot(mu_list, value_list)

    plt.title('Bifurcation diagram for the lid-driven cavity with $n_x=n_z={}$'.format(nx))
    plt.xlabel('Reynolds number')
    plt.ylabel('Volume averaged kinetic energy')
    plt.show()


if __name__ == '__main__':
    main()
