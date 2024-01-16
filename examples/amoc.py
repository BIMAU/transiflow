import numpy

import matplotlib.pyplot as plt

from transiflow import Continuation
from transiflow import Interface
from transiflow import plot_utils
from transiflow import utils


class Data:
    def __init__(self):
        self.mu = []
        self.value = []

    def append(self, mu, value):
        self.mu.append(mu)
        self.value.append(value)

    def filter(self):
        '''Filter out values obtained while converging onto a target'''
        idx = []
        for i, mu in enumerate(self.mu):
            if idx:
                idx = [j for j in idx if self.mu[j] < mu]

            idx.append(i)

        self.mu = [self.mu[i] for i in idx]
        self.value = [self.value[i] for i in idx]


def main():
    '''An example of performing a continuation for a double-gyre
    wind-driven ocean, plotting the streamfunction at different Reynolds
    numbers, and plotting the bifurcation diagram.'''

    dim = 2
    dof = 5
    nx = 60
    ny = 30
    nz = 1

    # Define the problem
    parameters = {'Problem Type': 'AMOC',
                  # Problem parameters
                  'Rayleigh Number': 4e4,
                  'Prandtl Number': 2.25,
                  'Lewis Number': 1,
                  'Freshwater Flux': 0,
                  'X-max': 5,
                  # Give back extra output (this is also more expensive)
                  'Verbose': False}

    interface = Interface(parameters, nx, ny, nz, dim, dof)
    continuation = Continuation(interface, parameters)

    # First increase the Rayleigh number to the desired value
    x0 = interface.vector()

    interface.set_parameter('Newton Tolerance', 1e-6)

    ds = 0.1
    target = 1
    x1 = continuation.continuation(x0, 'Temperature Forcing', 0, target, ds)[0]

    plot_utils.plot_streamfunction(x1, interface, title='Streamfunction at $\\sigma=0$')
    plot_utils.plot_vorticity(x1, interface, title='Vorticity at $\\sigma=0$')

    # Perform a continuation to freshwater flux 1 without detecting bifurcation points
    # and use this in the bifurcation diagram
    data2 = Data()
    interface.set_parameter('Postprocess', lambda interface, x, mu: data2.append(
        mu, numpy.max(utils.compute_streamfunction(x, interface))))
    interface.parameters['Postprocess'](interface, x1, 0)

    ds = 0.05
    target = 0.2
    interface.set_parameter('Minimum Step Size', 1e-12)
    x2, mu2 = continuation.continuation(x1, 'Freshwater Flux', 0, target, ds)

    data2.filter()

    sigma = mu2
    psi_max = numpy.max(utils.compute_streamfunction(x2, interface))
    plot_utils.plot_streamfunction(
        x2, interface, title=f'Streamfunction at $\\sigma={sigma:.2f}$ and $\\Psi_\\max={psi_max:.2e}$')
    plot_utils.plot_vorticity(
        x2, interface, title=f'Vorticity at $\\sigma={sigma:.2f}$ and $\\Psi_\\max={psi_max:.2e}$')
    plot_utils.plot_value(
        utils.create_state_mtx(x2, interface=interface)[:, :, 0, 4],
        interface, title=f'Salinity at $\\sigma={sigma:.2f}$ and $\\Psi_\\max={psi_max:.2e}$')

    # Add asymmetry to the problem
    ds = 0.05
    target = 1
    interface.set_parameter('Postprocess', None)
    interface.set_parameter('Maximum Continuation Steps', 1)
    interface.set_parameter('Freshwater Flux', 0)
    x3, mu3 = continuation.continuation(x1, 'Asymmetry Parameter', 0, target, ds)
    interface.set_parameter('Maximum Continuation Steps', 1000)

    # Perform a continuation to freshwater flux 1 with assymmetry added to the problem,
    # meaning we can't stay on the unstable branch
    ds = 0.01
    target = 0.2
    x4, mu4 = continuation.continuation(x3, 'Freshwater Flux', 0, target, ds)

    sigma = mu4
    psi_max = numpy.max(utils.compute_streamfunction(x4, interface))
    plot_utils.plot_streamfunction(
        x4, interface, title=f'Streamfunction at $\\sigma={sigma:.2f}$ and $\\Psi_\\max={psi_max:.2e}$')
    plot_utils.plot_vorticity(
        x4, interface, title=f'Vorticity at $\\sigma={sigma:.2f}$ and $\\Psi_\\max={psi_max:.2e}$')
    plot_utils.plot_value(
        utils.create_state_mtx(x4, interface=interface)[:, :, 0, 4],
        interface, title=f'Salinity at $\\sigma={sigma:.2f}$ and $\\Psi_\\max={psi_max:.2e}$')

    # Go back to the symmetric problem
    ds = -0.05
    target = 0
    x5, mu5 = continuation.continuation(x4, 'Asymmetry Parameter', mu3, target, ds)

    # Now compute the stable branch after the pitchfork bifurcation by going backwards
    # and use this in the bifurcation diagram
    data6 = Data()
    interface.set_parameter('Postprocess', lambda interface, x, mu: data6.append(
        mu, numpy.max(utils.compute_streamfunction(x, interface))))

    ds = -0.01
    target = 0.2
    interface.set_parameter('Maximum Step Size', 0.005)
    x6, mu6 = continuation.continuation(x5, 'Freshwater Flux', mu4, target, ds)

    sigma = mu6
    psi_max = numpy.max(utils.compute_streamfunction(x6, interface))
    plot_utils.plot_streamfunction(
        x6, interface, title=f'Streamfunction at $\\sigma={sigma:.2f}$ and $\\Psi_\\max={psi_max:.2e}$')
    plot_utils.plot_vorticity(
        x6, interface, title=f'Vorticity at $\\sigma={sigma:.2f}$ and $\\Psi_\\max={psi_max:.2e}$')
    plot_utils.plot_value(
        utils.create_state_mtx(x6, interface=interface)[:, :, 0, 4],
        interface, title=f'Salinity at $\\sigma={sigma:.2f}$ and $\\Psi_\\max={psi_max:.2e}$')

    # Plot a bifurcation diagram
    plt.title(f'Bifurcation diagram for the AMOC model with $n_x={nx}$, $n_y={ny}$')
    plt.xlabel('$\\sigma$')
    plt.ylabel('Maximum value of the streamfunction')
    plt.plot(data2.mu, data2.value)
    plt.plot(data6.mu, data6.value)
    plt.show()


if __name__ == '__main__':
    main()
