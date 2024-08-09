import numpy
import pickle

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

    def callback(self, interface, x, mu):
        self.append(mu, numpy.max(utils.compute_streamfunction(x, interface)))


def generate_plots(interface, x, sigma):
    '''Generate plots for the stream function, vorticity and salinity and write them to a file'''
    psi_max = numpy.max(utils.compute_streamfunction(x, interface))

    plot_utils.plot_streamfunction(
        x, interface, title=f'Stream function at $\\sigma={sigma:.2f}$ and $\\Psi_\\max={psi_max:.2e}$',
        legend=False, grid=False, show=False)
    plt.savefig(f'streamfunction_{sigma:.2f}_{psi_max:.2e}.eps')
    plt.close()

    plot_utils.plot_vorticity(
        x, interface, title=f'Vorticity at $\\sigma={sigma:.2f}$ and $\\Psi_\\max={psi_max:.2e}$',
        legend=False, grid=False, show=False)
    plt.savefig(f'vorticity_{sigma:.2f}_{psi_max:.2e}.eps')
    plt.close()

    plot_utils.plot_value(
        utils.create_state_mtx(x, interface=interface)[:, :, 0, 4],
        interface, title=f'Salinity at $\\sigma={sigma:.2f}$ and $\\Psi_\\max={psi_max:.2e}$',
        legend=False, grid=False, show=False)
    plt.savefig(f'salinity_{sigma:.2f}_{psi_max:.2e}.eps')
    plt.close()


def main():
    '''An example of performing a continuation for a 2D AMOC, where the plots and interim
    solutions are written to files instead of storing them in memory and showing them on
    the screen.'''

    dim = 2
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

    interface = Interface(parameters, nx, ny, nz, dim)
    continuation = Continuation(interface, newton_tolerance=1e-6)

    # First increase the temperature forcing to the desired value
    x0 = interface.vector()

    ds = 0.1
    target = 1
    x1 = continuation.continuation(x0, 'Temperature Forcing', 0, target, ds)[0]

    # Write the solution to a file
    numpy.save('x1', x1)

    generate_plots(interface, x1, 0)

    # Enable the lines below to load the solution instead. Same for the ones below
    # parameters['Temperature Forcing'] = 1
    # x1 = numpy.load('x1.npy')

    # Perform a continuation to freshwater flux 0.2 without detecting bifurcation points
    # and use this in the bifurcation diagram
    data2 = Data()

    ds = 0.05
    target = 0.2
    x2, mu2 = continuation.continuation(x1, 'Freshwater Flux', 0, target,
                                        ds, ds_min=1e-12, callback=data2.callback)

    # Write the solution to a file
    numpy.save('x2', x2)

    generate_plots(interface, x2, mu2)

    # Write the data to a file
    with open('data2', 'wb') as f:
        pickle.dump(data2, f)

    # Enable the lines below to load the data
    # with open('data2', 'rb') as f:
    #     data2 = pickle.load(f)

    # Add asymmetry to the problem
    ds = 0.05
    target = 1
    parameters['Freshwater Flux'] = 0
    x3, mu3 = continuation.continuation(x1, 'Asymmetry Parameter', 0, target, ds, maxit=1)

    # Perform a continuation to freshwater flux 0.2 with asymmetry added to the problem,
    # meaning we can't stay on the unstable branch
    ds = 0.01
    target = 0.2
    x4, mu4 = continuation.continuation(x3, 'Freshwater Flux', 0, target, ds, ds_min=1e-12)

    # Go back to the symmetric problem
    ds = -0.05
    target = 0
    x5, mu5 = continuation.continuation(x4, 'Asymmetry Parameter', mu3, target, ds)

    # Write the solution to a file
    numpy.save('x5', x5)

    generate_plots(interface, x5, mu5)

    x5 = numpy.load('x5.npy')
    mu4 = 0.2

    # Now compute the stable branch after the pitchfork bifurcation by going backwards
    # and use this in the bifurcation diagram
    data6 = Data()

    ds = -0.01
    target = 0.2
    x6, mu6 = continuation.continuation(x5, 'Freshwater Flux', mu4, target,
                                        ds, ds_min=1e-12, ds_max=0.005,
                                        callback=data6.callback)

    # Write the solution to a file
    numpy.save('x6', x6)

    generate_plots(interface, x6, mu6)

    # Plot a bifurcation diagram
    plt.title(f'Bifurcation diagram for the AMOC model with $n_x={nx}$, $n_y={ny}$')
    plt.xlabel('$\\sigma$')
    plt.ylabel('Maximum value of the streamfunction')
    plt.plot(data2.mu, data2.value)
    plt.plot(data6.mu, data6.value)
    plt.savefig('bifurcation_diagram.eps')
    plt.close()


if __name__ == '__main__':
    main()
