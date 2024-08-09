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

    def callback(self, interface, x, mu):
        self.append(mu, numpy.max(utils.compute_streamfunction(x, interface)))


def main():
    '''An example of performing a continuation for a double-gyre
    wind-driven ocean, plotting the streamfunction at different Reynolds
    numbers, and plotting the bifurcation diagram.'''

    nx = 32
    ny = nx

    # Define the problem
    parameters = {'Problem Type': 'Double Gyre',
                  # Problem parameters
                  'Reynolds Number': 16,
                  'Rossby Parameter': 1000,
                  'Wind Stress Parameter': 0,
                  # Give back extra output (this is also more expensive)
                  'Verbose': False}

    interface = Interface(parameters, nx, ny)
    continuation = Continuation(interface)

    # First activate the wind stress
    x0 = interface.vector()

    ds = 100
    target = 1000
    x1 = continuation.continuation(x0, 'Wind Stress Parameter', 0, target, ds)[0]

    plot_utils.plot_streamfunction(x1, interface, title='Streamfunction at Re=16')
    plot_utils.plot_vorticity(x1, interface, title='Vorticity at Re=16')

    # Perform a continuation to Reynolds number 40 without detecting bifurcation points
    # and use this in the bifurcation diagram
    data2 = Data()

    ds = 5
    target = 40
    x2, mu2 = continuation.continuation(x1, 'Reynolds Number', 16, target, ds,
                                        callback=data2.callback)

    plot_utils.plot_streamfunction(x2, interface, title='Streamfunction at Re={}'.format(mu2))

    # Add asymmetry to the problem
    ds = 10
    target = 1
    interface.set_parameter('Reynolds Number', 16)
    x3, mu3 = continuation.continuation(x1, 'Asymmetry Parameter', 0, target, ds, maxit=1)

    # Perform a continuation to Reynolds number 40 with asymmetry added to the problem,
    # meaning we can't stay on the unstable branch
    ds = 5
    target = 40
    x4, mu4 = continuation.continuation(x3, 'Reynolds Number', 16, target, ds)

    # Go back to the symmetric problem
    ds = -1
    target = 0
    x5, mu5 = continuation.continuation(x4, 'Asymmetry Parameter', mu3, target, ds)

    # Now compute the stable branch after the pitchfork bifurcation by going backwards
    # and use this in the bifurcation diagram
    data6 = Data()

    ds = -5
    target = 40
    x6, mu6 = continuation.continuation(x5, 'Reynolds Number', mu4, target, ds,
                                        callback=data6.callback)

    # Plot a bifurcation diagram
    plt.title('Bifurcation diagram for the QG model with $n_x=n_y={}$'.format(nx))
    plt.xlabel('Reynolds number')
    plt.ylabel('Maximum value of the streamfunction')
    plt.plot(data2.mu, data2.value)
    plt.plot(data6.mu, data6.value)
    plt.show()


if __name__ == '__main__':
    main()
