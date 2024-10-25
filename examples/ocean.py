import numpy
import matplotlib.pyplot as plt

from transiflow import Continuation
from transiflow import Interface
from transiflow import plot_utils
from transiflow import utils


def postprocess(data, interface, x, mu):
    psi = utils.compute_overturning_streamfunction(x, interface)

    print('Psi min', numpy.min(psi))
    print('Psi max', numpy.max(psi))

    if data:
        data['Rotation'].append(mu)
        data['Stream Function Maximum'].append(numpy.max(psi))


def main():
    nx = 16
    ny = 32
    nz = 16

    # Define the problem
    parameters = {'Problem Type': 'Ocean Basin',
                  'Grid Stretching': False,
                  'X-periodic': False,
                  'X-min': 286,
                  'X-max': 350,
                  'Y-min': -60,
                  'Y-max': 60,
                  'Depth': 4000,
                  'Rotation Flag': 0,
                  'Horizontal Friction Coefficient': 16e6,
                  'Vertical Friction Coefficient': 1e-3,
                  'Horizontal Heat Diffusivity': 1e3,
                  'Vertical Heat Diffusivity': 1e-4,
                  'Biot Number': 10,
                  'Bouyancy Ratio': 1,
                  'Salinity Flux Strength': 0,
                  }

    interface = Interface(parameters, nx, ny, nz)
    continuation = Continuation(interface)

    # Compute an initial guess
    x0 = interface.vector()

    callback = lambda interface, x, mu: postprocess(None, interface, x, mu)

    # Enable the temperature forcing
    x1 = continuation.continuation(x0, 'Temperature Forcing', 0, 10, 1, callback=callback)[0]

    # plot_utils.plot_overturning_streamfunction(x1, interface)

    # Enable the salinity forcing
    x2 = continuation.continuation(x1, 'Salinity Forcing', 0, 1, 1, callback=callback)[0]

    data = {'Rotation': [], 'Stream Function Maximum': []}
    callback = lambda interface, x, mu: postprocess(data, interface, x, mu)

    # Enable rotation
    x3 = continuation.continuation(x2, 'Rotation Flag', 0, 1, 0.1, ds_min=1e-6, callback=callback)[0]

    plot_utils.plot_overturning_streamfunction(x3, interface)

    # Plot a bifurcation diagram
    plt.title(f'The response of the overturning strength upon introducing rotation')
    plt.xlabel('$\\eta_f$')
    plt.ylabel('Maximum value of the streamfunction')
    plt.plot(data['Rotation'], data['Stream Function Maximum'])
    plt.show()


if __name__ == '__main__':
    main()
