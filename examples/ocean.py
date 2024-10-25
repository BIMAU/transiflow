import numpy
import matplotlib.pyplot as plt

from transiflow import Continuation
from transiflow import Interface
from transiflow import plot_utils
from transiflow import utils


def postprocess(data, name, interface, x, mu):
    psi = utils.compute_overturning_streamfunction(x, interface)

    print('Psi min', numpy.min(psi))
    print('Psi max', numpy.max(psi))

    if data:
        data[name].append(mu)
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

    callback = lambda interface, x, mu: postprocess(None, None, interface, x, mu)

    # Enable the temperature forcing
    x1 = continuation.continuation(x0, 'Temperature Forcing', 0, 10, 1, callback=callback)[0]

    interface.save_state('x1', x1)

    # plot_utils.plot_overturning_streamfunction(x1, interface)

    # Enable the salinity forcing
    x2 = continuation.continuation(x1, 'Salinity Forcing', 0, 1, 1, callback=callback)[0]

    interface.save_state('x2', x2)

    # Enable the wind forcing
    x3 = continuation.continuation(x2, 'Wind Forcing', 0, 1, 1, callback=callback)[0]

    interface.save_state('x3', x3)

    data = {'Rotation': [], 'Stream Function Maximum': []}
    callback = lambda interface, x, mu: postprocess(data, 'Rotation', interface, x, mu)

    # Enable rotation
    x4 = continuation.continuation(x3, 'Rotation Flag', 0, 1, 0.1, ds_min=1e-6, callback=callback)[0]

    interface.save_state('x4', x4)

    plot_utils.plot_overturning_streamfunction(x4, interface)

    # Plot a bifurcation diagram
    plt.title(f'The response of the overturning strength upon introducing rotation')
    plt.xlabel('$\\eta_f$')
    plt.ylabel('Maximum value of the streamfunction')
    plt.plot(data['Rotation'], data['Stream Function Maximum'])
    plt.show()

    data = {'Salinity Flux': [], 'Stream Function Maximum': []}
    callback = lambda interface, x, mu: postprocess(data, 'Salinity Flux', interface, x, mu)

    # Perform a continuation in gamma (with lambda = 1, this is
    # equivalent to performing a continuation in sigma).
    x5 = continuation.continuation(x4, 'Salinity Flux Strength', 0, 0.4, 0.01, ds_min=1e-6,
                                   detect_bifurcations=True, callback=callback)[0]

    plot_utils.plot_overturning_streamfunction(x5, interface)

    # Plot a bifurcation diagram
    plt.title(f'The response of the overturning strength upon introducing a freshwater flux')
    plt.xlabel('$\\sigma$')
    plt.ylabel('Maximum value of the streamfunction')
    plt.plot(data['Salinity Flux'], data['Stream Function Maximum'])
    plt.show()


if __name__ == '__main__':
    main()
