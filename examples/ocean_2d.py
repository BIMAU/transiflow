import numpy

from transiflow import Continuation
from transiflow import Interface
from transiflow import plot_utils


def main():
    nx = 1
    ny = 32
    nz = 16

    # Define the problem
    parameters = {'Problem Type': '2D Ocean',
                  'Grid Stretching': False,
                  'X-min': 0,
                  'X-max': 180 / numpy.pi,
                  'Y-min': -60,
                  'Y-max': 60,
                  'Depth': 4000,
                  'Rotation Flag': 0,
                  'Horizontal Friction Coefficient': 16e6,
                  'Vertical Friction Coefficient': 1e-3,
                  'Horizontal Heat Diffusivity': 1e3,
                  'Vertical Heat Diffusivity': 1e-4,
                  'Biot Number': 10,
                  'Rayleigh Number': 1.2e-4,
                  'Bouyancy Ratio': 1,
                  }

    interface = Interface(parameters, nx, ny, nz)
    continuation = Continuation(interface)

    # Compute an initial guess
    x0 = interface.vector()

    # Enable the temperature forcing
    x1 = continuation.continuation(x0, 'Temperature Forcing', 0, 10, 1)[0]

    # Enable the salinity forcing
    x2 = continuation.continuation(x1, 'Salinity Forcing', 0, 1, 1)[0]

    # Perform a continuation in gamma (with lambda = 1, this is
    # equivalent to performing a continuation in sigma).
    ds = 0.01
    x3 = continuation.continuation(x2, 'Salinity Flux Strength', 0, 0.2, ds, ds_min=1e-6, ds_max=ds,
                                   detect_bifurcations=True)[0]

    plot_utils.plot_overturning_streamfunction(x3, interface)


if __name__ == '__main__':
    main()
