import numpy

from fvm import Continuation
from fvm import Interface

def main():
    ''' An example of performing a continuation for a 2D lid-driven cavity and detecting a bifurcation point'''
    dim = 2
    dof = 3
    nx = 32
    ny = nx
    nz = 1

    # Define the problem
    parameters = {'Reynolds Number': 0,
                  'Problem Type': 'Lid-driven cavity',
                  'Grid Stretching Factor': 1.5,
                  'Maximum Step Size': 500}
    interface = Interface(parameters, nx, ny, nz, dim, dof)

    continuation = Continuation(interface, parameters)

    # Compute an initial guess
    x0 = numpy.zeros(dof * nx * ny * nz)
    x0 = continuation.newton(x0)

    # Perform an initial continuation to Reynolds number 5500 without detecting bifurcation points
    ds = 200
    maxit = 1000
    target = 5500
    x, mu = continuation.continuation(x0, 'Reynolds Number', target, ds, maxit)
    x0 = x

    parameters['Destination Tolerance'] = 1e-4
    parameters['Detect Bifurcation Points'] = True
    parameters['Eigenvalue Solver'] = {}
    parameters['Eigenvalue Solver']['Target'] = 2.8j

    # Now detect the bifurcation point
    target = 10000
    x, mu = continuation.continuation(x0, 'Reynolds Number', target, ds, maxit)

    print('A bifurcation ocurred at Re=' + str(mu))


if __name__ == '__main__':
    main()
