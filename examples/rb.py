from PyTrilinos import Epetra, Teuchos

from fvm import Continuation
from fvm import HYMLSInterface

def main():
    '''Detect the bifurcation in 2D Rayleigh-Benard'''
    dim = 2
    dof = 4
    nx = 32
    ny = 16
    nz = 1

    parameters = Teuchos.ParameterList()
    parameters.set('Bordered Solver', True)
    parameters.set('Verbose', True)

    parameters.set('Problem Type', 'Rayleigh-Benard')
    parameters.set('Reynolds Number', 1)
    parameters.set('Prandtl Number', 10)
    parameters.set('Biot Number', 1)
    parameters.set('xmax', 10)

    prec_parameters = parameters.sublist('Preconditioner')
    prec_parameters.set('Number of Levels', 0)

    comm = Epetra.PyComm()
    interface = HYMLSInterface.Interface(comm, parameters, nx, ny, nz, dim, dof)
    m = interface.map

    continuation = Continuation(interface, parameters)

    x0 = HYMLSInterface.Vector(m)
    x0.PutScalar(0.0)

    # Detect the bifurcation point
    parameters['Newton Tolerance'] = 1e-12
    parameters['Destination Tolerance'] = 1e-4
    parameters['Detect Bifurcation Points'] = True

    ds = 600
    target = 10000
    x, mu, data = continuation.continuation(x0, 'Rayleigh Number', 0, target, ds)


if __name__ == '__main__':
    main()
