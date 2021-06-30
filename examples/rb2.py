from jadapy import jdqz

from jadapy import EpetraInterface
from PyTrilinos import Teuchos, Epetra
from fvm import HYMLSInterface
from fvm import Continuation

from fvm.JadaHYMLSInterface import BorderedJadaHYMLSInterface

def hymls_main():
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
    prec_parameters.set('Number of Levels', 1)

    solver_parameters = parameters.sublist('Solver')

    iterative_solver_parameters = solver_parameters.sublist('Iterative Solver')
    iterative_solver_parameters.set('Maximum Iterations', 400)
    iterative_solver_parameters.set('Num Blocks', 80)

    comm = Epetra.PyComm()
    interface = HYMLSInterface.Interface(comm, parameters, nx, ny, nz, dim, dof)
    m = interface.map

    continuation = Continuation(interface, parameters)

    x0 = HYMLSInterface.Vector(m)
    x0.PutScalar(0.0)

    ds = 600
    target = 1553.083383
    x, mu, _ = continuation.continuation(x0, 'Rayleigh Number', 0, target, ds)

    jada_interface = BorderedJadaHYMLSInterface(interface)
    jac_op = EpetraInterface.CrsMatrix(interface.jacobian(x))
    mass_op = EpetraInterface.CrsMatrix(interface.mass_matrix())

    interface.preconditioner.Compute()

    alpha, beta = jdqz.jdqz(jac_op, mass_op, 5, tol=1e-7, subspace_dimensions=[30, 60],
                            interface=jada_interface, target=0.0)


if __name__ == '__main__':
    hymls_main()
