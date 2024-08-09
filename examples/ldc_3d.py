import numpy
import pickle

from transiflow import Continuation
from transiflow import Interface
from transiflow import utils


class Data:
    def __init__(self):
        self.mu = []
        self.value = []

    def append(self, mu, value):
        self.mu.append(mu)
        self.value.append(value)


def compute_volume_averaged_kinetic_energy(x_local, interface):
    # Because HYMLS uses local subdomains, we need a different interface for postprocessing purposes
    # that operates on the global domain
    postprocess_interface = Interface(interface.parameters,
                                      interface.nx_global, interface.ny_global, interface.nz_global,
                                      interface.discretization.dim, interface.discretization.dof)

    return utils.compute_volume_averaged_kinetic_energy(x_local.array, postprocess_interface)


def write_solution(interface, x, mu, name=None, enable_output=True):
    if not enable_output:
        return

    nx = interface.nx_global
    ny = interface.ny_global
    nz = interface.nz_global

    if not name:
        name = mu

    try:
        x_local = x.gather()
    except AttributeError:
        x_local = x

    if interface.comm.MyPID() == 0:
        ke = compute_volume_averaged_kinetic_energy(x_local, interface)

        with open('ldc_' + str(name) + '_ke_' + str(nx) + '_' + str(ny) + '_' + str(nz) + '.npy', 'wb') as f:
            numpy.save(f, ke)

        with open('ldc_' + str(name) + '_mu_' + str(nx) + '_' + str(ny) + '_' + str(nz) + '.npy', 'wb') as f:
            numpy.save(f, mu)

        with open('ldc_' + str(name) + '_x_' + str(nx) + '_' + str(ny) + '_' + str(nz) + '.npy', 'wb') as f:
            numpy.save(f, x_local.array)


def read_solution(interface, name):
    nx = interface.nx_global
    ny = interface.ny_global
    nz = interface.nz_global

    mu = numpy.load('ldc_' + str(name) + '_mu_' + str(nx) + '_' + str(ny) + '_' + str(nz) + '.npy')
    if interface.comm.MyPID() == 0:
        x = numpy.load('ldc_' + str(name) + '_x_' + str(nx) + '_' + str(ny) + '_' + str(nz) + '.npy')
    else:
        x = []

    x = interface.vector_from_array(x)

    return x, mu


def postprocess(data, interface, x, mu, enable_output):
    if not enable_output:
        return

    nx = interface.nx_global
    ny = interface.ny_global
    nz = interface.nz_global

    x_local = x.gather()
    if interface.comm.MyPID() == 0:
        # Store data for a bifurcation diagram at every continuation step
        data.append(mu, utils.compute_volume_averaged_kinetic_energy(x_local, interface))

        with open('ldc_bif_' + str(nx) + '_' + str(ny) + '_' + str(nz) + '.obj', 'wb') as f:
            pickle.dump(data, f)

        # Store the solution at every continuation step
        write_solution(interface, x_local, mu)


def main():
    ''' An example of performing a continuation for a 3D lid-driven cavity using HYMLS.
    Multiple processors can be used by calling this script using mpi, e.g.:

    OMP_NUM_THREADS=1 mpiexec -np 4 python examples/ldc_3d.py

    Disabling threading is adviced, since this is broken in Epetra.'''

    dim = 3
    dof = 4
    nx = 16
    ny = nx
    nz = nx

    # Define the problem
    parameters = {'Problem Type': 'Lid-driven Cavity',
                  # Problem parameters
                  'Reynolds Number': 1,
                  'Lid Velocity': 0,
                  # Use a stretched grid
                  'Grid Stretching Factor': 1.5
                  }

    enable_output = False

    # Set some parameters for the Belos solver (GMRES)
    parameters['Solver'] = {}
    parameters['Solver']['Iterative Solver'] = {}
    parameters['Solver']['Iterative Solver']['Maximum Iterations'] = 500
    parameters['Solver']['Iterative Solver']['Num Blocks'] = 100
    parameters['Solver']['Iterative Solver']['Convergence Tolerance'] = 1e-6

    # Use one level in the HYMLS preconditioner. More levels means a less accurate factorization,
    # but more parallelism and a smaller linear system at the coarsest level.
    parameters['Preconditioner'] = {}
    parameters['Preconditioner']['Number of Levels'] = 1

    # Define a HYMLS interface that handles everything that is different when using HYMLS+Trilinos
    # instead of NumPy as computational backend
    interface = Interface(parameters, nx, ny, nz, dim, dof, backend='HYMLS')

    data = Data()
    callback = lambda interface, x, mu: postprocess(data, interface, x, mu, enable_output)

    continuation = Continuation(interface, parameters)

    # Compute an initial guess
    x0 = interface.vector()
    x = continuation.continuation(x0, 'Lid Velocity', 0, 1, 1, callback=callback)[0]

    # Perform an initial continuation to Reynolds number 1700 without detecting bifurcation points
    ds = 100
    target = 1800
    x, mu = continuation.continuation(x, 'Reynolds Number', 0, target,
                                      ds, ds_max=100, callback=callback)

    # Store point b from which we start locating the bifurcation point
    write_solution(interface, x, mu, 'b', enable_output)

    # # Restart from point b. In this case the above code can be disabled
    # interface.set_parameter('Lid Velocity', 1)
    # x, mu = read_solution(interface, 'b')

    # Now detect the bifurcation point
    parameters['Eigenvalue Solver'] = {}
    parameters['Eigenvalue Solver']['Target'] = 0.4j
    parameters['Eigenvalue Solver']['Tolerance'] = 1e-8
    parameters['Eigenvalue Solver']['Number of Eigenvalues'] = 2

    ds = 100
    target = 2500
    x2, mu2 = continuation.continuation(x, 'Reynolds Number', mu, target, ds, ds_max=100,
                                        detect_bifurcations=True, callback=callback)

    # Store the solution at the bifurcation point
    write_solution(interface, x, mu, 'c', enable_output)


if __name__ == '__main__':
    main()
