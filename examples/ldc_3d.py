''' An example of performing a continuation for a 3D lid-driven cavity using HYMLS.
Multiple processors can be used by calling this script using mpi, e.g.:

OMP_NUM_THREADS=1 mpiexec -np 4 python examples/ldc_3d.py

Disabling threading is adviced, since this is broken in Epetra.'''

from transiflow import Continuation
from transiflow import Interface
from transiflow import utils


def postprocess(data, interface, x, mu, enable_output):
    if not enable_output:
        return

    nx = interface.nx
    ny = interface.ny
    nz = interface.nz

    # Store data for a bifurcation diagram at every continuation step
    data['Reynolds Number'].append(mu)
    data['Volume Averaged Kinetic Energy'].append(
        utils.compute_volume_averaged_kinetic_energy(x, interface))

    interface.save_json('ldc_bif_' + str(nx) + '_' + str(ny) + '_' + str(nz) + '.json', data)

    # Store the solution at every continuation step
    interface.save_state(str(mu), x)


def main(nx=16):
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

    # Use one level in the HYMLS preconditioner. More levels means a
    # less accurate factorization, but more parallelism and a smaller
    # linear system at the coarsest level.
    parameters['Preconditioner'] = {}
    parameters['Preconditioner']['Number of Levels'] = 1

    # Define a HYMLS interface that handles everything that is
    # different when using HYMLS+Trilinos instead of SciPy as
    # computational backend
    interface = Interface(parameters, nx, ny, nz, backend='HYMLS')
    continuation = Continuation(interface)

    # Compute an initial guess
    x0 = interface.vector()
    x = continuation.continuation(x0, 'Lid Velocity', 0, 1, 1)[0]

    # Perform an initial continuation to Reynolds number 1700 without
    # detecting bifurcation points
    data = {'Reynolds Number': [], 'Volume Averaged Kinetic Energy': []}
    callback = lambda interface, x, mu: postprocess(data, interface, x, mu, enable_output)

    ds = 100
    target = 1800
    x, mu = continuation.continuation(x, 'Reynolds Number', 1, target,
                                      ds, ds_max=100, callback=callback)

    # Store point b from which we start locating the bifurcation point
    interface.save_state('b', x)

    # # Restart from point b. In this case the above code can be disabled
    # x = interface.load_state('b')

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
    interface.save_state('c', x)


if __name__ == '__main__':
    main()
