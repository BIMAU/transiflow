import numpy

from fvm import Continuation
from fvm import Interface
from fvm import utils

from jadapy import jdqz

from fvm.JadaInterface import JadaInterface, JadaOp

import matplotlib.pyplot as plt

def main():
    ''' An example of performing a continuation for a 2D lid-driven cavity'''
    dim = 2
    dof = 3
    nx = 16
    ny = nx
    nz = 1
    n = dof * nx * ny * nz

    # Define the problem
    parameters = {'Problem Type': 'Lid-driven cavity',
                  # Problem parametes
                  'Reynolds Number': 1,
                  'Lid Velocity': 0,
                  # Value describes the value that is traced in the continuation
                  # and time integration methods
                  'Value': lambda x: utils.create_state_mtx(x, nx, ny, nz, dof)[nx // 2, ny // 4, 0, 0]}
    interface = Interface(parameters, nx, ny, nz, dim, dof)

    continuation = Continuation(interface, parameters)

    # Compute an initial guess
    x0 = numpy.zeros(dof * nx * ny * nz)
    x0 = continuation.continuation(x0, 'Lid Velocity', 0, 1, 0.1)[0]

    previous_subspaces = None

    # Perform a continuation to Reynolds numbers 7000-10000 with steps of 1000
    data_points = range(7000, 11000, 1000)
    eigs = numpy.zeros([len(data_points), 20], dtype=numpy.complex128)

    mu = 1
    for i, target in enumerate(data_points):
        ds = 100
        x, mu, _ = continuation.continuation(x0, 'Reynolds Number', mu, target, ds)
        x0 = x

        # Compute the eigenvalues of the generalized eigenvalue problem near a target 2.8i
        jac_op = JadaOp(interface.jacobian(x))
        mass_op = JadaOp(interface.mass_matrix())
        jada_interface = JadaInterface(interface, jac_op, mass_op, n, numpy.complex128)

        alpha, beta, q, z = jdqz.jdqz(jac_op, mass_op, eigs.shape[1], tol=1e-7, subspace_dimensions=[30, 60], target=2.8j,
                                      interface=jada_interface, arithmetic='complex', prec=jada_interface.shifted_prec,
                                      return_subspaces=True, initial_subspaces=previous_subspaces)

        # Store the eigenvalues
        eigs[i, :] = numpy.array(sorted(alpha / beta, key=lambda x: abs(x)))
        eigs[i, :] = alpha / beta

        # Use the subspaces in JDQZ computed for this Reynolds number as initial guesses for JDQZ at the next Reynolds number
        previous_subspaces = (q, z)

    # Plot the eigenvalues
    fig, ax = plt.subplots()
    for i in range(eigs.shape[0]):
        ax.scatter(eigs[i, :].real, eigs[i, :].imag, marker='+')
    ax.set_ylim(abs(eigs.imag).min() - 0.1, abs(eigs.imag).max() + 0.1)
    plt.show()


if __name__ == '__main__':
    main()
