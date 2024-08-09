import numpy

from transiflow import Continuation
from transiflow import Interface

from transiflow.interface import JaDa

from jadapy import jdqz

import matplotlib.pyplot as plt


def main():
    ''' An example of performing a continuation for a 2D lid-driven cavity and computing eigenvalues along the way'''
    nx = 16
    ny = nx
    nz = 1

    # Define the problem
    parameters = {'Problem Type': 'Lid-driven Cavity',
                  # Problem parametes
                  'Reynolds Number': 1,
                  'Lid Velocity': 0}

    interface = Interface(parameters, nx, ny, nz)
    continuation = Continuation(interface)

    # Compute an initial guess
    x0 = interface.vector()
    x0 = continuation.continuation(x0, 'Lid Velocity', 0, 1, 1)[0]

    previous_subspaces = None

    # Perform a continuation to Reynolds numbers 7000-10000 with steps of 1000
    data_points = range(7000, 11000, 1000)
    eigs = numpy.zeros([len(data_points), 20], dtype=numpy.complex128)

    mu = 1
    for i, target in enumerate(data_points):
        ds = 100
        x, mu = continuation.continuation(x0, 'Reynolds Number', mu, target, ds)
        x0 = x

        # Compute the eigenvalues of the generalized eigenvalue problem near a target 2.8i
        jac_op = JaDa.Op(interface.jacobian(x))
        mass_op = JaDa.Op(interface.mass_matrix())
        jada_interface = JaDa.Interface(interface, jac_op, mass_op, len(x), numpy.complex128)

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
    ax.set_xlabel('$\\sigma_r$')
    ax.set_ylabel('$\\sigma_i$')

    legend = []
    for i in data_points:
        legend.append('Re='+str(i))

    ax.legend(legend)

    plt.title('Complex conjugate eigenvalues in the upper half-plane')
    plt.show()


if __name__ == '__main__':
    main()
