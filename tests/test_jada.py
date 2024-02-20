import numpy
import pytest

from numpy.testing import assert_allclose

import matplotlib.pyplot as plt

from transiflow import utils

# Import common fixtures
from tests.jada_fixtures import * # noqa: F401, F403
from tests.jada_fixtures import check_eigenvalues

@pytest.fixture(autouse=True, scope='module')
def import_test():
    try:
        from transiflow.interface import JaDa # noqa: F401
    except ImportError:
        pytest.skip('jadapy not found')

class Operator:
    def __init__(self, A, B, alpha=0, beta=1):
        self.A = A
        self.B = B

        self.alpha = alpha
        self.beta = beta

        self.dtype = A.dtype
        self.shape = A.shape

    def matvec(self, x):
        return (self.A @ x) * self.beta - (self.B @ x) * self.alpha

    def proj(self, x):
        return x

def check_divfree(discretization, state):
    A = discretization.jacobian(state)
    x = A @ state
    for i in range(len(state)):
        if i % discretization.dof == discretization.dim:
            assert abs(x[i]) < 1e-14

def test_solve(interface, x, tol):
    from transiflow.interface import JaDa
    from jadapy.utils import norm

    numpy.random.seed(1234)

    jac_op = JaDa.Op(interface.jacobian(x))
    mass_op = JaDa.Op(interface.mass_matrix())
    jada_interface = JaDa.Interface(interface, jac_op, mass_op, len(x), preconditioned_solve=True)

    assert norm(x) > tol

    check_divfree(interface.discretization, x)

    b = numpy.zeros((x.shape[0], 1))
    b[:, 0] = x

    op = Operator(jac_op, mass_op)
    x = jada_interface.solve(op, b, tol, maxit=1)

    r = jac_op.matvec(x) - b
    r[interface.pressure_row] = 0

    assert norm(x) > tol
    assert norm(r) / norm(b) < tol

def test_shifted_solve(interface, x, tol):
    from transiflow.interface import JaDa
    from jadapy.utils import norm

    numpy.random.seed(1234)

    check_divfree(interface.discretization, x)
    assert norm(x) > tol

    jac_op = JaDa.Op(interface.jacobian(x))
    mass_op = JaDa.Op(interface.mass_matrix())

    b = numpy.zeros((x.shape[0], 1))
    b[:, 0] = x

    jada_interface = JaDa.Interface(interface, jac_op, mass_op, len(x), preconditioned_solve=True)

    op = Operator(jac_op, mass_op, 1, 2)
    x = jada_interface.solve(op, b, tol, maxit=1)

    r = (op.A @ x) * op.beta - (op.B @ x) * op.alpha - b
    r[interface.pressure_row] = 0

    assert norm(x) > tol
    assert norm(r) / norm(b) > tol

    jada_interface = JaDa.Interface(interface, jac_op, mass_op, len(x), preconditioned_solve=True, shifted=True)

    op = Operator(jac_op, mass_op, 1, 2)
    x = jada_interface.solve(op, b, tol, maxit=1)

    r = (op.A @ x) * op.beta - (op.B @ x) * op.alpha - b
    r[interface.pressure_row] = 0

    assert norm(x) > tol
    assert norm(r) / norm(b) < tol

def test_2D(arpack_eigs, interface, x, num_evs, tol, atol, interactive=False):
    from transiflow.interface import JaDa
    from jadapy import jdqz

    numpy.random.seed(1234)

    jac_op = JaDa.Op(interface.jacobian(x))
    mass_op = JaDa.Op(interface.mass_matrix())

    alpha, beta, q, z = jdqz.jdqz(jac_op, mass_op, num_evs, tol=tol, subspace_dimensions=[20, 40], return_subspaces=True)
    jdqz_eigs = numpy.array(sorted(alpha / beta, key=lambda x: abs(x)))
    jdqz_eigs = jdqz_eigs[:num_evs]

    assert_allclose(jdqz_eigs.real, arpack_eigs.real, rtol=0, atol=atol)
    assert_allclose(abs(jdqz_eigs.imag), abs(arpack_eigs.imag), rtol=0, atol=atol)

    H = utils.dot(q, q)
    assert_allclose(H, numpy.eye(q.shape[1]), rtol=0, atol=atol)

    if not interactive:
        return

    fig, ax = plt.subplots()
    ax.scatter(jdqz_eigs.real, jdqz_eigs.imag, marker='+')
    plt.show()

def test_complex_2D(arpack_eigs, interface, x, num_evs, tol, atol, interactive=False):
    from transiflow.interface import JaDa
    from jadapy import jdqz

    numpy.random.seed(1234)

    jac_op = JaDa.Op(interface.jacobian(x))
    mass_op = JaDa.Op(interface.mass_matrix())

    alpha, beta, v, q, z = jdqz.jdqz(jac_op, mass_op, num_evs, tol=tol, subspace_dimensions=[30, 60],
                                     arithmetic='complex', return_eigenvectors=True, return_subspaces=True)
    check_eigenvalues(jac_op, mass_op, alpha / beta, v, num_evs, atol)

    jdqz_eigs = numpy.array(sorted(alpha / beta, key=lambda x: abs(x)))

    assert_allclose(jdqz_eigs.real, arpack_eigs.real, rtol=0, atol=atol)
    assert_allclose(abs(jdqz_eigs.imag), abs(arpack_eigs.imag), rtol=0, atol=atol)

    H = utils.dot(q, q)
    assert_allclose(H, numpy.eye(q.shape[1]), rtol=0, atol=atol)

    if not interactive:
        return

    fig, ax = plt.subplots()
    ax.scatter(jdqz_eigs.real, jdqz_eigs.imag, marker='+')
    plt.show()

def test_prec_2D(arpack_eigs, interface, x, num_evs, tol, atol, interactive=False):
    from transiflow.interface import JaDa
    from jadapy import jdqz

    numpy.random.seed(1234)

    jac_op = JaDa.Op(interface.jacobian(x))
    mass_op = JaDa.Op(interface.mass_matrix())
    jada_interface = JaDa.Interface(interface, jac_op, mass_op, len(x))

    alpha, beta, v, q, z = jdqz.jdqz(jac_op, mass_op, num_evs, tol=tol, subspace_dimensions=[30, 60],
                                     return_eigenvectors=True, return_subspaces=True, prec=jada_interface.prec)

    jdqz_eigs = numpy.array(sorted(alpha / beta, key=lambda x: abs(x)))
    jdqz_eigs = jdqz_eigs[:num_evs]

    assert_allclose(jdqz_eigs.real, arpack_eigs.real, rtol=0, atol=atol)
    assert_allclose(abs(jdqz_eigs.imag), abs(arpack_eigs.imag), rtol=0, atol=atol)

    H = utils.dot(q, q)
    assert_allclose(H, numpy.eye(q.shape[1]), rtol=0, atol=atol)

    if not interactive:
        return

    fig, ax = plt.subplots()
    ax.scatter(jdqz_eigs.real, jdqz_eigs.imag, marker='+')
    plt.show()

def test_complex_prec_2D(arpack_eigs, interface, x, num_evs, tol, atol, interactive=False):
    from transiflow.interface import JaDa
    from jadapy import jdqz

    numpy.random.seed(1234)

    jac_op = JaDa.Op(interface.jacobian(x))
    mass_op = JaDa.Op(interface.mass_matrix())
    jada_interface = JaDa.Interface(interface, jac_op, mass_op, len(x), numpy.complex128)

    alpha, beta, v, q, z = jdqz.jdqz(jac_op, mass_op, num_evs, tol=tol, subspace_dimensions=[30, 60],
                                     arithmetic='complex', return_eigenvectors=True, return_subspaces=True,
                                     prec=jada_interface.prec)
    check_eigenvalues(jac_op, mass_op, alpha / beta, v, num_evs, atol)

    jdqz_eigs = numpy.array(sorted(alpha / beta, key=lambda x: abs(x)))

    assert_allclose(jdqz_eigs.real, arpack_eigs.real, rtol=0, atol=atol)
    assert_allclose(abs(jdqz_eigs.imag), abs(arpack_eigs.imag), rtol=0, atol=atol)

    H = utils.dot(q, q)
    assert_allclose(H, numpy.eye(q.shape[1]), rtol=0, atol=atol)

    if not interactive:
        return

    fig, ax = plt.subplots()
    ax.scatter(jdqz_eigs.real, jdqz_eigs.imag, marker='+')
    plt.show()

def test_complex_shifted_prec_2D(arpack_eigs, interface, x, num_evs, tol, atol, interactive=False):
    from transiflow.interface import JaDa
    from jadapy import jdqz

    numpy.random.seed(1234)

    jac_op = JaDa.Op(interface.jacobian(x))
    mass_op = JaDa.Op(interface.mass_matrix())
    jada_interface = JaDa.Interface(interface, jac_op, mass_op, len(x), numpy.complex128)

    alpha, beta, v, q, z = jdqz.jdqz(jac_op, mass_op, num_evs, tol=tol, subspace_dimensions=[30, 60],
                                     arithmetic='complex', return_eigenvectors=True, return_subspaces=True,
                                     prec=jada_interface.shifted_prec)
    check_eigenvalues(jac_op, mass_op, alpha / beta, v, num_evs, atol)

    jdqz_eigs = numpy.array(sorted(alpha / beta, key=lambda x: abs(x)))

    assert_allclose(jdqz_eigs.real, arpack_eigs.real, rtol=0, atol=atol)
    assert_allclose(abs(jdqz_eigs.imag), abs(arpack_eigs.imag), rtol=0, atol=atol)

    H = utils.dot(q, q)
    assert_allclose(H, numpy.eye(q.shape[1]), rtol=0, atol=atol)

    if not interactive:
        return

    fig, ax = plt.subplots()
    ax.scatter(jdqz_eigs.real, jdqz_eigs.imag, marker='+')
    plt.show()

def test_complex_solve_2D(arpack_eigs, interface, x, num_evs, tol, atol, interactive=False):
    from transiflow.interface import JaDa
    from jadapy import jdqz

    numpy.random.seed(1234)

    jac_op = JaDa.Op(interface.jacobian(x))
    mass_op = JaDa.Op(interface.mass_matrix())
    jada_interface = JaDa.Interface(interface, jac_op, mass_op, len(x), numpy.complex128)

    assert not jada_interface.preconditioned_solve
    assert not jada_interface.shifted

    alpha, beta, v, q, z = jdqz.jdqz(jac_op, mass_op, num_evs, tol=tol, subspace_dimensions=[30, 60],
                                     arithmetic='complex', return_eigenvectors=True, return_subspaces=True,
                                     interface=jada_interface)
    check_eigenvalues(jac_op, mass_op, alpha / beta, v, num_evs, atol)

    jdqz_eigs = numpy.array(sorted(alpha / beta, key=lambda x: abs(x)))

    assert_allclose(jdqz_eigs.real, arpack_eigs.real, rtol=0, atol=atol)
    assert_allclose(abs(jdqz_eigs.imag), abs(arpack_eigs.imag), rtol=0, atol=atol)

    H = utils.dot(q, q)
    assert_allclose(H, numpy.eye(q.shape[1]), rtol=0, atol=atol)

    if not interactive:
        return

    fig, ax = plt.subplots()
    ax.scatter(jdqz_eigs.real, jdqz_eigs.imag, marker='+')
    plt.show()

def test_complex_prec_solve_2D(arpack_eigs, interface, x, num_evs, tol, atol, interactive=False):
    from transiflow.interface import JaDa
    from jadapy import jdqz

    numpy.random.seed(1234)

    jac_op = JaDa.Op(interface.jacobian(x))
    mass_op = JaDa.Op(interface.mass_matrix())
    jada_interface = JaDa.Interface(interface, jac_op, mass_op, len(x), numpy.complex128,
                                    preconditioned_solve=True)

    assert jada_interface.preconditioned_solve
    assert not jada_interface.shifted

    alpha, beta, v, q, z = jdqz.jdqz(jac_op, mass_op, num_evs, tol=tol, subspace_dimensions=[30, 60],
                                     arithmetic='complex', return_eigenvectors=True, return_subspaces=True,
                                     interface=jada_interface)
    check_eigenvalues(jac_op, mass_op, alpha / beta, v, num_evs, atol)

    jdqz_eigs = numpy.array(sorted(alpha / beta, key=lambda x: abs(x)))

    assert_allclose(jdqz_eigs.real, arpack_eigs.real, rtol=0, atol=atol)
    assert_allclose(abs(jdqz_eigs.imag), abs(arpack_eigs.imag), rtol=0, atol=atol)

    H = utils.dot(q, q)
    assert_allclose(H, numpy.eye(q.shape[1]), rtol=0, atol=atol)

    if not interactive:
        return

    fig, ax = plt.subplots()
    ax.scatter(jdqz_eigs.real, jdqz_eigs.imag, marker='+')
    plt.show()

def test_complex_shifted_prec_solve_2D(arpack_eigs, interface, x, num_evs, tol, atol, interactive=False):
    from transiflow.interface import JaDa
    from jadapy import jdqz

    numpy.random.seed(1234)

    jac_op = JaDa.Op(interface.jacobian(x))
    mass_op = JaDa.Op(interface.mass_matrix())
    jada_interface = JaDa.Interface(interface, jac_op, mass_op, len(x), numpy.complex128,
                                    preconditioned_solve=True, shifted=True)

    assert jada_interface.preconditioned_solve
    assert jada_interface.shifted

    alpha, beta, v, q, z = jdqz.jdqz(jac_op, mass_op, num_evs, tol=tol, subspace_dimensions=[30, 60],
                                     arithmetic='complex', return_eigenvectors=True, return_subspaces=True,
                                     interface=jada_interface)
    check_eigenvalues(jac_op, mass_op, alpha / beta, v, num_evs, atol)

    jdqz_eigs = numpy.array(sorted(alpha / beta, key=lambda x: abs(x)))

    assert_allclose(jdqz_eigs.real, arpack_eigs.real, rtol=0, atol=atol)
    assert_allclose(abs(jdqz_eigs.imag), abs(arpack_eigs.imag), rtol=0, atol=atol)

    H = utils.dot(q, q)
    assert_allclose(H, numpy.eye(q.shape[1]), rtol=0, atol=atol)

    if not interactive:
        return

    fig, ax = plt.subplots()
    ax.scatter(jdqz_eigs.real, jdqz_eigs.imag, marker='+')
    plt.show()

def test_complex_shifted_prec_bordered_solve_2D(arpack_eigs, interface, x, num_evs, tol, atol, interactive=False):
    from transiflow.interface import JaDa
    from jadapy import jdqz

    numpy.random.seed(1234)

    jac_op = JaDa.Op(interface.jacobian(x))
    mass_op = JaDa.Op(interface.mass_matrix())
    jada_interface = JaDa.BorderedInterface(interface, jac_op, mass_op, len(x), numpy.complex128)

    alpha, beta, v, q, z = jdqz.jdqz(jac_op, mass_op, num_evs, tol=tol, subspace_dimensions=[30, 60],
                                     arithmetic='complex', return_eigenvectors=True, return_subspaces=True,
                                     interface=jada_interface)
    check_eigenvalues(jac_op, mass_op, alpha / beta, v, num_evs, atol)

    jdqz_eigs = numpy.array(sorted(alpha / beta, key=lambda x: abs(x)))

    assert_allclose(jdqz_eigs.real, arpack_eigs.real, rtol=0, atol=atol)
    assert_allclose(abs(jdqz_eigs.imag), abs(arpack_eigs.imag), rtol=0, atol=atol)

    H = utils.dot(q, q)
    assert_allclose(H, numpy.eye(q.shape[1]), rtol=0, atol=atol)
