import numpy
import pytest

from numpy.testing import assert_allclose

import matplotlib.pyplot as plt

# Import common fixtures
from test.jada_fixtures import * # noqa: F401, F403
from test.jada_fixtures import check_eigenvalues

@pytest.fixture(autouse=True, scope='module')
def import_test():
    try:
        from fvm import JadaInterface # noqa: F401
    except ImportError:
        pytest.skip('jadapy not found')

def test_2D(arpack_eigs, interface, x, num_evs, tol, atol, interactive=False):
    from fvm import JadaInterface
    from jadapy import jdqz

    numpy.random.seed(1234)

    jac_op = JadaInterface.JadaOp(interface.jacobian(x))
    mass_op = JadaInterface.JadaOp(interface.mass_matrix())

    alpha, beta = jdqz.jdqz(jac_op, mass_op, num_evs, tol=tol, subspace_dimensions=[10, 20], target=0.1)
    jdqz_eigs = numpy.array(sorted(alpha / beta, key=lambda x: abs(x)))

    assert_allclose(jdqz_eigs.real, arpack_eigs.real, rtol=0, atol=atol)
    assert_allclose(abs(jdqz_eigs.imag), abs(arpack_eigs.imag), rtol=0, atol=atol)

    if not interactive:
        return x

    fig, ax = plt.subplots()
    ax.scatter(jdqz_eigs.real, jdqz_eigs.imag, marker='+')
    plt.show()

def test_complex_2D(arpack_eigs, interface, x, num_evs, tol, atol, interactive=False):
    from fvm import JadaInterface
    from jadapy import jdqz

    numpy.random.seed(1234)

    jac_op = JadaInterface.JadaOp(interface.jacobian(x))
    mass_op = JadaInterface.JadaOp(interface.mass_matrix())

    alpha, beta, v = jdqz.jdqz(jac_op, mass_op, num_evs, tol=tol, subspace_dimensions=[20, 40],
                               arithmetic='complex', return_eigenvectors=True)
    check_eigenvalues(jac_op, mass_op, alpha / beta, v, num_evs, atol)

    jdqz_eigs = numpy.array(sorted(alpha / beta, key=lambda x: abs(x)))

    assert_allclose(jdqz_eigs.real, arpack_eigs.real, rtol=0, atol=atol)
    assert_allclose(abs(jdqz_eigs.imag), abs(arpack_eigs.imag), rtol=0, atol=atol)

    if not interactive:
        return x

    fig, ax = plt.subplots()
    ax.scatter(jdqz_eigs.real, jdqz_eigs.imag, marker='+')
    plt.show()

def test_prec_2D(arpack_eigs, interface, x, num_evs, tol, atol, interactive=False):
    from fvm import JadaInterface
    from jadapy import jdqz

    numpy.random.seed(1234)

    jac_op = JadaInterface.JadaOp(interface.jacobian(x))
    mass_op = JadaInterface.JadaOp(interface.mass_matrix())
    jada_interface = JadaInterface.JadaInterface(interface, jac_op, mass_op, len(x))

    alpha, beta, v = jdqz.jdqz(jac_op, mass_op, num_evs, tol=tol, subspace_dimensions=[20, 40],
                               return_eigenvectors=True, prec=jada_interface.prec)

    jdqz_eigs = numpy.array(sorted(alpha / beta, key=lambda x: abs(x)))

    assert_allclose(jdqz_eigs.real, arpack_eigs.real, rtol=0, atol=atol)
    assert_allclose(abs(jdqz_eigs.imag), abs(arpack_eigs.imag), rtol=0, atol=atol)

    if not interactive:
        return x

    fig, ax = plt.subplots()
    ax.scatter(jdqz_eigs.real, jdqz_eigs.imag, marker='+')
    plt.show()

def test_complex_prec_2D(arpack_eigs, interface, x, num_evs, tol, atol, interactive=False):
    from fvm import JadaInterface
    from jadapy import jdqz

    numpy.random.seed(1234)

    jac_op = JadaInterface.JadaOp(interface.jacobian(x))
    mass_op = JadaInterface.JadaOp(interface.mass_matrix())
    jada_interface = JadaInterface.JadaInterface(interface, jac_op, mass_op, len(x), numpy.complex128)

    alpha, beta, v = jdqz.jdqz(jac_op, mass_op, num_evs, tol=tol, subspace_dimensions=[20, 40],
                               arithmetic='complex', return_eigenvectors=True, prec=jada_interface.prec)
    check_eigenvalues(jac_op, mass_op, alpha / beta, v, num_evs, atol)

    jdqz_eigs = numpy.array(sorted(alpha / beta, key=lambda x: abs(x)))

    assert_allclose(jdqz_eigs.real, arpack_eigs.real, rtol=0, atol=atol)
    assert_allclose(abs(jdqz_eigs.imag), abs(arpack_eigs.imag), rtol=0, atol=atol)

    if not interactive:
        return x

    fig, ax = plt.subplots()
    ax.scatter(jdqz_eigs.real, jdqz_eigs.imag, marker='+')
    plt.show()

def test_complex_shifted_prec_2D(arpack_eigs, interface, x, num_evs, tol, atol, interactive=False):
    from fvm import JadaInterface
    from jadapy import jdqz

    numpy.random.seed(1234)

    jac_op = JadaInterface.JadaOp(interface.jacobian(x))
    mass_op = JadaInterface.JadaOp(interface.mass_matrix())
    jada_interface = JadaInterface.JadaInterface(interface, jac_op, mass_op, len(x), numpy.complex128)

    alpha, beta, v = jdqz.jdqz(jac_op, mass_op, num_evs, tol=tol, subspace_dimensions=[20, 40],
                               arithmetic='complex', return_eigenvectors=True, prec=jada_interface.shifted_prec)
    check_eigenvalues(jac_op, mass_op, alpha / beta, v, num_evs, atol)

    jdqz_eigs = numpy.array(sorted(alpha / beta, key=lambda x: abs(x)))

    assert_allclose(jdqz_eigs.real, arpack_eigs.real, rtol=0, atol=atol)
    assert_allclose(abs(jdqz_eigs.imag), abs(arpack_eigs.imag), rtol=0, atol=atol)

    if not interactive:
        return x

    fig, ax = plt.subplots()
    ax.scatter(jdqz_eigs.real, jdqz_eigs.imag, marker='+')
    plt.show()

def test_complex_solve_2D(arpack_eigs, interface, x, num_evs, tol, atol, interactive=False):
    from fvm import JadaInterface
    from jadapy import jdqz

    numpy.random.seed(1234)

    jac_op = JadaInterface.JadaOp(interface.jacobian(x))
    mass_op = JadaInterface.JadaOp(interface.mass_matrix())
    jada_interface = JadaInterface.JadaInterface(interface, jac_op, mass_op, len(x), numpy.complex128)

    assert not jada_interface.preconditioned_solve
    assert not jada_interface.shifted

    alpha, beta, v = jdqz.jdqz(jac_op, mass_op, num_evs, tol=tol, subspace_dimensions=[20, 40],
                               arithmetic='complex', return_eigenvectors=True, interface=jada_interface)
    check_eigenvalues(jac_op, mass_op, alpha / beta, v, num_evs, atol)

    jdqz_eigs = numpy.array(sorted(alpha / beta, key=lambda x: abs(x)))

    assert_allclose(jdqz_eigs.real, arpack_eigs.real, rtol=0, atol=atol)
    assert_allclose(abs(jdqz_eigs.imag), abs(arpack_eigs.imag), rtol=0, atol=atol)

    if not interactive:
        return x

    fig, ax = plt.subplots()
    ax.scatter(jdqz_eigs.real, jdqz_eigs.imag, marker='+')
    plt.show()

def test_complex_prec_solve_2D(arpack_eigs, interface, x, num_evs, tol, atol, interactive=False):
    from fvm import JadaInterface
    from jadapy import jdqz

    numpy.random.seed(1234)

    jac_op = JadaInterface.JadaOp(interface.jacobian(x))
    mass_op = JadaInterface.JadaOp(interface.mass_matrix())
    jada_interface = JadaInterface.JadaInterface(interface, jac_op, mass_op, len(x), numpy.complex128,
                                                 preconditioned_solve=True)

    assert jada_interface.preconditioned_solve
    assert not jada_interface.shifted

    alpha, beta, v = jdqz.jdqz(jac_op, mass_op, num_evs, tol=tol, subspace_dimensions=[20, 40],
                               arithmetic='complex', return_eigenvectors=True, interface=jada_interface)
    check_eigenvalues(jac_op, mass_op, alpha / beta, v, num_evs, atol)

    jdqz_eigs = numpy.array(sorted(alpha / beta, key=lambda x: abs(x)))

    assert_allclose(jdqz_eigs.real, arpack_eigs.real, rtol=0, atol=atol)
    assert_allclose(abs(jdqz_eigs.imag), abs(arpack_eigs.imag), rtol=0, atol=atol)

    if not interactive:
        return x

    fig, ax = plt.subplots()
    ax.scatter(jdqz_eigs.real, jdqz_eigs.imag, marker='+')
    plt.show()

def test_complex_shifted_prec_solve_2D(arpack_eigs, interface, x, num_evs, tol, atol, interactive=False):
    from fvm import JadaInterface
    from jadapy import jdqz

    numpy.random.seed(1234)

    jac_op = JadaInterface.JadaOp(interface.jacobian(x))
    mass_op = JadaInterface.JadaOp(interface.mass_matrix())
    jada_interface = JadaInterface.JadaInterface(interface, jac_op, mass_op, len(x), numpy.complex128,
                                                 preconditioned_solve=True, shifted=True)

    assert jada_interface.preconditioned_solve
    assert jada_interface.shifted

    alpha, beta, v = jdqz.jdqz(jac_op, mass_op, num_evs, tol=tol, subspace_dimensions=[20, 40],
                               arithmetic='complex', return_eigenvectors=True, interface=jada_interface)
    check_eigenvalues(jac_op, mass_op, alpha / beta, v, num_evs, atol)

    jdqz_eigs = numpy.array(sorted(alpha / beta, key=lambda x: abs(x)))

    assert_allclose(jdqz_eigs.real, arpack_eigs.real, rtol=0, atol=atol)
    assert_allclose(abs(jdqz_eigs.imag), abs(arpack_eigs.imag), rtol=0, atol=atol)

    if not interactive:
        return x

    fig, ax = plt.subplots()
    ax.scatter(jdqz_eigs.real, jdqz_eigs.imag, marker='+')
    plt.show()
