import numpy
import pytest

from numpy.testing import assert_allclose

import matplotlib.pyplot as plt

from transiflow import Continuation

# Import common fixtures
from tests.jada_fixtures import * # noqa: F401, F403

@pytest.fixture(autouse=True, scope='module')
def import_test():
    try:
        from transiflow.interface import JaDa # noqa: F401
    except ImportError:
        pytest.skip('jadapy not found')

    try:
        from transiflow.interface import HYMLS as HYMLSInterface # noqa: F401
    except ImportError:
        pytest.skip('HYMLS not found')

@pytest.fixture(scope='module')
def interface(nx):
    from transiflow.interface import HYMLS as HYMLSInterface
    from PyTrilinos import Teuchos

    dim = 2
    ny = nx
    nz = 1

    parameters = Teuchos.ParameterList()
    parameters.set('Reynolds Number', 0)
    parameters.set('Bordered Solver', True)

    interface = HYMLSInterface.Interface(parameters, nx, ny, nz, dim)

    return interface

@pytest.fixture(scope='module')
def x(interface):
    from transiflow.interface import HYMLS as HYMLSInterface

    continuation = Continuation(interface)

    x0 = HYMLSInterface.Vector(interface.map)
    x0 = continuation.newton(x0)

    start = 0
    target = 2000
    ds = 100
    return continuation.continuation(x0, 'Reynolds Number', start, target, ds)[0]

class Operator:
    def __init__(self, A):
        self.A = A

    def matvec(self, x):
        return self.A @ x

    def proj(self, x):
        return x

def test_solve(interface, x, tol):
    from transiflow.interface import HYMLS as HYMLSInterface
    from transiflow.interface import JaDaHYMLS

    from jadapy import EpetraInterface
    from jadapy.utils import norm

    from PyTrilinos import Epetra

    interface.teuchos_parameters.sublist('Preconditioner').set('Number of Levels', 0)
    interface.initialize()

    x_sol = HYMLSInterface.Vector(interface.solve_map)
    x_sol.Import(x, interface.solve_importer, Epetra.Insert)

    # Create a test vector to remove the nonzero first pressure
    t = HYMLSInterface.Vector(x_sol)
    t.PutScalar(0.0)
    for i in range(t.MyLength()):
        if t.Map().GID(i) != interface.dim:
            t[i] = 1.0

    jac_op = EpetraInterface.CrsMatrix(interface.jacobian(x))
    jada_interface = JaDaHYMLS.Interface(interface, preconditioned_solve=True)

    op = Operator(jac_op)
    b = op.matvec(x_sol)

    x = jada_interface.solve(op, b, tol, maxit=1)

    r = op.matvec(x) - b

    # Remove the nonzero first pressure
    r.Multiply(1.0, r, t, 0.0)

    assert norm(x) > tol
    assert norm(r) / norm(b) < tol

def test_prec_2D(arpack_eigs, interface, x, num_evs, tol, atol, interactive=False):
    from transiflow.interface import JaDaHYMLS

    from jadapy import EpetraInterface
    from jadapy import jdqz

    numpy.random.seed(1234)

    jac_op = EpetraInterface.CrsMatrix(interface.jacobian(x))
    mass_op = EpetraInterface.CrsMatrix(interface.mass_matrix())
    jada_interface = JaDaHYMLS.Interface(interface, preconditioned_solve=False)

    alpha, beta = jdqz.jdqz(jac_op, mass_op, num_evs, tol=tol, subspace_dimensions=[20, 40],
                            interface=jada_interface, prec=jada_interface.prec)
    jdqz_eigs = numpy.array(sorted(alpha / beta, key=lambda x: abs(x)))

    assert_allclose(jdqz_eigs.real, arpack_eigs.real, rtol=0, atol=atol)
    assert_allclose(abs(jdqz_eigs.imag), abs(arpack_eigs.imag), rtol=0, atol=atol)

    if not interactive:
        return

    fig, ax = plt.subplots()
    ax.scatter(jdqz_eigs.real, jdqz_eigs.imag, marker='+')
    plt.show()

def test_prec_solve_2D(arpack_eigs, interface, x, num_evs, tol, atol, interactive=False):
    from transiflow.interface import JaDaHYMLS

    from jadapy import EpetraInterface
    from jadapy import jdqz

    numpy.random.seed(1234)

    jac_op = EpetraInterface.CrsMatrix(interface.jacobian(x))
    mass_op = EpetraInterface.CrsMatrix(interface.mass_matrix())
    jada_interface = JaDaHYMLS.Interface(interface, preconditioned_solve=True)

    alpha, beta = jdqz.jdqz(jac_op, mass_op, num_evs, tol=tol, subspace_dimensions=[20, 40],
                            interface=jada_interface)
    jdqz_eigs = numpy.array(sorted(alpha / beta, key=lambda x: abs(x)))

    assert_allclose(jdqz_eigs.real, arpack_eigs.real, rtol=0, atol=atol)
    assert_allclose(abs(jdqz_eigs.imag), abs(arpack_eigs.imag), rtol=0, atol=atol)

    if not interactive:
        return

    fig, ax = plt.subplots()
    ax.scatter(jdqz_eigs.real, jdqz_eigs.imag, marker='+')
    plt.show()

def test_complex_prec_2D(arpack_eigs, interface, x, num_evs, tol, atol, interactive=False):
    from transiflow.interface import JaDaHYMLS

    from jadapy import ComplexEpetraInterface
    from jadapy import jdqz

    numpy.random.seed(1234)

    jac_op = ComplexEpetraInterface.CrsMatrix(interface.jacobian(x))
    mass_op = ComplexEpetraInterface.CrsMatrix(interface.mass_matrix())
    jada_interface = JaDaHYMLS.ComplexInterface(interface, preconditioned_solve=False)

    alpha, beta = jdqz.jdqz(jac_op, mass_op, num_evs, tol=tol, subspace_dimensions=[20, 40],
                            interface=jada_interface, prec=jada_interface.prec)
    jdqz_eigs = numpy.array(sorted(alpha / beta, key=lambda x: abs(x)))

    assert_allclose(jdqz_eigs.real, arpack_eigs.real, rtol=0, atol=atol)
    assert_allclose(abs(jdqz_eigs.imag), abs(arpack_eigs.imag), rtol=0, atol=atol)

    if not interactive:
        return

    fig, ax = plt.subplots()
    ax.scatter(jdqz_eigs.real, jdqz_eigs.imag, marker='+')
    plt.show()

def test_complex_prec_solve_2D(arpack_eigs, interface, x, num_evs, tol, atol, interactive=False):
    from transiflow.interface import JaDaHYMLS

    from jadapy import ComplexEpetraInterface
    from jadapy import jdqz

    numpy.random.seed(1234)

    jac_op = ComplexEpetraInterface.CrsMatrix(interface.jacobian(x))
    mass_op = ComplexEpetraInterface.CrsMatrix(interface.mass_matrix())
    jada_interface = JaDaHYMLS.ComplexInterface(interface, preconditioned_solve=True)

    alpha, beta = jdqz.jdqz(jac_op, mass_op, num_evs, tol=tol, subspace_dimensions=[20, 40],
                            interface=jada_interface)
    jdqz_eigs = numpy.array(sorted(alpha / beta, key=lambda x: abs(x)))

    assert_allclose(jdqz_eigs.real, arpack_eigs.real, rtol=0, atol=atol)
    assert_allclose(abs(jdqz_eigs.imag), abs(arpack_eigs.imag), rtol=0, atol=atol)

    if not interactive:
        return

    fig, ax = plt.subplots()
    ax.scatter(jdqz_eigs.real, jdqz_eigs.imag, marker='+')
    plt.show()

def test_bordered_prec_solve_2D(arpack_eigs, interface, x, num_evs, tol, atol, interactive=False):
    from transiflow.interface import JaDaHYMLS

    from jadapy import EpetraInterface
    from jadapy import jdqz

    numpy.random.seed(1234)

    jac_op = EpetraInterface.CrsMatrix(interface.jacobian(x))
    mass_op = EpetraInterface.CrsMatrix(interface.mass_matrix())
    jada_interface = JaDaHYMLS.BorderedInterface(interface)

    alpha, beta = jdqz.jdqz(jac_op, mass_op, num_evs, tol=tol, subspace_dimensions=[20, 40],
                            interface=jada_interface)
    jdqz_eigs = numpy.array(sorted(alpha / beta, key=lambda x: abs(x)))

    assert_allclose(jdqz_eigs.real, arpack_eigs.real, rtol=0, atol=atol)
    assert_allclose(abs(jdqz_eigs.imag), abs(arpack_eigs.imag), rtol=0, atol=atol)

    if not interactive:
        return

    fig, ax = plt.subplots()
    ax.scatter(jdqz_eigs.real, jdqz_eigs.imag, marker='+')
    plt.show()
