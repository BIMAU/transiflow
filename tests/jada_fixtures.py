import pytest

import numpy

from numpy.testing import assert_allclose

from scipy import sparse

from transiflow import Continuation

from transiflow.interface.SciPy import Interface


@pytest.fixture(autouse=True, scope='module')
def import_test():
    try:
        from transiflow.interface import JaDa # noqa: F401
    except ImportError:
        pytest.skip('jadapy not found')

@pytest.fixture(scope='module')
def nx():
    return 6

@pytest.fixture(scope='module')
def tol():
    return 1e-7

@pytest.fixture(scope='module')
def atol(tol):
    return tol * 100

@pytest.fixture(scope='module')
def num_evs():
    return 10

@pytest.fixture(scope='module')
def scipy_interface(nx):
    dim = 2
    dof = 3
    ny = nx
    nz = 1

    parameters = {}
    interface = Interface(parameters, nx, ny, nz, dim, dof)

    return interface

@pytest.fixture(scope='module')
def scipy_x(scipy_interface):
    n = scipy_interface.dof * scipy_interface.nx * scipy_interface.ny * scipy_interface.nz

    continuation = Continuation(scipy_interface)

    x0 = numpy.zeros(n)
    x0 = continuation.newton(x0)

    start = 0
    target = 2000
    ds = 100
    return continuation.continuation(x0, 'Reynolds Number', start, target, ds)[0]

def check_eigenvalues(A_op, B_op, eigs, v, num_evs, tol):
    from jadapy.utils import norm

    idx = range(len(eigs))
    idx = numpy.array(sorted(idx, key=lambda i: abs(eigs[i])))

    for i in range(num_evs):
        j = idx[i]
        assert norm(A_op @ v[:, j]) > tol
        assert_allclose(norm(A_op @ v[:, j] - B_op @ v[:, j] * eigs[j]), 0, rtol=0, atol=tol)

@pytest.fixture(scope='module')
def arpack_eigs(scipy_interface, scipy_x, num_evs, tol, atol):
    from transiflow.interface import JaDa

    A_op = JaDa.Op(scipy_interface.jacobian(scipy_x))
    B_op = JaDa.Op(scipy_interface.mass_matrix())

    # A_mat = A_op.mat.todense()
    # B_mat = B_op.mat.todense()

    # eigs, v = scipy.linalg.eig(A_mat, B_mat, left=False, right=True)
    eigs, v = sparse.linalg.eigs(A_op, num_evs, B_op, sigma=0.1, tol=tol)

    check_eigenvalues(A_op, B_op, eigs, v, num_evs, atol)
    eigs = numpy.array(sorted(eigs, key=lambda x: abs(x)))
    return eigs[:num_evs]

@pytest.fixture(scope='module')
def interface(scipy_interface):
    return scipy_interface

@pytest.fixture(scope='module')
def x(scipy_x):
    return scipy_x
