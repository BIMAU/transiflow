import pytest

import numpy

from numpy.testing import assert_allclose

from scipy import sparse

from fvm import Continuation
from fvm import Interface

@pytest.fixture(autouse=True, scope='module')
def import_test():
    try:
        from fvm import JadaInterface # noqa: F401
    except ImportError:
        pytest.skip('jadapy not found')

@pytest.fixture(scope='module')
def nx():
    return 4

@pytest.fixture(scope='module')
def tol():
    return 1e-7

@pytest.fixture(scope='module')
def atol(tol):
    return tol * 100

@pytest.fixture(scope='module')
def num_evs():
    return 9

@pytest.fixture(scope='module')
def numpy_interface(nx):
    dim = 2
    dof = 3
    ny = nx
    nz = 1

    parameters = {'Reynolds Number': 0}
    interface = Interface(parameters, nx, ny, nz, dim, dof)

    return interface

@pytest.fixture(scope='module')
def numpy_x(numpy_interface):
    n = numpy_interface.dof * numpy_interface.nx * numpy_interface.ny * numpy_interface.nz

    parameters = {}
    continuation = Continuation(numpy_interface, parameters)

    x0 = numpy.zeros(n)
    x0 = continuation.newton(x0)

    target = 2000
    ds = 100
    maxit = 20
    return continuation.continuation(x0, 'Reynolds Number', target, ds, maxit)

def check_eigenvalues(A_op, B_op, eigs, v, num_evs, tol):
    from jadapy.utils import norm

    idx = range(len(eigs))
    idx = numpy.array(sorted(idx, key=lambda i: abs(eigs[i])))

    for i in range(num_evs):
        j = idx[i]
        assert norm(A_op @ v[:, j]) > tol
        assert_allclose(norm(A_op @ v[:, j] - B_op @ v[:, j] * eigs[j]), 0, rtol=0, atol=tol)

@pytest.fixture(scope='module')
def arpack_eigs(numpy_interface, numpy_x, num_evs, tol, atol):
    from fvm import JadaInterface

    A_op = JadaInterface.JadaOp(numpy_interface.jacobian(numpy_x))
    B_op = JadaInterface.JadaOp(numpy_interface.mass_matrix())

    # A_mat = A_op.mat.todense()
    # B_mat = B_op.mat.todense()

    # eigs, v = scipy.linalg.eig(A_mat, B_mat, left=False, right=True)
    eigs, v = sparse.linalg.eigs(A_op, num_evs, B_op, sigma=0.1, tol=tol)

    check_eigenvalues(A_op, B_op, eigs, v, num_evs, atol)
    eigs = numpy.array(sorted(eigs, key=lambda x: abs(x)))
    return eigs[:num_evs]

@pytest.fixture(scope='module')
def interface(numpy_interface):
    return numpy_interface

@pytest.fixture(scope='module')
def x(numpy_x):
    return numpy_x
