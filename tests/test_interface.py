import numpy
import os
import pytest

from transiflow import utils
from transiflow.interface import create


def Interface(parameters, nx, ny, nz=1, dim=None, dof=None, backend="SciPy"):
    try:
        return create(parameters, nx, ny, nz, dim, dof, backend=backend)
    except ImportError:
        pytest.skip(backend + " not found")


def test_solve(nx=4):
    ny = nx
    nz = nx

    parameters = {}
    interface = Interface(parameters, nx, ny, nz)

    x0 = interface.vector()
    n = len(x0)

    A = interface.jacobian(x0)
    x = numpy.random.random(n)

    b = A @ x

    y = interface.solve(A, b)

    pressure = 3

    assert numpy.linalg.norm(y) > 0
    assert numpy.linalg.norm(y - x) - nx * ny * nz * x[pressure] < 1e-11

def test_bordered_matrix(nx=4):
    ny = nx
    nz = nx

    parameters = {}
    interface = Interface(parameters, nx, ny, nz)
    interface.border_scaling = 1

    x0 = interface.vector()
    n = len(x0)
    n2 = 3

    x = numpy.random.random(n)
    x2 = numpy.random.random(n2)
    x3 = numpy.append(x, x2)

    V = numpy.random.random((n, n2))
    W = numpy.random.random((n, n2))
    C = numpy.random.random((n2, n2))

    A = interface.jacobian(x0)

    b = A @ x + V @ x2
    b2 = W.T @ x + C @ x2
    b3 = numpy.append(b, b2)

    B = interface.compute_bordered_matrix(A, V, W, C)

    y = B @ x3

    assert numpy.linalg.norm(y) > 0
    assert numpy.linalg.norm(y - b3) < 1e-11


@pytest.mark.parametrize("backend", ["SciPy", "Epetra", "HYMLS", "PETSc"])
def test_save_load(backend, nx=4):
    ny = nx
    nz = nx

    try:
        from mpi4py import MPI
        if MPI.COMM_WORLD.Get_size() > 1 and backend == 'SciPy':
            pytest.skip('SciPy shouldn\'t be used in parallel')
    except ImportError:
        pass

    parameters = {'Eigenvalue Solver': {'Target': 1 + 3j}}
    interface = Interface(parameters, nx, ny, nz, backend=backend)

    x = interface.vector()
    n = x.size

    x = interface.vector_from_array(numpy.random.random(n))
    interface.save_state('x-test', x)

    assert os.path.isfile('x-test.npy')
    assert os.path.isfile('x-test.params')

    parameters['Eigenvalue Solver']['Target'] = 33

    x2 = interface.load_state('x-test')
    assert parameters['Eigenvalue Solver']['Target'] == 1 + 3j
    assert utils.norm(x - x2) < 1e-14
