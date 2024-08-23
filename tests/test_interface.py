import numpy
import os

from transiflow import Interface


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

def test_save_load(nx=4):
    ny = nx
    nz = nx

    parameters = {'Eigenvalue Solver': {'Target': 1 + 3j}}
    interface = Interface(parameters, nx, ny, nz)

    x = interface.vector()
    n = len(x)

    x = numpy.random.random(n)
    interface.save_state('x-test', x)

    assert os.path.isfile('x-test.npy')
    assert os.path.isfile('x-test.params')

    parameters['Eigenvalue Solver']['Target'] = 33

    x2 = interface.load_state('x-test')
    assert parameters['Eigenvalue Solver']['Target'] == 1 + 3j
    assert numpy.linalg.norm(x - x2) < 1e-14
