import numpy

from fvm.interface.SciPy import Interface


def test_solve(nx=4):
    dim = 3
    dof = 4
    ny = nx
    nz = nx

    parameters = {}
    interface = Interface(parameters, nx, ny, nz, dim, dof)

    n = dof * nx * ny * nz

    x = numpy.random.random(n)

    x0 = numpy.zeros(n)
    A = interface.jacobian(x0)

    b = A @ x

    y = interface.solve(A, b)

    assert numpy.linalg.norm(y) > 0
    assert numpy.linalg.norm(y - x) - nx * ny * nz * x[dim] < 1e-11

def test_bordered_matrix(nx=4):
    dim = 3
    dof = 4
    ny = nx
    nz = nx

    parameters = {}
    interface = Interface(parameters, nx, ny, nz, dim, dof)
    interface.border_scaling = 1

    n = dof * nx * ny * nz
    n2 = 3

    x = numpy.random.random(n)
    x2 = numpy.random.random(n2)
    x3 = numpy.append(x, x2)

    V = numpy.random.random((n, n2))
    W = numpy.random.random((n, n2))
    C = numpy.random.random((n2, n2))

    x0 = numpy.zeros(n)
    A = interface.jacobian(x0)

    b = A @ x + V @ x2
    b2 = W.T @ x + C @ x2
    b3 = numpy.append(b, b2)

    B = interface.compute_bordered_matrix(A, V, W, C)

    y = B @ x3

    assert numpy.linalg.norm(y) > 0
    assert numpy.linalg.norm(y - b3) < 1e-11
