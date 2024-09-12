import numpy
import os
import pytest

from transiflow import utils
from transiflow.interface import create


def Interface(*args, backend="SciPy", **kwargs):
    try:
        return create(*args, backend=backend, **kwargs)
    except ImportError:
        pytest.skip(backend + " not found")


@pytest.mark.parametrize("backend", ["SciPy", "Epetra", "HYMLS", "PETSc"])
def test_solve(backend, nx=4):
    ny = nx
    nz = nx

    parameters = {}
    interface = Interface(parameters, nx, ny, nz, backend="SciPy")

    x0 = interface.vector()
    n = x0.size

    A = interface.jacobian(x0)

    x = numpy.zeros(n)
    for i in range(n):
        x[i] = i + 1

    pressure_node = 3
    x[pressure_node] = 0

    b = A @ x

    interface = Interface(parameters, nx, ny, nz, backend=backend)

    x0 = interface.vector()
    A = interface.jacobian(x0)
    b = interface.vector_from_array(b)

    y = interface.solve(A, b)
    y = interface.array_from_vector(y)

    pressure = y[pressure_node] - x[pressure_node]

    assert utils.norm(y) > 0
    assert (utils.norm(y - x) - pressure * nx * ny * nz) / utils.norm(b) < 1e-7

def test_bordered_matrix(nx=4):
    ny = nx
    nz = nx

    parameters = {}
    interface = Interface(parameters, nx, ny, nz)
    interface.border_scaling = 1

    x0 = interface.vector()
    n = x0.size
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


@pytest.mark.parametrize("backend", ["SciPy", "Epetra", "HYMLS", "PETSc"])
def test_custom_bc(backend, nx=4):
    def boundaries(boundary_conditions, atom):
        boundary_conditions.heat_flux_east(atom, 0)
        boundary_conditions.heat_flux_west(atom, 0)
        boundary_conditions.no_slip_east(atom)
        boundary_conditions.no_slip_west(atom)

        boundary_conditions.heat_flux_north(atom, 0)
        boundary_conditions.heat_flux_south(atom, 0)
        boundary_conditions.no_slip_north(atom)
        boundary_conditions.no_slip_south(atom)

        Bi = 0
        boundary_conditions.heat_flux_top(atom, 0, Bi)
        boundary_conditions.temperature_bottom(atom, 0)
        boundary_conditions.free_slip_top(atom)
        boundary_conditions.no_slip_bottom(atom)

        return boundary_conditions.get_forcing()

    ny = nx
    nz = nx
    dim = 3
    dof = 5

    parameters = {'Rayleigh Number': 100, 'Prandtl Number': 100}
    interface = Interface(parameters, nx, ny, nz, dim, dof, backend=backend,
                          boundary_conditions=boundaries)

    x = interface.vector()
    interface.rhs(x)
