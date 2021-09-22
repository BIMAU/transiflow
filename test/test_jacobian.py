import numpy

from fvm import utils
from fvm import Discretization, CylindricalDiscretization

def create_coordinate_vector(nx):
    dx = 1 / (nx + 1)
    a = []
    for i in range(nx+3):
        a.append(-dx + dx * 1.2 ** i)
    return numpy.roll(a, -2)

def create_test_problem():
    nx = 13
    ny = 7
    nz = 5
    dim = 3
    dof = 5

    x = create_coordinate_vector(nx)
    y = create_coordinate_vector(ny)
    z = create_coordinate_vector(nz)

    parameters = {'Reynolds Number': 100}

    return (parameters, nx, ny, nz, dim, dof, x, y, z)

def check_divfree(discretization, state):
    A = discretization.jacobian(state)
    x = A @ state
    for i in range(len(state)):
        if i % discretization.dof == discretization.dim:
            assert abs(x[i]) < 1e-14

def make_divfree(discretization, state):
    A = discretization.jacobian(state)
    p = numpy.zeros((A.n, A.n // discretization.dof))

    for i in range(A.n):
        if i % discretization.dof == discretization.dim:
            for j in range(A.begA[i], A.begA[i+1]):
                p[A.jcoA[j], i // discretization.dof] = A.coA[j]

    state -= p @ numpy.linalg.solve(p.conj().T @ p, p.conj().T @ state)

    check_divfree(discretization, state)

    return state

def create_divfree_state(discretization):
    n = discretization.dof * discretization.nx * discretization.ny * discretization.nz

    state = numpy.random.random(n)
    return make_divfree(discretization, state)

def test_bilin():
    parameters, nx, ny, nz, dim, dof, x, y, z = create_test_problem()

    discretization = Discretization(parameters, nx, ny, nz, dim, dof, x, y, z)
    state = create_divfree_state(discretization)

    atomJ, atomF = discretization.nonlinear_part(state)
    A = discretization.assemble_jacobian(atomF)

    for i in range(A.n):
        for j in range(A.begA[i], A.begA[i+1]):
            assert i != A.jcoA[j]

def test_bilin_uniform():
    parameters, nx, ny, nz, dim, dof, x, y, z = create_test_problem()

    discretization = Discretization(parameters, nx, ny, nz, dim, dof)
    state = create_divfree_state(discretization)

    atomJ, atomF = discretization.nonlinear_part(state)
    A = discretization.assemble_jacobian(atomF)

    for i in range(A.n):
        for j in range(A.begA[i], A.begA[i+1]):
            assert i != A.jcoA[j]

def test_bilin_stretched():
    parameters, nx, ny, nz, dim, dof, x, y, z = create_test_problem()

    x = utils.create_stretched_coordinate_vector(0, 1, nx, 1.5)
    y = utils.create_stretched_coordinate_vector(0, 1, ny, 1.5)
    z = utils.create_stretched_coordinate_vector(0, 1, nz, 1.5)

    discretization = Discretization(parameters, nx, ny, nz, dim, dof, x, y, z)
    state = create_divfree_state(discretization)

    atomJ, atomF = discretization.nonlinear_part(state)
    A = discretization.assemble_jacobian(atomF)

    for i in range(A.n):
        for j in range(A.begA[i], A.begA[i+1]):
            assert i != A.jcoA[j]

def test_jac_consistency():
    parameters, nx, ny, nz, dim, dof, x, y, z = create_test_problem()

    n = dof * nx * ny * nz

    state = numpy.random.random(n)
    pert = numpy.random.random(n)

    discretization = Discretization(parameters, nx, ny, nz, dim, dof, x, y, z)
    A = discretization.jacobian(state)
    rhs = discretization.rhs(state)

    for i in range(3, 12):
        eps = 10 ** -i
        eps2 = max(eps, 10 ** (-14+i))
        rhs2 = discretization.rhs(state + eps * pert)
        assert numpy.linalg.norm((rhs2 - rhs) / eps - A @ pert) < eps2

def test_jac_consistency_uniform():
    parameters, nx, ny, nz, dim, dof, x, y, z = create_test_problem()

    n = dof * nx * ny * nz

    state = numpy.random.random(n)
    pert = numpy.random.random(n)

    discretization = Discretization(parameters, nx, ny, nz, dim, dof)
    A = discretization.jacobian(state)
    rhs = discretization.rhs(state)

    for i in range(3, 12):
        eps = 10 ** -i
        eps2 = max(eps, 10 ** (-14+i))
        rhs2 = discretization.rhs(state + eps * pert)
        assert numpy.linalg.norm((rhs2 - rhs) / eps - A @ pert) < eps2

def test_jac_consistency_stretched():
    parameters, nx, ny, nz, dim, dof, x, y, z = create_test_problem()

    x = utils.create_stretched_coordinate_vector(0, 1, nx, 1.5)
    y = utils.create_stretched_coordinate_vector(0, 1, ny, 1.5)
    z = utils.create_stretched_coordinate_vector(0, 1, nz, 1.5)

    n = dof * nx * ny * nz

    state = numpy.random.random(n)
    pert = numpy.random.random(n)

    discretization = Discretization(parameters, nx, ny, nz, dim, dof, x, y, z)
    A = discretization.jacobian(state)
    rhs = discretization.rhs(state)

    for i in range(3, 12):
        eps = 10 ** -i
        eps2 = max(eps, 10 ** (-14+i))
        rhs2 = discretization.rhs(state + eps * pert)
        assert numpy.linalg.norm((rhs2 - rhs) / eps - A @ pert) < eps2

def test_jac_consistency_uniform_2d():
    parameters, nx, ny, nz, dim, dof, x, y, z = create_test_problem()

    nz = 1
    dim = 2
    dof = 3
    n = dof * nx * ny * nz

    state = numpy.random.random(n)
    pert = numpy.random.random(n)

    discretization = Discretization(parameters, nx, ny, nz, dim, dof)
    A = discretization.jacobian(state)
    rhs = discretization.rhs(state)

    for i in range(3, 12):
        eps = 10 ** -i
        eps2 = max(eps, 10 ** (-14+i))
        rhs2 = discretization.rhs(state + eps * pert)
        assert numpy.linalg.norm((rhs2 - rhs) / eps - A @ pert) < eps2

def test_jac_consistency_stretched_2d():
    parameters, nx, ny, nz, dim, dof, x, y, z = create_test_problem()

    x = utils.create_stretched_coordinate_vector(0, 1, nx, 1.5)
    y = utils.create_stretched_coordinate_vector(0, 1, ny, 1.5)

    nz = 1
    dim = 2
    dof = 3
    n = dof * nx * ny * nz

    state = numpy.random.random(n)
    pert = numpy.random.random(n)

    discretization = Discretization(parameters, nx, ny, nz, dim, dof, x, y)
    A = discretization.jacobian(state)
    rhs = discretization.rhs(state)

    for i in range(3, 12):
        eps = 10 ** -i
        eps2 = max(eps, 10 ** (-14+i))
        rhs2 = discretization.rhs(state + eps * pert)
        assert numpy.linalg.norm((rhs2 - rhs) / eps - A @ pert) < eps2

def test_jac_consistency_tc_uniform_2d():
    parameters, nx, ny, nz, dim, dof, x, y, z = create_test_problem()

    nz = 1
    dim = 2
    dof = 3
    parameters['Problem Type'] = 'Taylor-Couette'
    parameters['Reynolds Number'] = 0
    n = dof * nx * ny * nz

    state = numpy.random.random(n)
    pert = numpy.random.random(n)

    discretization = CylindricalDiscretization(parameters, nx, ny, nz, dim, dof)
    A = discretization.jacobian(state)
    rhs = discretization.rhs(state)

    for i in range(3, 12):
        eps = 10 ** -i
        eps2 = max(eps, 10 ** (-12+i))
        rhs2 = discretization.rhs(state + eps * pert)
        assert numpy.linalg.norm((rhs2 - rhs) / eps - A @ pert) < eps2

def test_jac_consistency_tc_stretched_2d():
    parameters, nx, ny, nz, dim, dof, x, y, z = create_test_problem()

    x = utils.create_stretched_coordinate_vector(0, 1, nx, 1.5)

    nz = 1
    dim = 2
    dof = 3
    parameters['Problem Type'] = 'Taylor-Couette'
    parameters['Reynolds Number'] = 0
    n = dof * nx * ny * nz

    state = numpy.random.random(n)
    pert = numpy.random.random(n)

    discretization = CylindricalDiscretization(parameters, nx, ny, nz, dim, dof, x)
    A = discretization.jacobian(state)
    rhs = discretization.rhs(state)

    for i in range(3, 12):
        eps = 10 ** -i
        eps2 = max(eps, 10 ** (-12+i))
        rhs2 = discretization.rhs(state + eps * pert)
        assert numpy.linalg.norm((rhs2 - rhs) / eps - A @ pert) < eps2
