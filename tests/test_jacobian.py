import numpy

from transiflow import utils
from transiflow import Discretization, CylindricalDiscretization

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

def create_random_state(nx, ny, nz, dim, dof):
    n = nx * ny * nz * dof
    x = numpy.random.random(n)
    x = utils.create_state_mtx(x, nx, ny, nz, dof)
    x[nx-1, :, :, 0] = 0
    x[:, ny-1, :, 1] = 0
    if dim > 2:
        x[:, :, nz-1, 2] = 0
    return utils.create_state_vec(x, nx, ny, nz, dof)

def create_test_problem_tc():
    nx = 13
    ny = 7
    nz = 5
    dim = 3
    dof = 4

    x = create_coordinate_vector(nx)
    y = create_coordinate_vector(ny)
    z = create_coordinate_vector(nz)

    parameters = {'Reynolds Number': 100, 'Problem Type': 'Taylor-Couette'}

    return (parameters, nx, ny, nz, dim, dof, x, y, z)

def create_random_state_tc(nx, ny, nz, dim, dof):
    n = nx * ny * nz * dof
    x = numpy.random.random(n)
    x = utils.create_state_mtx(x, nx, ny, nz, dof)
    x[nx-1, :, :, 0] = 0
    x[:, :, nz-1, 2] = 0
    return utils.create_state_vec(x, nx, ny, nz, dof)

def check_divfree(discretization, state):
    A = discretization.jacobian(state)
    x = A @ state
    for i in range(len(state)):
        if i % discretization.dof == discretization.dim:
            assert abs(x[i]) < 1e-13

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
    nx = discretization.nx
    ny = discretization.ny
    nz = discretization.nz
    dim = discretization.dim
    dof = discretization.dof

    state = create_random_state(nx, ny, nz, dim, dof)
    return make_divfree(discretization, state)

def test_bilin():
    parameters, nx, ny, nz, dim, dof, x, y, z = create_test_problem()

    discretization = Discretization(parameters, nx, ny, nz, dim, dof, x, y, z)
    state = create_divfree_state(discretization)

    atomJ, atomF = discretization.nonlinear_part(state)
    discretization.boundaries(atomF)
    A = discretization.assemble_jacobian(atomF)

    for i in range(A.n):
        for j in range(A.begA[i], A.begA[i+1]):
            assert i != A.jcoA[j] or A.coA[j] == -1

def test_bilin_uniform():
    parameters, nx, ny, nz, dim, dof, x, y, z = create_test_problem()

    discretization = Discretization(parameters, nx, ny, nz, dim, dof)
    state = create_divfree_state(discretization)

    atomJ, atomF = discretization.nonlinear_part(state)
    discretization.boundaries(atomF)
    A = discretization.assemble_jacobian(atomF)

    for i in range(A.n):
        for j in range(A.begA[i], A.begA[i+1]):
            assert i != A.jcoA[j] or A.coA[j] == -1

def test_bilin_stretched():
    parameters, nx, ny, nz, dim, dof, x, y, z = create_test_problem()

    x = utils.create_stretched_coordinate_vector(0, 1, nx, 1.5)
    y = utils.create_stretched_coordinate_vector(0, 1, ny, 1.5)
    z = utils.create_stretched_coordinate_vector(0, 1, nz, 1.5)

    discretization = Discretization(parameters, nx, ny, nz, dim, dof, x, y, z)
    state = create_divfree_state(discretization)

    atomJ, atomF = discretization.nonlinear_part(state)
    discretization.boundaries(atomF)
    A = discretization.assemble_jacobian(atomF)

    for i in range(A.n):
        for j in range(A.begA[i], A.begA[i+1]):
            assert i != A.jcoA[j] or A.coA[j] == -1

def test_jac_consistency():
    parameters, nx, ny, nz, dim, dof, x, y, z = create_test_problem()

    state = create_random_state(nx, ny, nz, dim, dof)
    pert = create_random_state(nx, ny, nz, dim, dof)

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

    state = create_random_state(nx, ny, nz, dim, dof)
    pert = create_random_state(nx, ny, nz, dim, dof)

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

    state = create_random_state(nx, ny, nz, dim, dof)
    pert = create_random_state(nx, ny, nz, dim, dof)

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
    state = create_random_state(nx, ny, nz, dim, dof)
    pert = create_random_state(nx, ny, nz, dim, dof)

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
    state = create_random_state(nx, ny, nz, dim, dof)
    pert = create_random_state(nx, ny, nz, dim, dof)

    discretization = Discretization(parameters, nx, ny, nz, dim, dof, x, y)
    A = discretization.jacobian(state)
    rhs = discretization.rhs(state)

    for i in range(3, 12):
        eps = 10 ** -i
        eps2 = max(eps, 10 ** (-14+i))
        rhs2 = discretization.rhs(state + eps * pert)
        assert numpy.linalg.norm((rhs2 - rhs) / eps - A @ pert) < eps2

def test_jac_consistency_tc_uniform():
    parameters, nx, ny, nz, dim, dof, x, y, z = create_test_problem_tc()

    state = create_random_state_tc(nx, ny, nz, dim, dof)
    pert = create_random_state_tc(nx, ny, nz, dim, dof)

    discretization = CylindricalDiscretization(parameters, nx, ny, nz, dim, dof)
    A = discretization.jacobian(state)
    rhs = discretization.rhs(state)

    for i in range(3, 12):
        eps = 10 ** -i
        eps2 = 3 * max(eps, 10 ** (-14+i))
        rhs2 = discretization.rhs(state + eps * pert)
        assert numpy.linalg.norm((rhs2 - rhs) / eps - A @ pert) < eps2

def test_jac_consistency_tc_stretched():
    parameters, nx, ny, nz, dim, dof, x, y, z = create_test_problem_tc()

    x = utils.create_stretched_coordinate_vector(0, 1, nx, 1.5)
    z = utils.create_stretched_coordinate_vector(0, 1, nz, 1.5)

    state = create_random_state_tc(nx, ny, nz, dim, dof)
    pert = create_random_state_tc(nx, ny, nz, dim, dof)

    discretization = CylindricalDiscretization(parameters, nx, ny, nz, dim, dof, x, None, z)
    A = discretization.jacobian(state)
    rhs = discretization.rhs(state)

    for i in range(3, 12):
        eps = 10 ** -i
        eps2 = 3 * max(eps, 10 ** (-14+i))
        rhs2 = discretization.rhs(state + eps * pert)
        assert numpy.linalg.norm((rhs2 - rhs) / eps - A @ pert) < eps2
