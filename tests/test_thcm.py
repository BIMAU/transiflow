import numpy
import pytest

from tests import testutils

from transiflow import OceanDiscretization

def get_scaling(discretization, n):
    dof = 6

    x = discretization.x
    y = discretization.y
    z = discretization.z

    nx = discretization.nx
    ny = discretization.ny
    nz = discretization.nz
    dof = discretization.dof

    i = (n // dof) % nx
    j = (n // dof // nx) % ny
    k = (n // dof // nx // ny) % nz

    if n % dof == 0:
        return discretization._mass_x(i, j, k, x, y, z)
    elif n % dof == 1:
        return discretization._mass_x(j, i, k, y, x, z)
    elif n % dof == 2:
        return discretization._mass_x(k, j, i, z, y, x)
    elif n % dof == 3:
        return -discretization._mass_C(i, j, k, x, y, z)
    else:
        return discretization._mass_C(i, j, k, x, y, z)

def test_thcm_lin():
    nx = 4
    ny = nx
    nz = nx
    dim = 3
    dof = 6
    parameters = {'Problem Type': 'Ocean'}
    n = nx * ny * nz * dof

    discretization = OceanDiscretization(parameters, nx, ny, nz, dim, dof)
    atom = discretization.linear_part()
    A = testutils.assemble_jacobian(atom, nx, ny, nz, dof)

    B = testutils.read_matrix('data/thcm_lin_%sx%sx%s.txt' % (nx, ny, nz))

    for i in range(n):
        print(i)

        scaling = get_scaling(discretization, i)

        print('Expected:')
        print(B.jcoA[B.begA[i]:B.begA[i+1]])
        print(B.coA[B.begA[i]:B.begA[i+1]])

        print('Got:')
        print(A.jcoA[A.begA[i]:A.begA[i+1]])
        print(A.coA[A.begA[i]:A.begA[i+1]])

        print('Scaled:')
        print(-A.coA[A.begA[i]:A.begA[i+1]] / scaling)

        assert B.begA[i+1] - B.begA[i] == A.begA[i+1] - A.begA[i]
        for j in range(B.begA[i], B.begA[i+1]):
            assert B.jcoA[j] == A.jcoA[j]
            assert B.coA[j] == pytest.approx(-A.coA[j] / scaling)

def test_thcm_bnd():
    nx = 4
    ny = nx
    nz = nx
    dim = 3
    dof = 6
    parameters = {'Problem Type': 'Ocean'}
    n = nx * ny * nz * dof

    discretization = OceanDiscretization(parameters, nx, ny, nz, dim, dof)
    atom = discretization.linear_part()
    discretization.boundaries(atom)
    A = testutils.assemble_jacobian(atom, nx, ny, nz, dof)

    B = testutils.read_matrix('data/thcm_bnd_%sx%sx%s.txt' % (nx, ny, nz))

    for i in range(n):
        print(i)

        scaling = get_scaling(discretization, i)

        print('Expected:')
        print(B.jcoA[B.begA[i]:B.begA[i+1]])
        print(B.coA[B.begA[i]:B.begA[i+1]])

        print('Got:')
        print(A.jcoA[A.begA[i]:A.begA[i+1]])
        print(A.coA[A.begA[i]:A.begA[i+1]])

        print('Scaled:')
        print(-A.coA[A.begA[i]:A.begA[i+1]] / scaling)

        # Skip dirichlet conditions:
        if A.begA[i+1] - A.begA[i] == 1 and A.coA[A.begA[i]] == -1:
            continue

        assert B.begA[i+1] - B.begA[i] == A.begA[i+1] - A.begA[i]
        for j in range(B.begA[i+1] - B.begA[i]):
            jA = A.begA[i] + j
            jB = B.begA[i] + j
            assert B.jcoA[jB] == A.jcoA[jA]
            assert B.coA[jB] == pytest.approx(-A.coA[jA] / scaling)

def test_thcm_bil():
    nx = 4
    ny = nx
    nz = nx
    dim = 3
    dof = 6
    parameters = {'Problem Type': 'Ocean'}
    n = nx * ny * nz * dof

    state = numpy.zeros(n)
    for i in range(n):
        state[i] = i+1

    discretization = OceanDiscretization(parameters, nx, ny, nz, dim, dof)
    atom, atomF = discretization.nonlinear_part(state)
    discretization.boundaries(atom)
    A = testutils.assemble_jacobian(atom, nx, ny, nz, dof)

    B = testutils.read_matrix('data/thcm_bil_%sx%sx%s.txt' % (nx, ny, nz))

    for i in range(n):
        print(i)

        scaling = get_scaling(discretization, i)

        print('Expected:')
        print(B.jcoA[B.begA[i]:B.begA[i+1]])
        print(B.coA[B.begA[i]:B.begA[i+1]])

        print('Got:')
        print(A.jcoA[A.begA[i]:A.begA[i+1]])
        print(A.coA[A.begA[i]:A.begA[i+1]])

        print('Scaled:')
        print(-A.coA[A.begA[i]:A.begA[i+1]] / scaling)

        # Skip dirichlet conditions:
        if A.begA[i+1] - A.begA[i] == 1 and A.coA[A.begA[i]] == -1:
            continue

        assert B.begA[i+1] - B.begA[i] == A.begA[i+1] - A.begA[i]
        for j in range(B.begA[i+1] - B.begA[i]):
            jA = A.begA[i] + j
            jB = B.begA[i] + j
            assert B.jcoA[jB] == A.jcoA[jA]
            assert B.coA[jB] == pytest.approx(-A.coA[jA] / scaling)

def test_thcm():
    nx = 4
    ny = nx
    nz = nx
    dim = 3
    dof = 6
    parameters = {'Problem Type': 'Ocean'}
    n = nx * ny * nz * dof

    state = numpy.zeros(n)
    for i in range(n):
        state[i] = i+1

    discretization = OceanDiscretization(parameters, nx, ny, nz, dim, dof)
    A = discretization.jacobian(state)
    rhs = discretization.rhs(state)

    B = testutils.read_matrix('data/thcm_%sx%sx%s.txt' % (nx, ny, nz))
    rhs_B = testutils.read_vector('data/thcm_rhs_%sx%sx%s.txt' % (nx, ny, nz))

    for i in range(n):
        print(i)

        scaling = get_scaling(discretization, i)

        print('Expected:')
        print(B.jcoA[B.begA[i]:B.begA[i+1]])
        print(B.coA[B.begA[i]:B.begA[i+1]])

        print('Got:')
        print(A.jcoA[A.begA[i]:A.begA[i+1]])
        print(A.coA[A.begA[i]:A.begA[i+1]])

        # Integral condition
        if B.begA[i+1] - B.begA[i] == n // dof:
            dx = discretization.x[1] - discretization.x[0]
            dy = discretization.y[1] - discretization.y[0]
            dz = 1 / 4

            scaling = -dx * dy * dz

            rhs_B[i] = 0

        print('Scaled:')
        print(-A.coA[A.begA[i]:A.begA[i+1]] / scaling)

        # Skip dirichlet conditions:
        if A.begA[i+1] - A.begA[i] == 1 and A.coA[A.begA[i]] == -1:
            continue

        assert B.begA[i+1] - B.begA[i] == A.begA[i+1] - A.begA[i]
        for j in range(B.begA[i+1] - B.begA[i]):
            jA = A.begA[i] + j
            jB = B.begA[i] + j
            assert B.jcoA[jB] == A.jcoA[jA]
            assert B.coA[jB] == pytest.approx(-A.coA[jA] / scaling)

        assert rhs_B[i] == pytest.approx(rhs[i] / scaling)
