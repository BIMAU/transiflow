import numpy
import pytest

from tests import testutils

from transiflow import OceanDiscretization

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

    mass_matrix = discretization.mass_matrix()

    for i in range(n):
        print(i)

        scaling = mass_matrix[i, i] or -mass_matrix[i+1, i+1]

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

    mass_matrix = discretization.mass_matrix()

    for i in range(n):
        print(i)

        scaling = mass_matrix[i, i] or -mass_matrix[i+1, i+1]

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

    mass_matrix = discretization.mass_matrix()

    for i in range(n):
        print(i)

        scaling = mass_matrix[i, i] or -mass_matrix[i+1, i+1]

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

    mass_matrix = discretization.mass_matrix()

    for i in range(n):
        print(i)

        scaling = mass_matrix[i, i] or -mass_matrix[i+1, i+1]

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

        assert rhs_B[i] == pytest.approx(rhs[i] / scaling)
