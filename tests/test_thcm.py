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

    # TODO:
    n = B.n

    for i in range(n):
        print(i)

        scaling = mass_matrix[i, i]

        print('Expected:')
        print(B.jcoA[B.begA[i]:B.begA[i+1]])
        print(B.coA[B.begA[i]:B.begA[i+1]])

        print('Got:')
        print(A.jcoA[A.begA[i]:A.begA[i+1]])
        print(A.coA[A.begA[i]:A.begA[i+1]])

        print('Scaled:')
        print(A.coA[A.begA[i]:A.begA[i+1]] / scaling)

        assert B.begA[i+1] - B.begA[i] == A.begA[i+1] - A.begA[i]
        for j in range(B.begA[i], B.begA[i+1]):
            assert B.jcoA[j] == A.jcoA[j]
            assert B.coA[j] == pytest.approx(-A.coA[j] / scaling)
