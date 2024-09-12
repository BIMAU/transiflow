import os
import numpy
import pytest

from transiflow import utils
from transiflow import CrsMatrix
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

def test_uniform_grid():
    nx = 5
    dx = 1 / nx
    x = utils.create_uniform_coordinate_vector(0, 1, nx)
    for i in range(nx):
        assert x[i] == pytest.approx((i + 1) * dx)

    assert x[-1] == pytest.approx(0)
    assert x[0] > 0
    assert x[nx-1] == pytest.approx(1)

def test_shifted_uniform_grid():
    nx = 5
    dx = 3 / nx
    x = utils.create_uniform_coordinate_vector(1, 4, nx)
    for i in range(nx):
        assert x[i] == pytest.approx((i + 1) * dx + 1)

    assert x[-1] == pytest.approx(1)
    assert x[0] > 1
    assert x[nx-1] == pytest.approx(4)

def test_stretched_grid():
    nx = 5
    x = utils.create_stretched_coordinate_vector(0, 1, nx, 1.5)

    assert x[-1] == pytest.approx(0)
    assert x[0] > 0
    assert x[nx-1] == pytest.approx(1)

def test_u_xx():
    parameters, nx, ny, nz, dim, dof, x, y, z = create_test_problem()

    discretization = Discretization(parameters, nx, ny, nz, dim, dof, x, y, z)
    atom = discretization.u_xx()

    for i in range(nx):
        dx = x[i] - x[i-1]
        dxp1 = x[i+1] - x[i]
        for j in range(ny):
            dy = y[j] - y[j-1]
            for k in range(nz):
                dz = z[k] - z[k-1]
                print(i, j, k)
                assert atom[i, j, k, 0, 0, 0, 1, 1] == pytest.approx(1 / dx * dy * dz)
                assert atom[i, j, k, 0, 0, 2, 1, 1] == pytest.approx(1 / dxp1 * dy * dz)

def test_v_yy():
    parameters, nx, ny, nz, dim, dof, x, y, z = create_test_problem()

    discretization = Discretization(parameters, nx, ny, nz, dim, dof, x, y, z)
    atom = discretization.v_yy()

    for i in range(nx):
        dx = x[i] - x[i-1]
        for j in range(ny):
            dy = y[j] - y[j-1]
            dyp1 = y[j+1] - y[j]
            for k in range(nz):
                dz = z[k] - z[k-1]
                print(i, j, k)
                assert atom[i, j, k, 1, 1, 1, 0, 1] == pytest.approx(1 / dy * dx * dz)
                assert atom[i, j, k, 1, 1, 1, 2, 1] == pytest.approx(1 / dyp1 * dx * dz)

def test_w_zz():
    parameters, nx, ny, nz, dim, dof, x, y, z = create_test_problem()

    discretization = Discretization(parameters, nx, ny, nz, dim, dof, x, y, z)
    atom = discretization.w_zz()

    for i in range(nx):
        dx = x[i] - x[i-1]
        for j in range(ny):
            dy = y[j] - y[j-1]
            for k in range(nz):
                dz = z[k] - z[k-1]
                dzp1 = z[k+1] - z[k]
                print(i, j, k)
                assert atom[i, j, k, 2, 2, 1, 1, 0] == pytest.approx(1 / dz * dy * dx)
                assert atom[i, j, k, 2, 2, 1, 1, 2] == pytest.approx(1 / dzp1 * dy * dx)

def test_u_yy():
    parameters, nx, ny, nz, dim, dof, x, y, z = create_test_problem()

    discretization = Discretization(parameters, nx, ny, nz, dim, dof, x, y, z)
    atom = discretization.u_yy()

    for i in range(nx):
        dx = (x[i+1] - x[i-1]) / 2
        for j in range(ny):
            dy = (y[j] - y[j-2]) / 2
            dyp1 = (y[j+1] - y[j-1]) / 2
            for k in range(nz):
                dz = z[k] - z[k-1]
                print(i, j, k)
                assert atom[i, j, k, 0, 0, 1, 0, 1] == pytest.approx(1 / dy * dx * dz)
                assert atom[i, j, k, 0, 0, 1, 2, 1] == pytest.approx(1 / dyp1 * dx * dz)

def test_v_xx():
    parameters, nx, ny, nz, dim, dof, x, y, z = create_test_problem()

    discretization = Discretization(parameters, nx, ny, nz, dim, dof, x, y, z)
    atom = discretization.v_xx()

    for i in range(nx):
        dx = (x[i] - x[i-2]) / 2
        dxp1 = (x[i+1] - x[i-1]) / 2
        for j in range(ny):
            dy = (y[j+1] - y[j-1]) / 2
            for k in range(nz):
                dz = z[k] - z[k-1]
                print(i, j, k)
                assert atom[i, j, k, 1, 1, 0, 1, 1] == pytest.approx(1 / dx * dy * dz)
                assert atom[i, j, k, 1, 1, 2, 1, 1] == pytest.approx(1 / dxp1 * dy * dz)

def test_w_yy():
    parameters, nx, ny, nz, dim, dof, x, y, z = create_test_problem()

    discretization = Discretization(parameters, nx, ny, nz, dim, dof, x, y, z)
    atom = discretization.w_yy()

    for i in range(nx):
        dx = x[i] - x[i-1]
        for j in range(ny):
            dy = (y[j] - y[j-2]) / 2
            dyp1 = (y[j+1] - y[j-1]) / 2
            for k in range(nz):
                dz = (z[k+1] - z[k-1]) / 2
                print(i, j, k)
                assert atom[i, j, k, 2, 2, 1, 0, 1] == pytest.approx(1 / dy * dz * dx)
                assert atom[i, j, k, 2, 2, 1, 2, 1] == pytest.approx(1 / dyp1 * dz * dx)

def test_u_zz():
    parameters, nx, ny, nz, dim, dof, x, y, z = create_test_problem()

    discretization = Discretization(parameters, nx, ny, nz, dim, dof, x, y, z)
    atom = discretization.u_zz()

    for i in range(nx):
        dx = (x[i+1] - x[i-1]) / 2
        for j in range(ny):
            dy = y[j] - y[j-1]
            for k in range(nz):
                dz = (z[k] - z[k-2]) / 2
                dzp1 = (z[k+1] - z[k-1]) / 2
                print(i, j, k)
                assert atom[i, j, k, 0, 0, 1, 1, 0] == pytest.approx(1 / dz * dx * dy)
                assert atom[i, j, k, 0, 0, 1, 1, 2] == pytest.approx(1 / dzp1 * dx * dy)

def test_v_zz():
    parameters, nx, ny, nz, dim, dof, x, y, z = create_test_problem()

    discretization = Discretization(parameters, nx, ny, nz, dim, dof, x, y, z)
    atom = discretization.v_zz()

    for i in range(nx):
        dx = x[i] - x[i-1]
        for j in range(ny):
            dy = (y[j+1] - y[j-1]) / 2
            for k in range(nz):
                dz = (z[k] - z[k-2]) / 2
                dzp1 = (z[k+1] - z[k-1]) / 2
                print(i, j, k)
                assert atom[i, j, k, 1, 1, 1, 1, 0] == pytest.approx(1 / dz * dy * dx)
                assert atom[i, j, k, 1, 1, 1, 1, 2] == pytest.approx(1 / dzp1 * dy * dx)

def test_w_xx():
    parameters, nx, ny, nz, dim, dof, x, y, z = create_test_problem()

    discretization = Discretization(parameters, nx, ny, nz, dim, dof, x, y, z)
    atom = discretization.w_xx()

    for i in range(nx):
        dx = (x[i] - x[i-2]) / 2
        dxp1 = (x[i+1] - x[i-1]) / 2
        for j in range(ny):
            dy = y[j] - y[j-1]
            for k in range(nz):
                dz = (z[k+1] - z[k-1]) / 2
                print(i, j, k)
                assert atom[i, j, k, 2, 2, 0, 1, 1] == pytest.approx(1 / dx * dz * dy)
                assert atom[i, j, k, 2, 2, 2, 1, 1] == pytest.approx(1 / dxp1 * dz * dy)

def test_T_xx():
    parameters, nx, ny, nz, dim, dof, x, y, z = create_test_problem()

    discretization = Discretization(parameters, nx, ny, nz, dim, dof, x, y, z)
    atom = discretization.T_xx()

    for i in range(nx):
        dx = (x[i] - x[i-2]) / 2
        dxp1 = (x[i+1] - x[i-1]) / 2
        for j in range(ny):
            dy = y[j] - y[j-1]
            for k in range(nz):
                dz = z[k] - z[k-1]
                print(i, j, k)
                assert atom[i, j, k, 4, 4, 0, 1, 1] == pytest.approx(1 / dx * dy * dz)
                assert atom[i, j, k, 4, 4, 2, 1, 1] == pytest.approx(1 / dxp1 * dy * dz)

def test_T_yy():
    parameters, nx, ny, nz, dim, dof, x, y, z = create_test_problem()

    discretization = Discretization(parameters, nx, ny, nz, dim, dof, x, y, z)
    atom = discretization.T_yy()

    for i in range(nx):
        dx = x[i] - x[i-1]
        for j in range(ny):
            dy = (y[j] - y[j-2]) / 2
            dyp1 = (y[j+1] - y[j-1]) / 2
            for k in range(nz):
                dz = z[k] - z[k-1]
                print(i, j, k)
                assert atom[i, j, k, 4, 4, 1, 0, 1] == pytest.approx(1 / dy * dx * dz)
                assert atom[i, j, k, 4, 4, 1, 2, 1] == pytest.approx(1 / dyp1 * dx * dz)

def test_T_zz():
    parameters, nx, ny, nz, dim, dof, x, y, z = create_test_problem()

    discretization = Discretization(parameters, nx, ny, nz, dim, dof, x, y, z)
    atom = discretization.T_zz()

    for i in range(nx):
        dx = x[i] - x[i-1]
        for j in range(ny):
            dy = y[j] - y[j-1]
            for k in range(nz):
                dz = (z[k] - z[k-2]) / 2
                dzp1 = (z[k+1] - z[k-1]) / 2
                print(i, j, k)
                assert atom[i, j, k, 4, 4, 1, 1, 0] == pytest.approx(1 / dz * dy * dx)
                assert atom[i, j, k, 4, 4, 1, 1, 2] == pytest.approx(1 / dzp1 * dy * dx)

def test_p_x():
    parameters, nx, ny, nz, dim, dof, x, y, z = create_test_problem()

    discretization = Discretization(parameters, nx, ny, nz, dim, dof, x, y, z)
    atom = discretization.p_x()

    for i in range(nx):
        for j in range(ny):
            dy = y[j] - y[j-1]
            for k in range(nz):
                dz = z[k] - z[k-1]
                print(i, j, k)
                assert atom[i, j, k, 0, 3, 1, 1, 1] == pytest.approx(-dy * dz)
                assert atom[i, j, k, 0, 3, 2, 1, 1] == pytest.approx(dy * dz)

def test_p_y():
    parameters, nx, ny, nz, dim, dof, x, y, z = create_test_problem()

    discretization = Discretization(parameters, nx, ny, nz, dim, dof, x, y, z)
    atom = discretization.p_y()

    for i in range(nx):
        dx = x[i] - x[i-1]
        for j in range(ny):
            for k in range(nz):
                dz = z[k] - z[k-1]
                print(i, j, k)
                assert atom[i, j, k, 1, 3, 1, 1, 1] == pytest.approx(-dx * dz)
                assert atom[i, j, k, 1, 3, 1, 2, 1] == pytest.approx(dx * dz)

def test_p_z():
    parameters, nx, ny, nz, dim, dof, x, y, z = create_test_problem()

    discretization = Discretization(parameters, nx, ny, nz, dim, dof, x, y, z)
    atom = discretization.p_z()

    for i in range(nx):
        dx = x[i] - x[i-1]
        for j in range(ny):
            dy = y[j] - y[j-1]
            for k in range(nz):
                print(i, j, k)
                assert atom[i, j, k, 2, 3, 1, 1, 1] == pytest.approx(-dy * dx)
                assert atom[i, j, k, 2, 3, 1, 1, 2] == pytest.approx(dy * dx)

def test_u_x():
    parameters, nx, ny, nz, dim, dof, x, y, z = create_test_problem()

    discretization = Discretization(parameters, nx, ny, nz, dim, dof, x, y, z)
    atom = discretization.u_x()

    for i in range(nx):
        for j in range(ny):
            dy = y[j] - y[j-1]
            for k in range(nz):
                dz = z[k] - z[k-1]
                print(i, j, k)
                assert atom[i, j, k, 3, 0, 0, 1, 1] == pytest.approx(-dy * dz)
                assert atom[i, j, k, 3, 0, 1, 1, 1] == pytest.approx(dy * dz)

def test_u_y():
    parameters, nx, ny, nz, dim, dof, x, y, z = create_test_problem()

    discretization = Discretization(parameters, nx, ny, nz, dim, dof, x, y, z)
    atom = discretization.v_y()

    for i in range(nx):
        dx = x[i] - x[i-1]
        for j in range(ny):
            for k in range(nz):
                dz = z[k] - z[k-1]
                print(i, j, k)
                assert atom[i, j, k, 3, 1, 1, 0, 1] == pytest.approx(-dx * dz)
                assert atom[i, j, k, 3, 1, 1, 1, 1] == pytest.approx(dx * dz)

def test_u_z():
    parameters, nx, ny, nz, dim, dof, x, y, z = create_test_problem()

    discretization = Discretization(parameters, nx, ny, nz, dim, dof, x, y, z)
    atom = discretization.w_z()

    for i in range(nx):
        dx = x[i] - x[i-1]
        for j in range(ny):
            dy = y[j] - y[j-1]
            for k in range(nz):
                print(i, j, k)
                assert atom[i, j, k, 3, 2, 1, 1, 0] == pytest.approx(-dy * dx)
                assert atom[i, j, k, 3, 2, 1, 1, 1] == pytest.approx(dy * dx)

def test_v_v():
    parameters, nx, ny, nz, dim, dof, x, y, z = create_test_problem()
    n = nx * ny * nz * dof

    state = numpy.zeros(n)
    for i in range(n):
        state[i] = i+1

    state_mtx = utils.create_padded_state_mtx(state, nx, ny, nz, dof, False, True, False)

    parameters['Problem Type'] = 'Taylor-Couette'
    discretization = CylindricalDiscretization(parameters, nx, ny, nz, dim, dof, x, y, z)

    atomJ = numpy.zeros([nx, ny, nz, dof, dof, 3, 3, 3])
    atomF = numpy.zeros([nx, ny, nz, dof, dof, 3, 3, 3])
    discretization.v_v(atomJ, atomF, state_mtx)

    averages_v = discretization.weighted_average_x(state_mtx[:, :, :, 1])
    averages_v = (averages_v[:, 0:ny, :] + averages_v[:, 1:ny+1, :]) / 2

    rhs = discretization.assemble_rhs(state, atomF)

    atom_value = numpy.zeros(1)
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                print(i, j, k)
                Discretization._mass_x(atom_value, i, j, k, x, y, z)
                assert rhs[i * dof + j * nx * dof + k * nx * ny * dof] * x[i] == pytest.approx(
                    atom_value * averages_v[i, j, k+1] ** 2)

def read_matrix(fname):
    A = CrsMatrix([], [], [0])

    dirname = os.path.dirname(__file__)
    with open(os.path.join(dirname, fname), 'r') as f:
        rows = []
        idx = 0
        for i in f.readlines():
            r, c, v = [j.strip() for j in i.strip().split(' ') if j]
            r = int(r) - 1
            c = int(c) - 1
            v = float(v)
            rows.append(r)
            while r >= len(A.begA):
                A.begA.append(idx)
            A.jcoA.append(c)
            A.coA.append(v)
            idx += 1
        A.begA.append(idx)
        assert rows == sorted(rows)

    return A

def write_matrix(A, fname):
    dirname = os.path.dirname(__file__)
    with open(os.path.join(dirname, fname), 'w') as f:
        for i in range(len(A.begA)-1):
            for j in range(A.begA[i], A.begA[i+1]):
                f.write('%12d%12d %.16e\n' % (i+1, A.jcoA[j]+1, A.coA[j]))

def read_vector(fname):
    vec = numpy.array([])

    dirname = os.path.dirname(__file__)
    with open(os.path.join(dirname, fname), 'r') as f:
        for i in f.readlines():
            vec = numpy.append(vec, float(i.strip()))
    return vec

def write_vector(vec, fname):
    dirname = os.path.dirname(__file__)
    with open(os.path.join(dirname, fname), 'w') as f:
        for i in range(len(vec)):
            f.write('%.16e\n' % vec[i])

def assemble_jacobian(atom, nx, ny, nz, dof):
    row = 0
    idx = 0
    n = nx * ny * nz * dof
    coA = numpy.zeros(27*n)
    jcoA = numpy.zeros(27*n, dtype=int)
    begA = numpy.zeros(n+1, dtype=int)

    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                for d1 in range(dof):
                    for z in range(3):
                        for y in range(3):
                            for x in range(3):
                                for d2 in range(dof):
                                    if abs(atom[i, j, k, d1, d2, x, y, z]) > 1e-14:
                                        jcoA[idx] = row + (x-1) * dof + (y-1) * nx * dof + (z-1) * nx * ny * dof + d2 - d1
                                        coA[idx] = atom[i, j, k, d1, d2, x, y, z]
                                        idx += 1
                    row += 1
                    begA[row] = idx

    return CrsMatrix(coA, jcoA, begA)

def rotate_atom(atom, nx, ny, nz, dof):
    atom2 = numpy.zeros([nx, ny, nz, dof, dof, 3, 3, 3])
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                for d1 in range(dof):
                    for z in range(3):
                        for y in range(3):
                            for x in range(3):
                                for d2 in range(dof):
                                    d3 = [2, 1, 0, 3][d1]
                                    d4 = [2, 1, 0, 3][d2]
                                    atom2[k, j, i, d3, d4, z, y, x] = atom[i, j, k, d1, d2, x, y, z]

    return atom2

def rotate_state(state, nx, ny, nz, dof):
    out = utils.create_state_mtx(state, nx, ny, nz, dof)
    state_mtx = utils.create_state_mtx(state, nx, ny, nz, dof)
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                for d in range(dof):
                    d2 = [2, 1, 0, 3][d]
                    out[k, j, i, d2] = state_mtx[i, j, k, d]

    return utils.create_state_vec(out, nx, ny, nz, dof)

def test_rotate_lin():
    nx = 4
    ny = nx
    nz = nx
    dim = 3
    dof = 4
    parameters = {'Reynolds Number': 100, 'Grid Stretching': True}

    parameters['X-max'] = 6
    parameters['Z-max'] = 1
    discretization = Discretization(parameters, nx, ny, nz, dim, dof)
    atom1 = discretization.linear_part()
    discretization.boundaries(atom1)

    parameters['X-max'] = 1
    parameters['Z-max'] = 6
    discretization = Discretization(parameters, nx, ny, nz, dim, dof)
    atom2 = discretization.linear_part()
    discretization.boundaries(atom2)

    atom2 = rotate_atom(atom2, nx, ny, nz, dof)

    tol = 1e-13
    assert numpy.all(abs(atom1 - atom2) < tol)

def test_rotate_bil():
    nx = 4
    ny = nx
    nz = nx
    dim = 3
    dof = 4
    parameters = {'Reynolds Number': 100, 'Grid Stretching': True}
    n = nx * ny * nz * dof

    state = numpy.zeros(n)
    for i in range(n):
        state[i] = i+1

    parameters['X-max'] = 6
    parameters['Z-max'] = 1
    discretization = Discretization(parameters, nx, ny, nz, dim, dof)
    atomJ1, atomF1 = discretization.nonlinear_part(state)
    discretization.boundaries(atomJ1)
    frc1 = discretization.boundaries(atomF1)

    state = rotate_state(state, nx, ny, nz, dof)

    parameters['X-max'] = 1
    parameters['Z-max'] = 6
    discretization = Discretization(parameters, nx, ny, nz, dim, dof)
    atomJ2, atomF2 = discretization.nonlinear_part(state)
    discretization.boundaries(atomJ2)
    frc2 = discretization.boundaries(atomF2)

    atomJ2 = rotate_atom(atomJ2, nx, ny, nz, dof)
    atomF2 = rotate_atom(atomF2, nx, ny, nz, dof)
    frc2 = rotate_state(frc2, nx, ny, nz, dof)

    tol = 1e-13
    assert numpy.all(abs(atomJ1 - atomJ2) < tol)
    assert numpy.all(abs(atomF1 - atomF2) < tol)
    assert numpy.all(abs(frc1 - frc2) < tol)

def test_ldc_lin():
    nx = 4
    ny = nx
    nz = nx
    dim = 3
    dof = 4
    parameters = {'Reynolds Number': 100}
    n = nx * ny * nz * dof

    discretization = Discretization(parameters, nx, ny, nz, dim, dof)
    atom = discretization.linear_part()
    A = assemble_jacobian(atom, nx, ny, nz, dof)

    B = read_matrix('data/ldc_lin_%sx%sx%s.txt' % (nx, ny, nz))

    for i in range(n):
        print(i)

        print('Expected:')
        print(B.jcoA[B.begA[i]:B.begA[i+1]])
        print(B.coA[B.begA[i]:B.begA[i+1]])

        print('Got:')
        print(A.jcoA[A.begA[i]:A.begA[i+1]])
        print(A.coA[A.begA[i]:A.begA[i+1]])

        assert B.begA[i+1] - B.begA[i] == A.begA[i+1] - A.begA[i]
        for j in range(B.begA[i], B.begA[i+1]):
            assert B.jcoA[j] == A.jcoA[j]
            assert B.coA[j] == A.coA[j]

def test_bous_lin():
    nx = 4
    ny = nx
    nz = nx
    dim = 3
    dof = 5
    parameters = {'Reynolds Number': 1, 'Rayleigh Number': 100, 'Prandtl Number': 100}
    n = nx * ny * nz * dof

    discretization = Discretization(parameters, nx, ny, nz, dim, dof)
    atom = discretization.linear_part()
    A = assemble_jacobian(atom, nx, ny, nz, dof)

    B = read_matrix('data/bous_lin_%sx%sx%s.txt' % (nx, ny, nz))

    for i in range(n):
        print(i)

        print('Expected:')
        print(B.jcoA[B.begA[i]:B.begA[i+1]])
        print(B.coA[B.begA[i]:B.begA[i+1]])

        print('Got:')
        print(A.jcoA[A.begA[i]:A.begA[i+1]])
        print(A.coA[A.begA[i]:A.begA[i+1]])

        assert B.begA[i+1] - B.begA[i] == A.begA[i+1] - A.begA[i]
        for j in range(B.begA[i], B.begA[i+1]):
            assert B.jcoA[j] == A.jcoA[j]
            assert B.coA[j] == A.coA[j]

def test_assemble_jacobian():
    nx = 4
    ny = nx
    nz = nx
    dim = 3
    dof = 4
    parameters = {'Reynolds Number': 100}
    n = nx * ny * nz * dof

    discretization = Discretization(parameters, nx, ny, nz, dim, dof)
    atom = discretization.linear_part()
    discretization.boundaries(atom)
    A = discretization.assemble_jacobian(atom)

    B = assemble_jacobian(atom, nx, ny, nz, dof)

    for i in range(n):
        print(i)

        print('Expected:')
        print(B.jcoA[B.begA[i]:B.begA[i+1]])
        print(B.coA[B.begA[i]:B.begA[i+1]])

        print('Got:')
        print(A.jcoA[A.begA[i]:A.begA[i+1]])
        print(A.coA[A.begA[i]:A.begA[i+1]])

        assert B.begA[i+1] - B.begA[i] == A.begA[i+1] - A.begA[i]
        for j in range(B.begA[i], B.begA[i+1]):
            assert B.jcoA[j] == A.jcoA[j]
            assert B.coA[j] == A.coA[j]

def test_ldc_bnd():
    nx = 4
    ny = nx
    nz = nx
    dim = 3
    dof = 4
    parameters = {'Reynolds Number': 100}
    n = nx * ny * nz * dof

    discretization = Discretization(parameters, nx, ny, nz, dim, dof)
    atom = discretization.linear_part()
    discretization.boundaries(atom)
    A = discretization.assemble_jacobian(atom)

    B = read_matrix('data/ldc_bnd_%sx%sx%s.txt' % (nx, ny, nz))

    for i in range(n):
        print(i)

        print('Expected:')
        print(B.jcoA[B.begA[i]:B.begA[i+1]])
        print(B.coA[B.begA[i]:B.begA[i+1]])

        print('Got:')
        print(A.jcoA[A.begA[i]:A.begA[i+1]])
        print(A.coA[A.begA[i]:A.begA[i+1]])

        assert B.begA[i+1] - B.begA[i] == A.begA[i+1] - A.begA[i]
        for j in range(B.begA[i], B.begA[i+1]):
            assert B.jcoA[j] == A.jcoA[j]
            assert B.coA[j] == A.coA[j]

def test_bous_bnd():
    nx = 4
    ny = nx
    nz = nx
    dim = 3
    dof = 5
    parameters = {'Reynolds Number': 1, 'Rayleigh Number': 100, 'Prandtl Number': 100, 'Problem Type': 'Rayleigh-Benard'}
    n = nx * ny * nz * dof

    discretization = Discretization(parameters, nx, ny, nz, dim, dof)
    atom = discretization.linear_part()
    discretization.boundaries(atom)
    A = discretization.assemble_jacobian(atom)

    B = read_matrix('data/bous_bnd_%sx%sx%s.txt' % (nx, ny, nz))

    for i in range(n):
        print(i)

        print('Expected:')
        print(B.jcoA[B.begA[i]:B.begA[i+1]])
        print(B.coA[B.begA[i]:B.begA[i+1]])

        print('Got:')
        print(A.jcoA[A.begA[i]:A.begA[i+1]])
        print(A.coA[A.begA[i]:A.begA[i+1]])

        assert B.begA[i+1] - B.begA[i] == A.begA[i+1] - A.begA[i]
        for j in range(B.begA[i], B.begA[i+1]):
            assert B.jcoA[j] == A.jcoA[j]
            assert B.coA[j] == pytest.approx(A.coA[j])

def test_manual_bous_bnd():
    nx = 4
    ny = nx
    nz = nx
    dim = 3
    dof = 5
    parameters = {'Rayleigh Number': 100, 'Prandtl Number': 100}
    n = nx * ny * nz * dof

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

    discretization = Discretization(parameters, nx, ny, nz, dim, dof,
                                    boundary_conditions=boundaries)
    atom = discretization.linear_part()
    discretization.boundaries(atom)
    A = discretization.assemble_jacobian(atom)

    B = read_matrix('data/bous_bnd_%sx%sx%s.txt' % (nx, ny, nz))

    for i in range(n):
        print(i)

        print('Expected:')
        print(B.jcoA[B.begA[i]:B.begA[i+1]])
        print(B.coA[B.begA[i]:B.begA[i+1]])

        print('Got:')
        print(A.jcoA[A.begA[i]:A.begA[i+1]])
        print(A.coA[A.begA[i]:A.begA[i+1]])

        assert B.begA[i+1] - B.begA[i] == A.begA[i+1] - A.begA[i]
        for j in range(B.begA[i], B.begA[i+1]):
            assert B.jcoA[j] == A.jcoA[j]
            assert B.coA[j] == pytest.approx(A.coA[j])

def test_ldc_bil():
    nx = 4
    ny = nx
    nz = nx
    dim = 3
    dof = 4
    parameters = {'Reynolds Number': 100}
    n = nx * ny * nz * dof

    state = numpy.zeros(n)
    for i in range(n):
        state[i] = i+1

    discretization = Discretization(parameters, nx, ny, nz, dim, dof)
    atom, atomF = discretization.nonlinear_part(state)
    discretization.boundaries(atom)
    A = discretization.assemble_jacobian(atom)

    B = read_matrix('data/ldc_bil_%sx%sx%s.txt' % (nx, ny, nz))

    for i in range(n):
        print(i)

        if i+1 >= len(B.begA):
            break

        print('Expected:')
        print(B.jcoA[B.begA[i]:B.begA[i+1]])
        print(B.coA[B.begA[i]:B.begA[i+1]])

        print('Got:')
        print(A.jcoA[A.begA[i]:A.begA[i+1]])
        print(A.coA[A.begA[i]:A.begA[i+1]])

        if A.begA[i+1] - A.begA[i] == 1 and A.coA[A.begA[i]] == -1:
            # Inserted boundary condition
            continue

        assert B.begA[i+1] - B.begA[i] == A.begA[i+1] - A.begA[i]
        for j in range(B.begA[i+1] - B.begA[i]):
            assert B.jcoA[B.begA[i] + j] == A.jcoA[A.begA[i] + j]
            assert B.coA[B.begA[i] + j] == A.coA[A.begA[i] + j]

def test_bous_bil():
    nx = 4
    ny = nx
    nz = nx
    dim = 3
    dof = 5
    parameters = {'Reynolds Number': 1, 'Rayleigh Number': 100, 'Prandtl Number': 100}
    n = nx * ny * nz * dof

    state = numpy.zeros(n)
    for i in range(n):
        state[i] = i+1

    discretization = Discretization(parameters, nx, ny, nz, dim, dof)
    atom, atomF = discretization.nonlinear_part(state)
    discretization.boundaries(atom)
    A = discretization.assemble_jacobian(atom)

    B = read_matrix('data/bous_bil_%sx%sx%s.txt' % (nx, ny, nz))

    for i in range(n):
        print(i)

        if i+1 >= len(B.begA):
            break

        print('Expected:')
        print(B.jcoA[B.begA[i]:B.begA[i+1]])
        print(B.coA[B.begA[i]:B.begA[i+1]])

        print('Got:')
        print(A.jcoA[A.begA[i]:A.begA[i+1]])
        print(A.coA[A.begA[i]:A.begA[i+1]])

        if A.begA[i+1] - A.begA[i] == 1 and A.coA[A.begA[i]] == -1:
            # Inserted boundary condition
            continue

        assert B.begA[i+1] - B.begA[i] == A.begA[i+1] - A.begA[i]
        for j in range(B.begA[i+1] - B.begA[i]):
            assert B.jcoA[B.begA[i] + j] == A.jcoA[A.begA[i] + j]
            assert B.coA[B.begA[i] + j] == A.coA[A.begA[i] + j]

def test_ldc():
    nx = 4
    ny = nx
    nz = nx
    dim = 3
    dof = 4
    parameters = {'Reynolds Number': 100}
    n = nx * ny * nz * dof

    state = numpy.zeros(n)
    for i in range(n):
        state[i] = i+1

    discretization = Discretization(parameters, nx, ny, nz, dim, dof)
    A = discretization.jacobian(state)
    rhs = discretization.rhs(state)

    B = read_matrix('data/ldc_%sx%sx%s.txt' % (nx, ny, nz))
    rhs_B = read_vector('data/ldc_rhs_%sx%sx%s.txt' % (nx, ny, nz))

    for i in range(n):
        print(i)

        print('Expected:')
        print(B.jcoA[B.begA[i]:B.begA[i+1]])
        print(B.coA[B.begA[i]:B.begA[i+1]])

        print('Got:')
        print(A.jcoA[A.begA[i]:A.begA[i+1]])
        print(A.coA[A.begA[i]:A.begA[i+1]])

        assert A.begA[i+1] - A.begA[i] == B.begA[i+1] - B.begA[i]
        for j in range(A.begA[i], A.begA[i+1]):
            assert A.jcoA[j] == B.jcoA[j]
            assert A.coA[j] == pytest.approx(B.coA[j])

        assert rhs_B[i] == pytest.approx(rhs[i])

def test_ldc_stretched():
    nx = 4
    ny = nx
    nz = nx
    dim = 3
    dof = 4
    parameters = {'Reynolds Number': 100, 'Grid Stretching': True}
    n = nx * ny * nz * dof

    state = numpy.zeros(n)
    for i in range(n):
        state[i] = i+1

    discretization = Discretization(parameters, nx, ny, nz, dim, dof)
    A = discretization.jacobian(state)
    rhs = discretization.rhs(state)

    B = read_matrix('data/ldc_stretched_%sx%sx%s.txt' % (nx, ny, nz))
    rhs_B = read_vector('data/ldc_stretched_rhs_%sx%sx%s.txt' % (nx, ny, nz))

    for i in range(n):
        print(i)

        print('Expected:')
        print(B.jcoA[B.begA[i]:B.begA[i+1]])
        print(B.coA[B.begA[i]:B.begA[i+1]])

        print('Got:')
        print(A.jcoA[A.begA[i]:A.begA[i+1]])
        print(A.coA[A.begA[i]:A.begA[i+1]])

        assert A.begA[i+1] - A.begA[i] == B.begA[i+1] - B.begA[i]
        for j in range(A.begA[i], A.begA[i+1]):
            assert A.jcoA[j] == B.jcoA[j]
            assert A.coA[j] == pytest.approx(B.coA[j])

        assert rhs_B[i] == pytest.approx(rhs[i])

def test_ldc_2D():
    nx = 4
    ny = nx
    nz = 1
    dim = 3
    dof1 = 4
    parameters = {'Reynolds Number': 100}
    n = nx * ny * nz * dof1

    state1 = numpy.zeros(n)
    for i in range(n):
        state1[i] = i+1

    discretization = Discretization(parameters, nx, ny, nz, dim, dof1)
    A1 = discretization.jacobian(state1)
    rhs1 = discretization.rhs(state1)

    dim = 2
    dof2 = 3
    n = nx * ny * nz * dof2

    state2 = numpy.zeros(n)
    for i in range(n):
        state2[i] = state1[i + (i + 1) // dof2]

    discretization = Discretization(parameters, nx, ny, nz, dim, dof2)
    A2 = discretization.jacobian(state2)
    rhs2 = discretization.rhs(state2)

    for i in range(n):
        print(i)

        i1 = i + (i + 1) // dof2

        print('Expected:')
        print(A1.jcoA[A1.begA[i1]:A1.begA[i1+1]])
        print(A1.coA[A1.begA[i1]:A1.begA[i1+1]])

        print('Got:')
        print(A2.jcoA[A2.begA[i]:A2.begA[i+1]])
        print(A2.coA[A2.begA[i]:A2.begA[i+1]])

        assert A1.begA[i1+1] - A1.begA[i1] == A2.begA[i+1] - A2.begA[i]
        for j in range(A2.begA[i+1] - A2.begA[i]):
            j1 = A1.begA[i1] + j
            j2 = A2.begA[i] + j
            assert A1.jcoA[j1] == A2.jcoA[j2] + (A2.jcoA[j2] + 1) // dof2
            assert A1.coA[j1] == pytest.approx(A2.coA[j2])

        assert rhs2[i] == pytest.approx(rhs1[i1])

def test_bous():
    nx = 4
    ny = nx
    nz = nx
    dim = 3
    dof = 5
    parameters = {'Reynolds Number': 1, 'Rayleigh Number': 100, 'Prandtl Number': 100,
                  'Problem Type': 'Rayleigh-Benard'}
    n = nx * ny * nz * dof

    state = numpy.zeros(n)
    for i in range(n):
        state[i] = i+1

    discretization = Discretization(parameters, nx, ny, nz, dim, dof)
    A = discretization.jacobian(state)
    rhs = discretization.rhs(state)

    B = read_matrix('data/bous_%sx%sx%s.txt' % (nx, ny, nz))
    rhs_B = read_vector('data/bous_rhs_%sx%sx%s.txt' % (nx, ny, nz))

    for i in range(n):
        print(i)

        print('Expected:')
        print(B.jcoA[B.begA[i]:B.begA[i+1]])
        print(B.coA[B.begA[i]:B.begA[i+1]])

        print('Got:')
        print(A.jcoA[A.begA[i]:A.begA[i+1]])
        print(A.coA[A.begA[i]:A.begA[i+1]])

        assert A.begA[i+1] - A.begA[i] == B.begA[i+1] - B.begA[i]
        for j in range(A.begA[i], A.begA[i+1]):
            assert A.jcoA[j] == B.jcoA[j]
            assert A.coA[j] == pytest.approx(B.coA[j])

        assert rhs_B[i] == pytest.approx(rhs[i])

def test_bous_stretched():
    nx = 4
    ny = nx
    nz = nx
    dim = 3
    dof = 5
    parameters = {'Reynolds Number': 1, 'Rayleigh Number': 100, 'Prandtl Number': 100,
                  'Problem Type': 'Rayleigh-Benard', 'Grid Stretching': True}
    n = nx * ny * nz * dof

    state = numpy.zeros(n)
    for i in range(n):
        state[i] = i+1

    discretization = Discretization(parameters, nx, ny, nz, dim, dof)
    A = discretization.jacobian(state)
    rhs = discretization.rhs(state)

    B = read_matrix('data/bous_stretched_%sx%sx%s.txt' % (nx, ny, nz))
    rhs_B = read_vector('data/bous_stretched_rhs_%sx%sx%s.txt' % (nx, ny, nz))

    for i in range(n):
        print(i)

        print('Expected:')
        print(B.jcoA[B.begA[i]:B.begA[i+1]])
        print(B.coA[B.begA[i]:B.begA[i+1]])

        print('Got:')
        print(A.jcoA[A.begA[i]:A.begA[i+1]])
        print(A.coA[A.begA[i]:A.begA[i+1]])

        assert A.begA[i+1] - A.begA[i] == B.begA[i+1] - B.begA[i]
        for j in range(A.begA[i], A.begA[i+1]):
            assert A.jcoA[j] == B.jcoA[j]
            assert A.coA[j] == pytest.approx(B.coA[j])

        assert rhs_B[i] == pytest.approx(rhs[i])

def test_bous_2D():
    nx = 4
    ny = nx
    nz = 1
    dim = 3
    dof1 = 5
    parameters = {'Reynolds Number': 1, 'Rayleigh Number': 100, 'Prandtl Number': 100,
                  'Problem Type': 'Rayleigh-Benard'}
    n = nx * ny * nz * dof1

    state1 = numpy.zeros(n)
    for i in range(n):
        state1[i] = i+1

    discretization = Discretization(parameters, nx, ny, nz, dim, dof1)
    A1 = discretization.jacobian(state1)
    rhs1 = discretization.rhs(state1)

    dim = 2
    dof2 = 4
    n = nx * ny * nz * dof2

    state2 = numpy.zeros(n)
    for i in range(n):
        state2[i] = state1[i + (i + 2) // dof2]

    discretization = Discretization(parameters, nx, ny, nz, dim, dof2)
    A2 = discretization.jacobian(state2)
    rhs2 = discretization.rhs(state2)

    for i in range(n):
        print(i)

        i1 = i + (i + 2) // dof2

        print('Expected:')
        print(A1.jcoA[A1.begA[i1]:A1.begA[i1+1]])
        print(A1.coA[A1.begA[i1]:A1.begA[i1+1]])

        print('Got:')
        print(A2.jcoA[A2.begA[i]:A2.begA[i+1]])
        print(A2.coA[A2.begA[i]:A2.begA[i+1]])

        assert A1.begA[i1+1] - A1.begA[i1] == A2.begA[i+1] - A2.begA[i]
        for j in range(A2.begA[i+1] - A2.begA[i]):
            j1 = A1.begA[i1] + j
            j2 = A2.begA[i] + j
            assert A1.jcoA[j1] == A2.jcoA[j2] + (A2.jcoA[j2] + 2) // dof2
            assert A1.coA[j1] == pytest.approx(A2.coA[j2])

        assert rhs2[i] == pytest.approx(rhs1[i1])

def test_dhc():
    nx = 4
    ny = nx
    nz = nx
    dim = 3
    dof = 5
    parameters = {'Reynolds Number': 1, 'Rayleigh Number': 100, 'Prandtl Number': 100,
                  'Problem Type': 'Differentially Heated Cavity'}
    n = nx * ny * nz * dof

    state = numpy.zeros(n)
    for i in range(n):
        state[i] = i+1

    discretization = Discretization(parameters, nx, ny, nz, dim, dof)
    A = discretization.jacobian(state)
    rhs = discretization.rhs(state)

    B = read_matrix('data/dhc_%sx%sx%s.txt' % (nx, ny, nz))
    rhs_B = read_vector('data/dhc_rhs_%sx%sx%s.txt' % (nx, ny, nz))

    for i in range(n):
        print(i)

        print('Expected:')
        print(B.jcoA[B.begA[i]:B.begA[i+1]])
        print(B.coA[B.begA[i]:B.begA[i+1]])

        print('Got:')
        print(A.jcoA[A.begA[i]:A.begA[i+1]])
        print(A.coA[A.begA[i]:A.begA[i+1]])

        assert A.begA[i+1] - A.begA[i] == B.begA[i+1] - B.begA[i]
        for j in range(A.begA[i], A.begA[i+1]):
            assert A.jcoA[j] == B.jcoA[j]
            assert A.coA[j] == pytest.approx(B.coA[j])

        assert rhs_B[i] == pytest.approx(rhs[i])

def test_tc():
    nx = 4
    ny = nx
    nz = nx
    dim = 3
    dof = 4
    parameters = {'Reynolds Number': 1,
                  'Problem Type': 'Taylor-Couette'}
    n = nx * ny * nz * dof

    state = numpy.zeros(n)
    for i in range(n):
        state[i] = i+1

    discretization = CylindricalDiscretization(parameters, nx, ny, nz, dim, dof)
    A = discretization.jacobian(state)
    rhs = discretization.rhs(state)

    B = read_matrix('data/tc_%sx%sx%s.txt' % (nx, ny, nz))
    rhs_B = read_vector('data/tc_rhs_%sx%sx%s.txt' % (nx, ny, nz))

    for i in range(n):
        print(i)

        print('Expected:')
        print(B.jcoA[B.begA[i]:B.begA[i+1]])
        print(B.coA[B.begA[i]:B.begA[i+1]])

        print('Got:')
        print(A.jcoA[A.begA[i]:A.begA[i+1]])
        print(A.coA[A.begA[i]:A.begA[i+1]])

        assert A.begA[i+1] - A.begA[i] == B.begA[i+1] - B.begA[i]
        for j in range(A.begA[i], A.begA[i+1]):
            assert A.jcoA[j] == B.jcoA[j]
            assert A.coA[j] == pytest.approx(B.coA[j])

        assert rhs_B[i] == pytest.approx(rhs[i])

def test_qg():
    nx = 4
    ny = nx
    nz = 1
    dim = 2
    dof = 4
    parameters = {'Reynolds Number': 16, 'Rossby Parameter': 1000,
                  'Wind Stress Parameter': 1000,
                  'Problem Type': 'Double Gyre'}
    n = nx * ny * nz * dof

    state = numpy.zeros(n)
    for i in range(n):
        state[i] = i+1

    discretization = Discretization(parameters, nx, ny, nz, dim, dof)
    A = discretization.jacobian(state)
    rhs = discretization.rhs(state)

    B = read_matrix('data/qg_%sx%sx%s.txt' % (nx, ny, nz))
    rhs_B = read_vector('data/qg_rhs_%sx%sx%s.txt' % (nx, ny, nz))

    for i in range(n):
        print(i)

        print('Expected:')
        print(B.jcoA[B.begA[i]:B.begA[i+1]])
        print(B.coA[B.begA[i]:B.begA[i+1]])

        print('Got:')
        print(A.jcoA[A.begA[i]:A.begA[i+1]])
        print(A.coA[A.begA[i]:A.begA[i+1]])

        assert A.begA[i+1] - A.begA[i] == B.begA[i+1] - B.begA[i]
        for j in range(A.begA[i], A.begA[i+1]):
            assert A.jcoA[j] == B.jcoA[j]
            assert A.coA[j] == pytest.approx(B.coA[j])

        assert rhs_B[i] == pytest.approx(rhs[i])

def test_amoc():
    nx = 4
    ny = nx
    nz = 1
    dim = 2
    dof = 5
    parameters = {'Reynolds Number': 16, 'Rayleigh Number': 4e4,
                  'Prandtl Number': 2.25, 'Lewis Number': 1,
                  'Temperature Forcing': 1, 'Freshwater Flux': 1,
                  'Problem Type': 'AMOC'}
    n = nx * ny * nz * dof

    state = numpy.zeros(n)
    for i in range(n):
        state[i] = i+1

    discretization = Discretization(parameters, nx, ny, nz, dim, dof)
    A = discretization.jacobian(state)
    rhs = discretization.rhs(state)

    B = read_matrix('data/amoc_%sx%sx%s.txt' % (nx, ny, nz))
    rhs_B = read_vector('data/amoc_rhs_%sx%sx%s.txt' % (nx, ny, nz))

    for i in range(n):
        print(i)

        print('Expected:')
        print(B.jcoA[B.begA[i]:B.begA[i+1]])
        print(B.coA[B.begA[i]:B.begA[i+1]])

        print('Got:')
        print(A.jcoA[A.begA[i]:A.begA[i+1]])
        print(A.coA[A.begA[i]:A.begA[i+1]])

        assert A.begA[i+1] - A.begA[i] == B.begA[i+1] - B.begA[i]
        for j in range(A.begA[i], A.begA[i+1]):
            assert A.jcoA[j] == B.jcoA[j]
            assert A.coA[j] == pytest.approx(B.coA[j])

        assert rhs_B[i] == pytest.approx(rhs[i])

def test_ldc8():
    nx = 8
    ny = nx
    nz = nx
    dim = 3
    dof = 4
    parameters = {'Reynolds Number': 100}
    n = nx * ny * nz * dof

    state = numpy.zeros(n)
    for i in range(n):
        state[i] = i+1

    discretization = Discretization(parameters, nx, ny, nz, dim, dof)
    A = discretization.jacobian(state)
    rhs = discretization.rhs(state)

    if not os.path.isfile('ldc_%sx%sx%s.txt' % (nx, ny, nz)):
        return

    B = read_matrix('data/ldc_%sx%sx%s.txt' % (nx, ny, nz))
    rhs_B = read_vector('data/ldc_rhs_%sx%sx%s.txt' % (nx, ny, nz))

    for i in range(n):
        print(i)

        print('Expected:')
        print(B.jcoA[B.begA[i]:B.begA[i+1]])
        print(B.coA[B.begA[i]:B.begA[i+1]])

        print('Got:')
        print(A.jcoA[A.begA[i]:A.begA[i+1]])
        print(A.coA[A.begA[i]:A.begA[i+1]])

        assert A.begA[i+1] - A.begA[i] == B.begA[i+1] - B.begA[i]
        for j in range(A.begA[i], A.begA[i+1]):
            assert A.jcoA[j] == B.jcoA[j]
            assert A.coA[j] == pytest.approx(B.coA[j])

        assert rhs_B[i] == pytest.approx(rhs[i])


if __name__ == '__main__':
    test_ldc8()
