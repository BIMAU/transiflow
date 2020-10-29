import os
import numpy
import pytest

from fvm import CrsMatrix
from fvm import Discretization

def create_coordinate_vector(nx):
    dx = 1 / (nx + 1)
    a = []
    for i in range(nx+3):
        a.append(-dx + dx**i)
    return numpy.roll(a, -2)

def create_test_problem():
    nx = 5
    ny = 3
    nz = 2
    dof = 4

    x = create_coordinate_vector(nx)
    y = create_coordinate_vector(ny)
    z = create_coordinate_vector(nz)

    return (nx, ny, nz, dof, x, y, z)

def test_u_xx():
    nx, ny, nz, dof, x, y, z = create_test_problem()

    discretization = Discretization(nx, ny, nz, dof)
    atom = discretization.u_xx(x, y, z)

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
    nx, ny, nz, dof, x, y, z = create_test_problem()

    discretization = Discretization(nx, ny, nz, dof)
    atom =  discretization.v_yy(x, y, z)

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
    nx, ny, nz, dof, x, y, z = create_test_problem()

    discretization = Discretization(nx, ny, nz, dof)
    atom =  discretization.w_zz(x, y, z)

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
    nx, ny, nz, dof, x, y, z = create_test_problem()

    discretization = Discretization(nx, ny, nz, dof)
    atom =  discretization.u_yy(x, y, z)

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
    nx, ny, nz, dof, x, y, z = create_test_problem()

    discretization = Discretization(nx, ny, nz, dof)
    atom =  discretization.v_xx(x, y, z)

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
    nx, ny, nz, dof, x, y, z = create_test_problem()

    discretization = Discretization(nx, ny, nz, dof)
    atom =  discretization.w_yy(x, y, z)

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
    nx, ny, nz, dof, x, y, z = create_test_problem()

    discretization = Discretization(nx, ny, nz, dof)
    atom =  discretization.u_zz(x, y, z)

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
    nx, ny, nz, dof, x, y, z = create_test_problem()

    discretization = Discretization(nx, ny, nz, dof)
    atom =  discretization.v_zz(x, y, z)

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
    nx, ny, nz, dof, x, y, z = create_test_problem()

    discretization = Discretization(nx, ny, nz, dof)
    atom =  discretization.w_xx(x, y, z)

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

def test_p_x():
    nx, ny, nz, dof, x, y, z = create_test_problem()

    discretization = Discretization(nx, ny, nz, dof)
    atom =  discretization.p_x(x, y, z)

    for i in range(nx):
        for j in range(ny):
            dy = y[j] - y[j-1]
            for k in range(nz):
                dz = z[k] - z[k-1]
                print(i, j, k)
                assert atom[i, j, k, 0, 3, 1, 1, 1] == pytest.approx(-dy * dz)
                assert atom[i, j, k, 0, 3, 2, 1, 1] == pytest.approx(dy * dz)

def test_p_y():
    nx, ny, nz, dof, x, y, z = create_test_problem()

    discretization = Discretization(nx, ny, nz, dof)
    atom =  discretization.p_y(x, y, z)

    for i in range(nx):
        dx = x[i] - x[i-1]
        for j in range(ny):
            for k in range(nz):
                dz = z[k] - z[k-1]
                print(i, j, k)
                assert atom[i, j, k, 1, 3, 1, 1, 1] == pytest.approx(-dx * dz)
                assert atom[i, j, k, 1, 3, 1, 2, 1] == pytest.approx(dx * dz)

def test_p_z():
    nx, ny, nz, dof, x, y, z = create_test_problem()

    discretization = Discretization(nx, ny, nz, dof)
    atom =  discretization.p_z(x, y, z)

    for i in range(nx):
        dx = x[i] - x[i-1]
        for j in range(ny):
            dy = y[j] - y[j-1]
            for k in range(nz):
                print(i, j, k)
                assert atom[i, j, k, 2, 3, 1, 1, 1] == pytest.approx(-dy * dx)
                assert atom[i, j, k, 2, 3, 1, 1, 2] == pytest.approx(dy * dx)

def test_u_x():
    nx, ny, nz, dof, x, y, z = create_test_problem()

    discretization = Discretization(nx, ny, nz, dof)
    atom =  discretization.u_x(x, y, z)

    for i in range(nx):
        for j in range(ny):
            dy = y[j] - y[j-1]
            for k in range(nz):
                dz = z[k] - z[k-1]
                print(i, j, k)
                assert atom[i, j, k, 3, 0, 0, 1, 1] == pytest.approx(-dy * dz)
                assert atom[i, j, k, 3, 0, 1, 1, 1] == pytest.approx(dy * dz)

def test_u_y():
    nx, ny, nz, dof, x, y, z = create_test_problem()

    discretization = Discretization(nx, ny, nz, dof)
    atom =  discretization.v_y(x, y, z)

    for i in range(nx):
        dx = x[i] - x[i-1]
        for j in range(ny):
            for k in range(nz):
                dz = z[k] - z[k-1]
                print(i, j, k)
                assert atom[i, j, k, 3, 1, 1, 0, 1] == pytest.approx(-dx * dz)
                assert atom[i, j, k, 3, 1, 1, 1, 1] == pytest.approx(dx * dz)

def test_u_z():
    nx, ny, nz, dof, x, y, z = create_test_problem()

    discretization = Discretization(nx, ny, nz, dof)
    atom =  discretization.w_z(x, y, z)

    for i in range(nx):
        dx = x[i] - x[i-1]
        for j in range(ny):
            dy = y[j] - y[j-1]
            for k in range(nz):
                print(i, j, k)
                assert atom[i, j, k, 3, 2, 1, 1, 0] == pytest.approx(-dy * dx)
                assert atom[i, j, k, 3, 2, 1, 1, 1] == pytest.approx(dy * dx)

def test_MxU():
    import importlib.util
    spec = importlib.util.spec_from_file_location('Discretization', 'fvm/Discretization.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    nx, ny, nz, dof, x, y, z = create_test_problem()
    dof = 4
    n = dof * nx * ny * nz

    bil = numpy.zeros([nx, ny, nz, 2, dof, dof, 2])
    convective_term = module.ConvectiveTerm(nx, ny, nz)

    state = numpy.zeros(n)
    for i in range(n):
        state[i] = i+1

    state_mtx = numpy.zeros([nx, ny, nz, dof])
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                for d in range(dof):
                    state_mtx[i, j, k, d] = state[d + i * dof + j * dof * nx + k * dof * nx * ny]

    averages = numpy.zeros([nx, ny, nz, 3, 3])
    convective_term.backward_average_x(bil[:, :, :, :, :, 0, :], averages[:, :, :, :, 0], state_mtx[:, :, :, 0])

    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                average = 0
                if i < nx-1:
                    average += state[i * dof + j * dof * nx + k * dof * nx * ny] / 2
                if i > 0:
                     average += state[(i-1) * dof + j * dof * nx + k * dof * nx * ny] / 2
                print(i, j, k)
                assert averages[i, j, k, 0, 0] == average

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

def read_bous_matrix(fname):
    A = read_matrix(fname)

    dof = 5
    B = CrsMatrix([], [], [0])
    # Swap indices since the Fortran code had T at position 3
    for i in range(len(A.begA)-1):
        i2 = i + (i % dof == 3) - (i % dof == 4)
        if i2 + 1 >= len(A.begA):
            continue

        row = []
        for j in range(A.begA[i2], A.begA[i2+1]):
            col = A.jcoA[j] + (A.jcoA[j] % dof == 3) - (A.jcoA[j] % dof == 4)
            row.append((col, A.coA[j]))

        row = sorted(row, key=lambda col: col[0])
        for col, val in row:
            B.jcoA.append(col)
            B.coA.append(val)
        B.begA.append(len(B.jcoA))

    return B

def read_vector(fname):
    vec = numpy.array([])

    dirname = os.path.dirname(__file__)
    with open(os.path.join(dirname, fname), 'r') as f:
        rows = []
        idx = 0
        for i in f.readlines():
            vec = numpy.append(vec, float(i.strip()))
    return vec

def read_bous_vector(fname):
    vec = read_vector(fname)

    dof = 5
    out = numpy.zeros(len(vec))
    for i in range(len(vec)):
        out[i] = vec[i + (i % dof == 3) - (i % dof == 4)]
    return out

def test_ldc_lin():
    nx = 4
    ny = nx
    nz = nx
    dof = 4
    Re = 100
    n = nx * ny * nz * dof

    discretization = Discretization(nx, ny, nz, dof)
    atom = discretization.linear_part(Re)
    A = discretization.jacobian(atom)

    B = read_matrix('ldc_lin_%sx%sx%s.txt' % (nx, ny, nz))

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
    dof = 5
    Re = 1
    Ra = 100
    Pr = 100
    n = nx * ny * nz * dof

    discretization = Discretization(nx, ny, nz, dof)
    atom = discretization.linear_part(Re, Ra, Pr)
    A = discretization.jacobian(atom)

    B = read_bous_matrix('bous_lin_%sx%sx%s.txt' % (nx, ny, nz))

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
    dof = 4
    Re = 100
    n = nx * ny * nz * dof

    discretization = Discretization(nx, ny, nz, dof)
    atom = discretization.linear_part(Re)
    discretization.boundaries(atom)
    A = discretization.jacobian(atom)

    B = read_matrix('ldc_bnd_%sx%sx%s.txt' % (nx, ny, nz))

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
    dof = 5
    Re = 1
    Ra = 100
    Pr = 100
    n = nx * ny * nz * dof

    discretization = Discretization(nx, ny, nz, dof)
    atom = discretization.linear_part(Re, Ra, Pr)
    discretization.boundaries(atom, 'Rayleigh-Benard')
    A = discretization.jacobian(atom)

    B = read_bous_matrix('bous_bnd_%sx%sx%s.txt' % (nx, ny, nz))

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
    dof = 4
    n = nx * ny * nz * dof

    state = numpy.zeros(n)
    for i in range(n):
        state[i] = i+1

    discretization = Discretization(nx, ny, nz, dof)
    atom, atomF = discretization.nonlinear_part(state)
    A = discretization.jacobian(atom)

    B = read_matrix('ldc_bil_%sx%sx%s.txt' % (nx, ny, nz))

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

        assert B.begA[i+1] - B.begA[i] == A.begA[i+1] - A.begA[i]
        for j in range(B.begA[i], B.begA[i+1]):
            assert B.jcoA[j] == A.jcoA[j]
            assert B.coA[j] == A.coA[j]

def test_bous_bil():
    nx = 4
    ny = nx
    nz = nx
    dof = 5
    n = nx * ny * nz * dof

    state = numpy.zeros(n)
    for i in range(n):
        state[i] = i+1

    discretization = Discretization(nx, ny, nz, dof)
    atom, atomF = discretization.nonlinear_part(state)
    A = discretization.jacobian(atom)

    B = read_bous_matrix('bous_bil_%sx%sx%s.txt' % (nx, ny, nz))

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

        assert B.begA[i+1] - B.begA[i] == A.begA[i+1] - A.begA[i]
        for j in range(B.begA[i], B.begA[i+1]):
            assert B.jcoA[j] == A.jcoA[j]
            assert B.coA[j] == A.coA[j]

def test_ldc():
    nx = 4
    ny = nx
    nz = nx
    dof = 4
    Re = 100
    n = nx * ny * nz * dof

    state = numpy.zeros(n)
    for i in range(n):
        state[i] = i+1

    discretization = Discretization(nx, ny, nz, dof)
    atom = discretization.linear_part(Re)
    frc = discretization.boundaries(atom)
    atomJ, atomF = discretization.nonlinear_part(state)

    atomJ += atom
    atomF += atom

    A = discretization.jacobian(atomJ)
    rhs = discretization.rhs(state, atomF) - frc

    B = read_matrix('ldc_%sx%sx%s.txt' % (nx, ny, nz))
    rhs_B = read_vector('ldc_rhs_%sx%sx%s.txt' % (nx, ny, nz))

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

        assert rhs_B[i] == pytest.approx(-rhs[i])

def test_bous():
    nx = 4
    ny = nx
    nz = nx
    dof = 5
    Re = 1
    Ra = 100
    Pr = 100
    n = nx * ny * nz * dof

    state = numpy.zeros(n)
    for i in range(n):
        state[i] = i+1

    discretization = Discretization(nx, ny, nz, dof)
    atom = discretization.linear_part(Re, Ra, Pr)
    frc = discretization.boundaries(atom, 'Rayleigh-Benard')
    atomJ, atomF = discretization.nonlinear_part(state)

    atomJ += atom
    atomF += atom

    A = discretization.jacobian(atomJ)
    rhs = discretization.rhs(state, atomF) - frc

    B = read_bous_matrix('bous_%sx%sx%s.txt' % (nx, ny, nz))
    rhs_B = read_bous_vector('bous_rhs_%sx%sx%s.txt' % (nx, ny, nz))

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

        assert rhs_B[i] == pytest.approx(-rhs[i])

def test_ldc8():
    nx = 8
    ny = nx
    nz = nx
    dof = 4
    Re = 100
    n = nx * ny * nz * dof

    state = numpy.zeros(n)
    for i in range(n):
        state[i] = i+1

    discretization = Discretization(nx, ny, nz, dof)
    atom = discretization.linear_part(Re)
    frc = discretization.boundaries(atom)
    atomJ, atomF = discretization.nonlinear_part(state)

    atomJ += atom
    atomF += atom

    A = discretization.jacobian(atomJ)
    rhs = discretization.rhs(state, atomF) - frc

    if not os.path.isfile('ldc_%sx%sx%s.txt' % (nx, ny, nz)):
        return

    B = read_matrix('ldc_%sx%sx%s.txt' % (nx, ny, nz))
    rhs_B = read_vector('ldc_rhs_%sx%sx%s.txt' % (nx, ny, nz))

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

        assert rhs_B[i] == pytest.approx(-rhs[i])

if __name__ == '__main__':
    test_full()
