import os
import numpy
import fvm
import pytest

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

    x = create_coordinate_vector(nx)
    y = create_coordinate_vector(ny)
    z = create_coordinate_vector(nz)

    return (nx, ny, nz, x, y, z)

def test_u_xx():
    nx, ny, nz, x, y, z = create_test_problem()

    derivatives = fvm.Derivatives(nx, ny, nz)
    atom = derivatives.u_xx(x, y, z)

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
    nx, ny, nz, x, y, z = create_test_problem()

    derivatives = fvm.Derivatives(nx, ny, nz)
    atom =  derivatives.v_yy(x, y, z)

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
    nx, ny, nz, x, y, z = create_test_problem()

    derivatives = fvm.Derivatives(nx, ny, nz)
    atom =  derivatives.w_zz(x, y, z)

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
    nx, ny, nz, x, y, z = create_test_problem()

    derivatives = fvm.Derivatives(nx, ny, nz)
    atom =  derivatives.u_yy(x, y, z)

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
    nx, ny, nz, x, y, z = create_test_problem()

    derivatives = fvm.Derivatives(nx, ny, nz)
    atom =  derivatives.v_xx(x, y, z)

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
    nx, ny, nz, x, y, z = create_test_problem()

    derivatives = fvm.Derivatives(nx, ny, nz)
    atom =  derivatives.w_yy(x, y, z)

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
    nx, ny, nz, x, y, z = create_test_problem()

    derivatives = fvm.Derivatives(nx, ny, nz)
    atom =  derivatives.u_zz(x, y, z)

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
    nx, ny, nz, x, y, z = create_test_problem()

    derivatives = fvm.Derivatives(nx, ny, nz)
    atom =  derivatives.v_zz(x, y, z)

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
    nx, ny, nz, x, y, z = create_test_problem()

    derivatives = fvm.Derivatives(nx, ny, nz)
    atom =  derivatives.w_xx(x, y, z)

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
    nx, ny, nz, x, y, z = create_test_problem()

    derivatives = fvm.Derivatives(nx, ny, nz)
    atom =  derivatives.p_x(x, y, z)

    for i in range(nx):
        for j in range(ny):
            dy = y[j] - y[j-1]
            for k in range(nz):
                dz = z[k] - z[k-1]
                print(i, j, k)
                assert atom[i, j, k, 0, 3, 1, 1, 1] == pytest.approx(-dy * dz)
                assert atom[i, j, k, 0, 3, 2, 1, 1] == pytest.approx(dy * dz)

def test_p_y():
    nx, ny, nz, x, y, z = create_test_problem()

    derivatives = fvm.Derivatives(nx, ny, nz)
    atom =  derivatives.p_y(x, y, z)

    for i in range(nx):
        dx = x[i] - x[i-1]
        for j in range(ny):
            for k in range(nz):
                dz = z[k] - z[k-1]
                print(i, j, k)
                assert atom[i, j, k, 1, 3, 1, 1, 1] == pytest.approx(-dx * dz)
                assert atom[i, j, k, 1, 3, 1, 2, 1] == pytest.approx(dx * dz)

def test_p_z():
    nx, ny, nz, x, y, z = create_test_problem()

    derivatives = fvm.Derivatives(nx, ny, nz)
    atom =  derivatives.p_z(x, y, z)

    for i in range(nx):
        dx = x[i] - x[i-1]
        for j in range(ny):
            dy = y[j] - y[j-1]
            for k in range(nz):
                print(i, j, k)
                assert atom[i, j, k, 2, 3, 1, 1, 1] == pytest.approx(-dy * dx)
                assert atom[i, j, k, 2, 3, 1, 1, 2] == pytest.approx(dy * dx)

def test_u_x():
    nx, ny, nz, x, y, z = create_test_problem()

    derivatives = fvm.Derivatives(nx, ny, nz)
    atom =  derivatives.u_x(x, y, z)

    for i in range(nx):
        for j in range(ny):
            dy = y[j] - y[j-1]
            for k in range(nz):
                dz = z[k] - z[k-1]
                print(i, j, k)
                assert atom[i, j, k, 3, 0, 0, 1, 1] == pytest.approx(-dy * dz)
                assert atom[i, j, k, 3, 0, 1, 1, 1] == pytest.approx(dy * dz)

def test_u_y():
    nx, ny, nz, x, y, z = create_test_problem()

    derivatives = fvm.Derivatives(nx, ny, nz)
    atom =  derivatives.v_y(x, y, z)

    for i in range(nx):
        dx = x[i] - x[i-1]
        for j in range(ny):
            for k in range(nz):
                dz = z[k] - z[k-1]
                print(i, j, k)
                assert atom[i, j, k, 3, 1, 1, 0, 1] == pytest.approx(-dx * dz)
                assert atom[i, j, k, 3, 1, 1, 1, 1] == pytest.approx(dx * dz)

def test_u_z():
    nx, ny, nz, x, y, z = create_test_problem()

    derivatives = fvm.Derivatives(nx, ny, nz)
    atom =  derivatives.w_z(x, y, z)

    for i in range(nx):
        dx = x[i] - x[i-1]
        for j in range(ny):
            dy = y[j] - y[j-1]
            for k in range(nz):
                print(i, j, k)
                assert atom[i, j, k, 3, 2, 1, 1, 0] == pytest.approx(-dy * dx)
                assert atom[i, j, k, 3, 2, 1, 1, 1] == pytest.approx(dy * dx)

def test_MxU():
    nx, ny, nz, x, y, z = create_test_problem()
    dof = 4
    n = dof * nx * ny * nz

    bil = numpy.zeros([nx, ny, nz, 6, 3, 2])
    convective_term = fvm.ConvectiveTerm(nx, ny, nz)
    convective_term.averages(bil)

    state = numpy.zeros(n)
    for i in range(n):
        state[i] = i+1

    state_mtx = numpy.zeros([nx, ny, nz, dof])
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                for d in range(dof):
                    state_mtx[i, j, k, d] = state[d + i * dof + j * dof * nx + k * dof * nx * ny]

    convective_term.dirichlet_east(bil)
    convective_term.dirichlet_west(bil)
    convective_term.dirichlet_north(bil)
    convective_term.dirichlet_south(bil)
    convective_term.dirichlet_top(bil)
    convective_term.dirichlet_bottom(bil)

    averages = numpy.zeros([nx, ny, nz, 3, 3])
    convective_term.MxU(averages, bil, state_mtx)

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
    A = fvm.CrsMatrix([], [], [0])

    with open(fname, 'r') as f:
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

def read_vector(fname):
    vec = numpy.array([])

    with open(fname, 'r') as f:
        rows = []
        idx = 0
        for i in f.readlines():
            vec = numpy.append(vec, float(i.strip()))
    return vec

def test_lin():
    nx = 4
    ny = nx
    nz = nx
    dof = 4
    Re = 100
    n = nx * ny * nz * dof

    atom = fvm.linear_part(Re, nx, ny, nz)
    A = fvm.assemble(atom, nx, ny, nz)

    B = read_matrix('lin_%sx%sx%s.txt' % (nx, ny, nz))

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

def test_bnd():
    nx = 4
    ny = nx
    nz = nx
    dof = 4
    Re = 100
    n = nx * ny * nz * dof

    atom = fvm.linear_part(Re, nx, ny, nz)
    fvm.boundaries(atom, nx, ny, nz)
    A = fvm.assemble(atom, nx, ny, nz)

    B = read_matrix('bnd_%sx%sx%s.txt' % (nx, ny, nz))

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

def test_bil():
    nx = 4
    ny = nx
    nz = nx
    dof = 4
    Re = 100
    n = nx * ny * nz * dof

    state = numpy.zeros(n)
    for i in range(n):
        state[i] = i+1

    atom, atomF = fvm.convection(state, nx, ny, nz)
    A = fvm.assemble(atom, nx, ny, nz)

    B = read_matrix('bil_%sx%sx%s.txt' % (nx, ny, nz))

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

def test_full():
    nx = 4
    ny = nx
    nz = nx
    dof = 4
    Re = 100
    n = nx * ny * nz * dof

    state = numpy.zeros(n)
    for i in range(n):
        state[i] = i+1

    atom = fvm.linear_part(Re, nx, ny, nz)
    frc = fvm.boundaries(atom, nx, ny, nz)
    atomJ, atomF = fvm.convection(state, nx, ny, nz)

    atomJ += atom
    atomF += atom

    A = fvm.assemble(atomJ, nx, ny, nz)
    rhs = fvm.rhs(state, atomF, nx, ny, nz) - frc

    B = read_matrix('full_%sx%sx%s.txt' % (nx, ny, nz))
    rhs_B = read_vector('rhs_%sx%sx%s.txt' % (nx, ny, nz))

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

def test_full8():
    nx = 8
    ny = nx
    nz = nx
    dof = 4
    Re = 100
    n = nx * ny * nz * dof

    state = numpy.zeros(n)
    for i in range(n):
        state[i] = i+1

    atom = fvm.linear_part(Re, nx, ny, nz)
    frc = fvm.boundaries(atom, nx, ny, nz)
    atomJ, atomF = fvm.convection(state, nx, ny, nz)

    atomJ += atom
    atomF += atom

    A = fvm.assemble(atomJ, nx, ny, nz)
    rhs = fvm.rhs(state, atomF, nx, ny, nz) - frc

    if not os.path.isfile('full_%sx%sx%s.txt' % (nx, ny, nz)):
        return

    B = read_matrix('full_%sx%sx%s.txt' % (nx, ny, nz))
    rhs_B = read_vector('rhs_%sx%sx%s.txt' % (nx, ny, nz))

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
