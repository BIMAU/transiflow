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

    atom = numpy.zeros([nx, ny, nz, 3, 3, 3])

    return (nx, ny, nz, x, y, z, atom)

def test_u_xx():
    nx, ny, nz, x, y, z, atom = create_test_problem()

    fvm.Derivatives.u_xx(atom, nx, ny, nz, x, y, z)

    for i in range(nx):
        dx = x[i] - x[i-1]
        dxp1 = x[i+1] - x[i]
        for j in range(ny):
            dy = y[j] - y[j-1]
            for k in range(nz):
                dz = z[k] - z[k-1]
                print(i, j, k)
                assert atom[i, j, k, 0, 1, 1] == pytest.approx(1 / dx * dy * dz)
                assert atom[i, j, k, 2, 1, 1] == pytest.approx(1 / dxp1 * dy * dz)

def test_v_yy():
    nx, ny, nz, x, y, z, atom = create_test_problem()

    fvm.Derivatives.v_yy(atom, nx, ny, nz, x, y, z)

    for i in range(nx):
        dx = x[i] - x[i-1]
        for j in range(ny):
            dy = y[j] - y[j-1]
            dyp1 = y[j+1] - y[j]
            for k in range(nz):
                dz = z[k] - z[k-1]
                print(i, j, k)
                assert atom[i, j, k, 1, 0, 1] == pytest.approx(1 / dy * dx * dz)
                assert atom[i, j, k, 1, 2, 1] == pytest.approx(1 / dyp1 * dx * dz)

def test_w_zz():
    nx, ny, nz, x, y, z, atom = create_test_problem()

    fvm.Derivatives.w_zz(atom, nx, ny, nz, x, y, z)

    for i in range(nx):
        dx = x[i] - x[i-1]
        for j in range(ny):
            dy = y[j] - y[j-1]
            for k in range(nz):
                dz = z[k] - z[k-1]
                dzp1 = z[k+1] - z[k]
                print(i, j, k)
                assert atom[i, j, k, 1, 1, 0] == pytest.approx(1 / dz * dy * dx)
                assert atom[i, j, k, 1, 1, 2] == pytest.approx(1 / dzp1 * dy * dx)

def test_u_yy():
    nx, ny, nz, x, y, z, atom = create_test_problem()

    fvm.Derivatives.u_yy(atom, nx, ny, nz, x, y, z)

    for i in range(nx):
        dx = (x[i+1] - x[i-1]) / 2
        for j in range(ny):
            dy = (y[j] - y[j-2]) / 2
            dyp1 = (y[j+1] - y[j-1]) / 2
            for k in range(nz):
                dz = z[k] - z[k-1]
                print(i, j, k)
                assert atom[i, j, k, 0, 1, 1] == pytest.approx(1 / dy * dx * dz)
                assert atom[i, j, k, 2, 1, 1] == pytest.approx(1 / dyp1 * dx * dz)

def test_v_xx():
    nx, ny, nz, x, y, z, atom = create_test_problem()

    fvm.Derivatives.v_xx(atom, nx, ny, nz, x, y, z)

    for i in range(nx):
        dx = (x[i] - x[i-2]) / 2
        dxp1 = (x[i+1] - x[i-1]) / 2
        for j in range(ny):
            dy = (y[j+1] - y[j-1]) / 2
            for k in range(nz):
                dz = z[k] - z[k-1]
                print(i, j, k)
                assert atom[i, j, k, 1, 0, 1] == pytest.approx(1 / dx * dy * dz)
                assert atom[i, j, k, 1, 2, 1] == pytest.approx(1 / dxp1 * dy * dz)

def test_w_yy():
    nx, ny, nz, x, y, z, atom = create_test_problem()

    fvm.Derivatives.w_yy(atom, nx, ny, nz, x, y, z)

    for i in range(nx):
        dx = x[i] - x[i-1]
        for j in range(ny):
            dy = (y[j] - y[j-2]) / 2
            dyp1 = (y[j+1] - y[j-1]) / 2
            for k in range(nz):
                dz = (z[k+1] - z[k-1]) / 2
                print(i, j, k)
                assert atom[i, j, k, 1, 1, 0] == pytest.approx(1 / dy * dz * dx)
                assert atom[i, j, k, 1, 1, 2] == pytest.approx(1 / dyp1 * dz * dx)

def test_u_zz():
    nx, ny, nz, x, y, z, atom = create_test_problem()

    fvm.Derivatives.u_zz(atom, nx, ny, nz, x, y, z)

    for i in range(nx):
        dx = (x[i+1] - x[i-1]) / 2
        for j in range(ny):
            dy = y[j] - y[j-1]
            for k in range(nz):
                dz = (z[k] - z[k-2]) / 2
                dzp1 = (z[k+1] - z[k-1]) / 2
                print(i, j, k)
                assert atom[i, j, k, 0, 1, 1] == pytest.approx(1 / dz * dx * dy)
                assert atom[i, j, k, 2, 1, 1] == pytest.approx(1 / dzp1 * dx * dy)

def test_v_zz():
    nx, ny, nz, x, y, z, atom = create_test_problem()

    fvm.Derivatives.v_zz(atom, nx, ny, nz, x, y, z)

    for i in range(nx):
        dx = x[i] - x[i-1]
        for j in range(ny):
            dy = (y[j+1] - y[j-1]) / 2
            for k in range(nz):
                dz = (z[k] - z[k-2]) / 2
                dzp1 = (z[k+1] - z[k-1]) / 2
                print(i, j, k)
                assert atom[i, j, k, 1, 0, 1] == pytest.approx(1 / dz * dy * dx)
                assert atom[i, j, k, 1, 2, 1] == pytest.approx(1 / dzp1 * dy * dx)

def test_w_xx():
    nx, ny, nz, x, y, z, atom = create_test_problem()

    fvm.Derivatives.w_xx(atom, nx, ny, nz, x, y, z)

    for i in range(nx):
        dx = (x[i] - x[i-2]) / 2
        dxp1 = (x[i+1] - x[i-1]) / 2
        for j in range(ny):
            dy = y[j] - y[j-1]
            for k in range(nz):
                dz = (z[k+1] - z[k-1]) / 2
                print(i, j, k)
                assert atom[i, j, k, 1, 1, 0] == pytest.approx(1 / dx * dz * dy)
                assert atom[i, j, k, 1, 1, 2] == pytest.approx(1 / dxp1 * dz * dy)

def test_p_x():
    nx, ny, nz, x, y, z, atom = create_test_problem()

    fvm.Derivatives.p_x(atom, nx, ny, nz, x, y, z)

    for i in range(nx):
        for j in range(ny):
            dy = y[j] - y[j-1]
            for k in range(nz):
                dz = z[k] - z[k-1]
                print(i, j, k)
                assert atom[i, j, k, 1, 1, 1] == pytest.approx(-dy * dz)
                assert atom[i, j, k, 2, 1, 1] == pytest.approx(dy * dz)

def test_p_y():
    nx, ny, nz, x, y, z, atom = create_test_problem()

    fvm.Derivatives.p_y(atom, nx, ny, nz, x, y, z)

    for i in range(nx):
        dx = x[i] - x[i-1]
        for j in range(ny):
            for k in range(nz):
                dz = z[k] - z[k-1]
                print(i, j, k)
                assert atom[i, j, k, 1, 1, 1] == pytest.approx(-dx * dz)
                assert atom[i, j, k, 1, 2, 1] == pytest.approx(dx * dz)

def test_p_z():
    nx, ny, nz, x, y, z, atom = create_test_problem()

    fvm.Derivatives.p_z(atom, nx, ny, nz, x, y, z)

    for i in range(nx):
        dx = x[i] - x[i-1]
        for j in range(ny):
            dy = y[j] - y[j-1]
            for k in range(nz):
                print(i, j, k)
                assert atom[i, j, k, 1, 1, 1] == pytest.approx(-dy * dx)
                assert atom[i, j, k, 1, 1, 2] == pytest.approx(dy * dx)

def test_u_x():
    nx, ny, nz, x, y, z, atom = create_test_problem()

    fvm.Derivatives.u_x(atom, nx, ny, nz, x, y, z)

    for i in range(nx):
        for j in range(ny):
            dy = y[j] - y[j-1]
            for k in range(nz):
                dz = z[k] - z[k-1]
                print(i, j, k)
                assert atom[i, j, k, 0, 1, 1] == pytest.approx(-dy * dz)
                assert atom[i, j, k, 1, 1, 1] == pytest.approx(dy * dz)

def test_u_y():
    nx, ny, nz, x, y, z, atom = create_test_problem()

    fvm.Derivatives.u_y(atom, nx, ny, nz, x, y, z)

    for i in range(nx):
        dx = x[i] - x[i-1]
        for j in range(ny):
            for k in range(nz):
                dz = z[k] - z[k-1]
                print(i, j, k)
                assert atom[i, j, k, 1, 0, 1] == pytest.approx(-dx * dz)
                assert atom[i, j, k, 1, 1, 1] == pytest.approx(dx * dz)

def test_u_z():
    nx, ny, nz, x, y, z, atom = create_test_problem()

    fvm.Derivatives.u_z(atom, nx, ny, nz, x, y, z)

    for i in range(nx):
        dx = x[i] - x[i-1]
        for j in range(ny):
            dy = y[j] - y[j-1]
            for k in range(nz):
                print(i, j, k)
                assert atom[i, j, k, 1, 1, 0] == pytest.approx(-dy * dx)
                assert atom[i, j, k, 1, 1, 1] == pytest.approx(dy * dx)
