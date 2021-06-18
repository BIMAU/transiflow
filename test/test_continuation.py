import numpy
import pytest

from fvm import TimeIntegration
from fvm import Continuation
from fvm import plot_utils
from fvm import Interface
from fvm import utils

def test_continuation(nx=4, interactive=False):
    dim = 3
    dof = 4
    ny = nx
    nz = nx

    parameters = {}
    interface = Interface(parameters, nx, ny, nz, dim, dof)

    continuation = Continuation(interface, parameters)

    x0 = numpy.zeros(dof * nx * ny * nz)
    x0 = continuation.newton(x0)

    start = 0
    target = 100
    ds = 100
    x = continuation.continuation(x0, 'Reynolds Number', start, target, ds)[0]

    assert numpy.linalg.norm(x) > 0

    if not interactive:
        return

    print(x)

    x = plot_utils.create_state_mtx(x, nx, ny, nz, dof)
    plot_utils.plot_velocity_magnitude(x[:, ny // 2, :, 0], x[:, ny // 2, :, 2], nx, nz)

def continuation_semi_2D(nx=4, interactive=False):
    dim = 3
    dof = 4
    ny = nx
    nz = 1

    parameters = {}
    interface = Interface(parameters, nx, ny, nz, dim, dof)

    continuation = Continuation(interface, parameters)

    x0 = numpy.zeros(dof * nx * ny * nz)
    x0 = continuation.newton(x0)

    start = 0
    target = 2000
    ds = 100
    x = continuation.continuation(x0, 'Reynolds Number', start, target, ds)[0]

    assert numpy.linalg.norm(x) > 0

    if not interactive:
        return x

    print(x)

    x = plot_utils.create_state_mtx(x, nx, ny, nz, dof)
    plot_utils.plot_velocity_magnitude(x[:, :, 0, 0], x[:, :, 0, 1], interface)

def continuation_2D(nx=4, interactive=False):
    dim = 2
    dof = 3
    ny = nx
    nz = 1

    parameters = {}
    interface = Interface(parameters, nx, ny, nz, dim, dof)

    continuation = Continuation(interface, parameters)

    x0 = numpy.zeros(dof * nx * ny * nz)
    x0 = continuation.newton(x0)

    start = 0
    target = 2000
    ds = 100
    x = continuation.continuation(x0, 'Reynolds Number', start, target, ds)[0]

    assert numpy.linalg.norm(x) > 0

    if not interactive:
        return x

    print(x)

    x = plot_utils.create_state_mtx(x, nx, ny, nz, dof)
    plot_utils.plot_velocity_magnitude(x[:, :, 0, 0], x[:, :, 0, 1], interface)

def test_continuation_2D_equals():
    x1 = continuation_2D()
    x2 = continuation_semi_2D()

    dof1 = 3
    dof2 = 4

    assert numpy.linalg.norm(x1[0:-1:dof1] - x2[0:-1:dof2]) < 1e-2
    assert numpy.linalg.norm(x1[1:-1:dof1] - x2[1:-1:dof2]) < 1e-2
    assert numpy.linalg.norm(x1[2:-1:dof1] - x2[3:-1:dof2]) < 1e-2

def test_continuation_2D_stretched(nx=4, interactive=False):
    dim = 2
    dof = 3
    ny = nx
    nz = 1

    xpos = utils.create_stretched_coordinate_vector(0, 1, nx, 1.5)
    ypos = utils.create_stretched_coordinate_vector(0, 1, ny, 1.5)

    parameters = {}
    interface = Interface(parameters, nx, ny, nz, dim, dof, xpos, ypos)

    continuation = Continuation(interface, parameters)

    x0 = numpy.zeros(dof * nx * ny * nz)
    x0 = continuation.newton(x0)

    start = 0
    target = 2000
    ds = 100
    x = continuation.continuation(x0, 'Reynolds Number', start, target, ds)[0]

    assert numpy.linalg.norm(x) > 0

    if not interactive:
        return x

    print(x)

    x = plot_utils.create_state_mtx(x, nx, ny, nz, dof)
    plot_utils.plot_velocity_magnitude(x[:, :, 0, 0], x[:, :, 0, 1], interface)

def test_continuation_bifurcation(nx=8, interactive=False):
    try:
        from fvm import JadaInterface # noqa: F401
    except ImportError:
        pytest.skip('jadapy not found')

    dim = 2
    dof = 4
    ny = nx
    nz = 1

    parameters = {'Problem Type': 'Rayleigh-Benard',
                  'Prandtl Number': 10,
                  'Biot Number': 1,
                  'xmax': 10,
                  'Bordered Solver': True}

    interface = Interface(parameters, nx, ny, nz, dim, dof)

    continuation = Continuation(interface, parameters)

    x0 = numpy.zeros(dof * nx * ny * nz)
    x0 = continuation.newton(x0)

    start = 0
    target = 1700
    ds = 200
    x, mu, _ = continuation.continuation(x0, 'Rayleigh Number', start, target, ds)

    parameters['Detect Bifurcation Points'] = True
    parameters['Eigenvalue Solver'] = {}
    parameters['Eigenvalue Solver']['Number of Eigenvalues'] = 2

    target = 5000
    ds = 50
    x, mu, _ = continuation.continuation(x, 'Rayleigh Number', mu, target, ds)

    assert numpy.linalg.norm(x) > 0
    assert mu > 0
    assert mu < target

def test_continuation_time_integration(nx=4, interactive=False):
    dim = 2
    dof = 3
    ny = nx
    nz = 1

    parameters = {'Newton Tolerance': 1e-6}
    interface = Interface(parameters, nx, ny, nz, dim, dof)

    continuation = Continuation(interface, parameters)

    x0 = numpy.zeros(dof * nx * ny * nz)
    x0 = continuation.newton(x0)

    start = 0
    target = 2000
    ds = 100
    x = continuation.continuation(x0, 'Reynolds Number', start, target, ds)[0]

    # Start from a perturbed solution
    x2 = utils.create_state_mtx(x, nx, ny, nz, dof)
    x2[1:nx-1, 1:ny-1, :, 0] += 0.1 * numpy.random.random((nx - 2, ny - 2, nz))
    x2 = utils.create_state_vec(x2, nx, ny, nz, dof)

    assert numpy.linalg.norm(x[0:len(x):dof] - x2[0:len(x):dof]) > 1e-2
    assert numpy.linalg.norm(x[1:len(x):dof] - x2[1:len(x):dof]) < 1e-4

    time_integration = TimeIntegration(interface, parameters)
    x3 = time_integration.integration(x2, 1, 1)[0]

    assert numpy.linalg.norm(x[0:len(x):dof] - x3[0:len(x):dof]) > 1e-2
    assert numpy.linalg.norm(x[1:len(x):dof] - x3[1:len(x):dof]) > 1e-2

    time_integration = TimeIntegration(interface, parameters)
    x3 = time_integration.integration(x2, 100, 1000)[0]

    assert numpy.linalg.norm(x[0:len(x):dof] - x3[0:len(x):dof]) < 1e-4
    assert numpy.linalg.norm(x[1:len(x):dof] - x3[1:len(x):dof]) < 1e-4

    # Start from zero
    x2[:] = 0

    time_integration = TimeIntegration(interface, parameters)
    x3 = time_integration.integration(x2, 100, 1000)[0]

    assert numpy.linalg.norm(x[0:len(x):dof] - x3[0:len(x):dof]) < 1e-4
    assert numpy.linalg.norm(x[1:len(x):dof] - x3[1:len(x):dof]) < 1e-4

def test_continuation_double_gyre(nx=8, interactive=False):
    try:
        from fvm import JadaInterface # noqa: F401
    except ImportError:
        pytest.skip('jadapy not found')

    dim = 2
    dof = 3
    ny = nx
    nz = 1

    parameters = {'Problem Type': 'Double Gyre',
                  'Reynolds Number': 16,
                  'Rossby Parameter': 1000,
                  'Wind Stress Parameter': 0}

    interface = Interface(parameters, nx, ny, nz, dim, dof)

    continuation = Continuation(interface, parameters)

    x0 = numpy.zeros(dof * nx * ny * nz)

    start = 0
    target = 1000
    ds = 200
    x, mu, _ = continuation.continuation(x0, 'Wind Stress Parameter', start, target, ds)

    parameters['Detect Bifurcation Points'] = True
    parameters['Eigenvalue Solver'] = {}
    parameters['Eigenvalue Solver']['Number of Eigenvalues'] = 2

    target = 100
    ds = 5
    x, mu, _ = continuation.continuation(x, 'Reynolds Number', 16, target, ds)

    assert numpy.linalg.norm(x) > 0
    assert mu > 0
    assert mu < target


if __name__ == '__main__':
    # test_continuation(8, False)
    # continuation_2D(16, True)
    test_continuation_2D_stretched(32, True)
