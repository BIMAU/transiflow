import numpy
import pytest

from transiflow import TimeIntegration
from transiflow import Continuation
from transiflow import utils

from transiflow.interface import create


def Interface(*args, backend="SciPy", **kwargs):
    try:
        return create(*args, backend=backend, **kwargs)
    except ImportError:
        pytest.skip(backend + " not found")


@pytest.mark.parametrize("backend", ["SciPy", "Epetra", "HYMLS", "PETSc"])
def test_continuation_ldc(backend, nx=4):
    numpy.random.seed(1234)

    ny = nx
    nz = nx

    parameters = {}
    interface = Interface(parameters, nx, ny, nz, backend=backend)
    continuation = Continuation(interface)

    x0 = interface.vector()
    x0 = continuation.newton(x0)

    start = 0
    target = 100
    ds = 100
    x = continuation.continuation(x0, 'Reynolds Number', start, target, ds)[0]

    assert utils.norm(x) > 0

def continuation_ldc_semi_2D(backend, nx=4):
    numpy.random.seed(1234)

    dim = 3
    ny = nx
    nz = 1

    parameters = {}
    interface = Interface(parameters, nx, ny, nz, dim, backend=backend)
    continuation = Continuation(interface)

    x0 = interface.vector()
    x0 = continuation.newton(x0)

    start = 0
    target = 2000
    ds = 100
    x = continuation.continuation(x0, 'Reynolds Number', start, target, ds)[0]

    assert utils.norm(x) > 0
    return interface.array_from_vector(x)

def continuation_ldc_2D(backend, nx=4):
    numpy.random.seed(1234)

    dim = 2
    ny = nx
    nz = 1

    parameters = {}
    interface = Interface(parameters, nx, ny, nz, dim, backend=backend)
    continuation = Continuation(interface)

    x0 = interface.vector()
    x0 = continuation.newton(x0)

    start = 0
    target = 2000
    ds = 100
    x = continuation.continuation(x0, 'Reynolds Number', start, target, ds)[0]

    assert utils.norm(x) > 0
    return interface.array_from_vector(x)

@pytest.mark.parametrize("backend", ["SciPy", "HYMLS", "PETSc"])
def test_continuation_ldc_2D_equals(backend):
    x1 = continuation_ldc_2D(backend)
    x2 = continuation_ldc_semi_2D(backend)

    dof1 = 3
    dof2 = 4

    assert utils.norm(x1[0:-1:dof1] - x2[0:-1:dof2]) < 1e-2
    assert utils.norm(x1[1:-1:dof1] - x2[1:-1:dof2]) < 1e-2
    assert utils.norm(x1[2:-1:dof1] - x2[3:-1:dof2]) < 1e-2

@pytest.mark.parametrize("backend", ["SciPy", "Epetra", "HYMLS", "PETSc"])
def test_continuation_ldc_2D_stretched(backend, nx=8):
    numpy.random.seed(1234)

    ny = nx

    parameters = {'Grid Stretching': True}
    interface = Interface(parameters, nx, ny, backend=backend)

    x = interface.discretization.get_coordinate_vector(0, 1, nx)
    y = interface.discretization.get_coordinate_vector(0, 1, ny)
    assert x[1] - x[0] < x[2] - x[1]
    assert y[1] - y[0] < y[2] - y[1]

    continuation = Continuation(interface)

    x0 = interface.vector()
    x0 = continuation.newton(x0)

    start = 0
    target = 2000
    ds = 100
    x = continuation.continuation(x0, 'Reynolds Number', start, target, ds)[0]

    assert utils.norm(x) > 0

def test_continuation_time_integration(nx=4):
    numpy.random.seed(1234)

    dof = 3
    ny = nx
    nz = 1

    parameters = {}
    interface = Interface(parameters, nx, ny, nz, dof=dof)
    continuation = Continuation(interface, newton_tolerance=1e-8)

    x0 = interface.vector()
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

    time_integration = TimeIntegration(interface)
    x3 = time_integration.integration(x2, 1, 1)[0]

    assert numpy.linalg.norm(x[0:len(x):dof] - x3[0:len(x):dof]) > 1e-2
    assert numpy.linalg.norm(x[1:len(x):dof] - x3[1:len(x):dof]) > 1e-2

    x3 = time_integration.integration(x2, 100, 1000)[0]

    assert numpy.linalg.norm(x[0:len(x):dof] - x3[0:len(x):dof]) < 1e-4
    assert numpy.linalg.norm(x[1:len(x):dof] - x3[1:len(x):dof]) < 1e-4

    # Start from zero
    x2[:] = 0

    x3 = time_integration.integration(x2, 100, 1000)[0]

    assert numpy.linalg.norm(x[0:len(x):dof] - x3[0:len(x):dof]) < 1e-4
    assert numpy.linalg.norm(x[1:len(x):dof] - x3[1:len(x):dof]) < 1e-4

@pytest.mark.parametrize("backend", ["SciPy", "HYMLS"])
def test_continuation_dhc(backend, nx=8):
    try:
        from transiflow.interface import JaDa # noqa: F401
    except ImportError:
        pytest.skip('jadapy not found')

    numpy.random.seed(1234)

    ny = nx

    parameters = {'Problem Type': 'Differentially Heated Cavity',
                  'Rayleigh Number': 1,
                  'Prandtl Number': 1000,
                  'Reynolds Number': 1}

    interface = Interface(parameters, nx, ny, backend=backend)
    continuation = Continuation(interface, newton_tolerance=1e-9)

    x0 = interface.vector()

    start = 0
    target = 9e7
    ds = 1e4
    x, mu = continuation.continuation(x0, 'Rayleigh Number', start, target, ds, ds_max=1e7)

    parameters['Eigenvalue Solver'] = {}
    parameters['Eigenvalue Solver']['Number of Eigenvalues'] = 2

    target = 1e9
    ds = 5e6
    x, mu = continuation.continuation(x, 'Rayleigh Number', mu, target, ds, ds_max=5e6,
                                      detect_bifurcations=True)

    assert utils.norm(x) > 0
    assert mu > 9e7
    assert mu < target

@pytest.mark.parametrize("backend", ["SciPy", "HYMLS"])
def test_continuation_rayleigh_benard(backend, nx=8):
    try:
        from transiflow.interface import JaDa # noqa: F401
    except ImportError:
        pytest.skip('jadapy not found')

    numpy.random.seed(1234)

    ny = nx

    parameters = {'Problem Type': 'Rayleigh-Benard',
                  'Prandtl Number': 10,
                  'Biot Number': 1,
                  'X-max': 10,
                  'Bordered Solver': True}

    interface = Interface(parameters, nx, ny, backend=backend)
    continuation = Continuation(interface)

    x0 = interface.vector()
    x0 = continuation.newton(x0)

    start = 0
    target = 1800
    ds = 200
    x, mu = continuation.continuation(x0, 'Rayleigh Number', start, target, ds)

    parameters['Eigenvalue Solver'] = {}
    parameters['Eigenvalue Solver']['Arithmetic'] = 'real'
    parameters['Eigenvalue Solver']['Number of Eigenvalues'] = 2

    target = 5000
    ds = 50
    x2, mu2, dx, dmu = continuation.continuation(x, 'Rayleigh Number', mu, target, ds,
                                                 detect_bifurcations=True,
                                                 return_step=True)

    assert utils.norm(x2) > 0
    assert mu2 > mu
    assert mu2 < target

    x3, mu3 = continuation.continuation(x2, 'Rayleigh Number', mu2, target, ds,
                                        detect_bifurcations=True)

    assert utils.norm(x3) > 0
    assert mu3 > mu2
    assert mu3 < target

@pytest.mark.parametrize("backend", ["SciPy", "HYMLS"])
def test_continuation_rayleigh_benard_formulations_2D(backend, nx=8):
    try:
        from transiflow.interface import JaDa # noqa: F401
    except ImportError:
        pytest.skip('jadapy not found')

    numpy.random.seed(1234)

    dim = 2
    dof = 4
    ny = nx
    nz = 1

    parameters = {'Problem Type': 'Rayleigh-Benard',
                  'Prandtl Number': 10,
                  'Biot Number': 1,
                  'X-max': 10,
                  'Bordered Solver': True}

    interface = Interface(parameters, nx, ny, nz, dim, dof, backend=backend)
    continuation = Continuation(interface)

    x0 = interface.vector()
    x0 = continuation.newton(x0)

    start = 0
    target = 1800
    ds = 200
    x, mu = continuation.continuation(x0, 'Rayleigh Number', start, target, ds)

    parameters['Eigenvalue Solver'] = {}
    parameters['Eigenvalue Solver']['Arithmetic'] = 'real'
    parameters['Eigenvalue Solver']['Number of Eigenvalues'] = 2

    target = 5000
    ds = 50
    x2, mu2 = continuation.continuation(x, 'Rayleigh Number', mu, target, ds,
                                        detect_bifurcations=True)

    assert utils.norm(x2) > 0
    assert mu2 > mu
    assert mu2 < target

    parameters['Problem Type'] = 'Rayleigh-Benard Perturbation'

    # Subtract the motionless state
    y = interface.discretization.get_coordinate_vector(0, 1, ny)
    t = numpy.zeros((nx, ny, nz, dof))
    for j in range(ny):
        t[:, j, 0, dim+1] = 1 - (y[j] + y[j-1]) / 4
    t = utils.create_state_vec(t, nx, ny, nz, dof)
    x = x - interface.vector_from_array(t)

    x3, mu3 = continuation.continuation(x, 'Rayleigh Number', mu, target, ds,
                                        detect_bifurcations=True)

    # Test that the solution obtained from both formulations are the same
    x2 = interface.array_from_vector(x2)
    x3 = interface.array_from_vector(x3)

    assert utils.norm(x3[0:x.size:dof] - x2[0:x.size:dof]) < 1e-4
    assert utils.norm(x3[1:x.size:dof] - x2[1:x.size:dof]) < 1e-4
    assert utils.norm(x3[3:x.size:dof] - x2[3:x.size:dof] + t[3:x.size:dof]) < 1e-4
    assert mu3 > mu
    assert mu3 < target
    assert abs(mu3 - mu2) < 1

@pytest.mark.parametrize("backend", ["SciPy", "HYMLS"])
def test_continuation_rayleigh_benard_formulations(backend, nx=4):
    try:
        from transiflow.interface import JaDa # noqa: F401
    except ImportError:
        pytest.skip('jadapy not found')

    numpy.random.seed(1234)

    dim = 3
    dof = 5
    ny = nx
    nz = nx

    parameters = {'Problem Type': 'Rayleigh-Benard',
                  'Prandtl Number': 10,
                  'Biot Number': 1,
                  'X-max': 10,
                  'Y-max': 10,
                  'Bordered Solver': True}

    interface = Interface(parameters, nx, ny, nz, dim, dof, backend=backend)
    continuation = Continuation(interface)

    x0 = interface.vector()
    x0 = continuation.newton(x0)

    start = 0
    target = 1800
    ds = 200
    x, mu = continuation.continuation(x0, 'Rayleigh Number', start, target, ds)

    parameters['Eigenvalue Solver'] = {}
    parameters['Eigenvalue Solver']['Arithmetic'] = 'real'
    parameters['Eigenvalue Solver']['Number of Eigenvalues'] = 2

    target = 5000
    ds = 50
    x2, mu2 = continuation.continuation(x, 'Rayleigh Number', mu, target, ds,
                                        detect_bifurcations=True)

    assert utils.norm(x2) > 0
    assert mu2 > mu
    assert mu2 < target

    parameters['Problem Type'] = 'Rayleigh-Benard Perturbation'

    # Subtract the motionless state
    z = interface.discretization.get_coordinate_vector(0, 1, nz)
    t = numpy.zeros((nx, ny, nz, dof))
    for k in range(nz):
        t[:, :, k, dim+1] = 1 - (z[k] + z[k-1]) / 4
    t = utils.create_state_vec(t, nx, ny, nz, dof)
    x = x - interface.vector_from_array(t)

    x3, mu3 = continuation.continuation(x, 'Rayleigh Number', mu, target, ds,
                                        detect_bifurcations=True)

    # Test that the solution obtained from both formulations are the same
    x2 = interface.array_from_vector(x2)
    x3 = interface.array_from_vector(x3)

    assert utils.norm(x3[0:x.size:dof] - x2[0:x.size:dof]) < 1e-4
    assert utils.norm(x3[1:x.size:dof] - x2[1:x.size:dof]) < 1e-4
    assert utils.norm(x3[2:x.size:dof] - x2[2:x.size:dof]) < 1e-4
    assert utils.norm(x3[4:x.size:dof] - x2[4:x.size:dof] + t[4:x.size:dof]) < 1e-4
    assert mu3 > mu
    assert mu3 < target
    assert abs(mu3 - mu2) < 1

@pytest.mark.parametrize("backend", ["SciPy", "HYMLS"])
def test_continuation_double_gyre(backend, nx=8):
    try:
        from transiflow.interface import JaDa # noqa: F401
    except ImportError:
        pytest.skip('jadapy not found')

    numpy.random.seed(1234)

    ny = nx

    parameters = {'Problem Type': 'Double Gyre',
                  'Reynolds Number': 16,
                  'Rossby Parameter': 1000,
                  'Wind Stress Parameter': 0}

    interface = Interface(parameters, nx, ny, backend=backend)
    continuation = Continuation(interface)

    x0 = interface.vector()

    start = 0
    target = 1000
    ds = 200
    x, mu = continuation.continuation(x0, 'Wind Stress Parameter', start, target, ds)

    parameters['Eigenvalue Solver'] = {}
    parameters['Eigenvalue Solver']['Number of Eigenvalues'] = 2

    target = 100
    ds = 5
    x, mu = continuation.continuation(x, 'Reynolds Number', 16, target, ds,
                                      detect_bifurcations=True)

    assert utils.norm(x) > 0
    assert mu > 16
    assert mu < target

@pytest.mark.parametrize("backend", ["SciPy", "HYMLS"])
def test_continuation_amoc(backend, nx=16):
    try:
        from transiflow.interface import JaDa # noqa: F401
    except ImportError:
        pytest.skip('jadapy not found')

    numpy.random.seed(1234)

    ny = nx // 2

    parameters = {'Problem Type': 'AMOC',
                  'Rayleigh Number': 4e4,
                  'Prandtl Number': 2.25,
                  'X-max': 5}

    interface = Interface(parameters, nx, ny, backend=backend)
    continuation = Continuation(interface)

    x0 = interface.vector()

    target = 1
    ds = 0.1
    x, mu = continuation.continuation(x0, 'Temperature Forcing', 0, target, ds)

    parameters['Eigenvalue Solver'] = {}
    parameters['Eigenvalue Solver']['Number of Eigenvalues'] = 2

    target = 0.2
    ds = 0.01
    continuation = Continuation(interface, newton_tolerance=1e-6)
    x, mu = continuation.continuation(x, 'Freshwater Flux', 0, target,
                                      ds, ds_min=1e-6,
                                      detect_bifurcations=True)

    assert numpy.linalg.norm(x) > 0
    assert mu > 0.1
    assert mu < target

def test_continuation_2D_tc(nx=8):
    try:
        from transiflow.interface import JaDa # noqa: F401
    except ImportError:
        pytest.skip('jadapy not found')

    numpy.random.seed(1234)

    ny = 1
    nz = nx

    ri = 0.8
    ro = 1

    parameters = {'Problem Type': 'Taylor-Couette',
                  'Reynolds Number': 1,
                  'R-min': ri,
                  'R-max': ro,
                  'Z-max': 1,
                  'Z-periodic': True,
                  'Inner Angular Velocity': 1 / ri / (ro - ri),
                  'Outer Angular Velocity': 0}

    interface = Interface(parameters, nx, ny, nz)
    continuation = Continuation(interface)

    x0 = interface.vector()

    start = 0
    target = 80
    ds = 30
    x, mu = continuation.continuation(x0, 'Reynolds Number', start, target, ds)

    parameters['Bordered Solver'] = True
    parameters['Eigenvalue Solver'] = {}
    parameters['Eigenvalue Solver']['Number of Eigenvalues'] = 2
    parameters['Eigenvalue Solver']['Arithmetic'] = 'real'

    target = 100
    ds = 1
    continuation = Continuation(interface, newton_tolerance=1e-12)
    x, mu = continuation.continuation(x, 'Reynolds Number', mu, target, ds,
                                      detect_bifurcations=True)

    assert numpy.linalg.norm(x) > 0
    assert mu > 0
    assert mu < target
