import pytest

from fvm import Continuation
from fvm import plot_utils
from fvm import utils

def gather(x):
    from PyTrilinos import Epetra

    local_elements = []
    if x.Comm().MyPID() == 0:
        local_elements = range(x.Map().NumGlobalElements())
    local_map = Epetra.Map(-1, local_elements, 0, x.Comm())
    importer = Epetra.Import(local_map, x.Map())
    out = Epetra.Vector(local_map)
    out.Import(x, importer, Epetra.Insert)
    return out

def test_HYMLS(nx=4, interactive=False):
    try:
        from fvm import HYMLSInterface
        from PyTrilinos import Epetra
        from PyTrilinos import Teuchos
    except ImportError:
        pytest.skip("HYMLS not found")

    dim = 3
    dof = 4
    ny = nx
    nz = nx

    parameters = Teuchos.ParameterList()
    parameters.set('Reynolds Number', 0)
    parameters.set('Bordered Solver', True)

    comm = Epetra.PyComm()
    interface = HYMLSInterface.Interface(comm, parameters, nx, ny, nz, dim, dof)
    m = interface.map

    continuation = Continuation(interface, parameters)

    x0 = HYMLSInterface.Vector(m)
    x0.PutScalar(0.0)
    x0 = continuation.newton(x0)

    target = 100
    ds = 100
    maxit = 20
    x = continuation.continuation(x0, 'Reynolds Number', target, ds, maxit)

    assert x.Norm2() > 0

    if not interactive:
        return

    x = gather(x)
    if comm.MyPID() == 0:
        print(x)

        x = plot_utils.create_velocity_magnitude_mtx(x, nx, ny, nz, dof)
        plot_utils.plot_velocity_magnitude(x[:, ny // 2, :, 0], x[:, ny // 2, :, 2], nx, nz)

def test_HYMLS_2D(nx=8, interactive=False):
    try:
        from fvm import HYMLSInterface
        from PyTrilinos import Epetra
        from PyTrilinos import Teuchos
    except ImportError:
        pytest.skip("HYMLS not found")

    dim = 3
    dof = 4
    ny = nx
    nz = 1

    parameters = Teuchos.ParameterList()
    parameters.set('Reynolds Number', 0)
    parameters.set('Bordered Solver', True)

    comm = Epetra.PyComm()
    interface = HYMLSInterface.Interface(comm, parameters, nx, ny, nz, dim, dof)
    m = interface.map

    continuation = Continuation(interface, parameters)

    x0 = HYMLSInterface.Vector(m)
    x0.PutScalar(0.0)
    x0 = continuation.newton(x0)

    target = 2000
    ds = 100
    maxit = 20
    x = continuation.continuation(x0, 'Reynolds Number', target, ds, maxit)

    assert x.Norm2() > 0

    if not interactive:
        return

    x = gather(x)
    if comm.MyPID() == 0:
        print(x)

        x = plot_utils.create_velocity_magnitude_mtx(x, nx, ny, nz, dof)
        plot_utils.plot_velocity_magnitude(x[:, :, 0, 0], x[:, :, 0, 1], nx, ny)

def test_HYMLS_2D_stretched(nx=8, interactive=False):
    try:
        from fvm import HYMLSInterface
        from PyTrilinos import Epetra
        from PyTrilinos import Teuchos
    except ImportError:
        pytest.skip("HYMLS not found")

    dim = 3
    dof = 4
    ny = nx
    nz = 1

    parameters = Teuchos.ParameterList()
    parameters.set('Reynolds Number', 0)
    parameters.set('Bordered Solver', True)
    parameters.set('Grid Stretching', True)
    parameters.set('Verbose', True)

    comm = Epetra.PyComm()
    interface = HYMLSInterface.Interface(comm, parameters, nx, ny, nz, dim, dof)
    m = interface.map

    continuation = Continuation(interface, parameters)

    x0 = HYMLSInterface.Vector(m)
    x0.PutScalar(0.0)
    x0 = continuation.newton(x0)

    target = 2000
    ds = 100
    maxit = 20
    x = continuation.continuation(x0, 'Reynolds Number', target, ds, maxit)

    assert x.Norm2() > 0

    if not interactive:
        return

    x = gather(x)
    if comm.MyPID() == 0:
        print(x)

        xpos = utils.create_stretched_coordinate_vector(0, 1, nx, 1.5)
        ypos = utils.create_stretched_coordinate_vector(0, 1, ny, 1.5)

        x = plot_utils.create_velocity_magnitude_mtx(x, nx, ny, nz, dof)
        plot_utils.plot_velocity_magnitude(x[:, :, 0, 0], x[:, :, 0, 1], nx, ny, xpos[:-3], ypos[:-3])


if __name__ == '__main__':
    # test_HYMLS(8, True)
    # test_HYMLS_2D(16, True)
    test_HYMLS_2D_stretched(32, True)
