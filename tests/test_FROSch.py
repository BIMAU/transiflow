import pytest
import numpy

from fvm import Continuation

def test_FROSch(nx=4, interactive=False):
    try:
        from fvm import FROSchInterface
        from PyTrilinos import Epetra
    except ImportError:
        pytest.skip("FROSch not found")

    numpy.random.seed(1234)

    dim = 3
    dof = 4
    ny = nx
    nz = nx

    parameters = {'Reynolds Number': 0}

    comm = Epetra.PyComm()
    interface = FROSchInterface.Interface(comm, parameters, nx, ny, nz, dim, dof)
    m = interface.map

    continuation = Continuation(interface, parameters)

    x0 = FROSchInterface.Vector(m)
    x0.PutScalar(0.0)
    x0 = continuation.newton(x0)

    start = 0
    target = 100
    ds = 100
    x = continuation.continuation(x0, 'Reynolds Number', start, target, ds)[0]

    assert x.Norm2() > 0


if __name__ == '__main__':
    test_FROSch()
