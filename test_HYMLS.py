import HYMLSInterface
import continuation

from PyTrilinos import Epetra

def test_HYMLS():
    dof = 4
    nx = 4
    ny = nx
    nz = nx
    n = dof * nx * ny * nz

    comm = Epetra.PyComm()
    m = Epetra.Map(n, 0, comm)
    interface = HYMLSInterface.Interface(m, nx, ny, nz)

    x0 = HYMLSInterface.Vector(m)
    x0.PutScalar(0.0)
    x0 = continuation.newton(interface, x0, 0)

    l = 0
    target = 100
    ds = 100
    maxit = 20
    x = continuation.continuation(interface, x0, l, target, ds, maxit)

    assert x.Norm2() > 0

if __name__ == '__main__':
    test_HYMLS()
