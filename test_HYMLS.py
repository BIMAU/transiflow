import continuation

def test_HYMLS():
    try:
        import HYMLSInterface
        from PyTrilinos import Epetra
        from PyTrilinos import Teuchos
    except ImportError:
        return

    nx = 4
    ny = nx
    nz = nx

    params = Teuchos.ParameterList()
    prec_params = params.sublist('Preconditioner')
    prec_params.set('Separator Length', 4)
    prec_params.set('Number of Levels', 0)

    comm = Epetra.PyComm()
    interface = HYMLSInterface.Interface(comm, params, nx, ny, nz)
    m = interface.map

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
