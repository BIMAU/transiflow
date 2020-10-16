import continuation
import plot_utils

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

def test_HYMLS():
    try:
        import HYMLSInterface
        from PyTrilinos import Epetra
        from PyTrilinos import Teuchos
    except ImportError:
        return

    dof = 4
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

    # x = gather(x)
    # if comm.MyPID() == 0:
    #     print(x)

    #     x = plot_utils.get_state_mtx(x, nx, ny, nz, dof)
    #     plot_utils.plot_state(x[:,ny//2,:,0], x[:,ny//2,:,2], nx, nz)

if __name__ == '__main__':
    test_HYMLS()
