import pytest
import numpy as np
from fvm import Interface, Continuation

import matplotlib.pyplot as plt

def solve_nonlinear_system(para, nx=4):
    dim = 1
    dof = 1
    ny = 1
    nz = 1

    parameters = para
    interface = Interface(parameters, nx, ny, nz, dim, dof)

    continuation = Continuation(interface, parameters)

    x0 = np.zeros(dof * (nx-1) * ny * nz)
    x0 = continuation.newton(x0)
    return x0


def test_second_order_convergence(para, N=16):
    u1 = solve_nonlinear_system(para, N)
    u2 = solve_nonlinear_system(para, 2 * N)
    u3 = solve_nonlinear_system(para, 4 * N)

    res = (u3[3] - u2[1]) / (u2[1] - u1[0])
    print("(u_h - u_2h) / (u_2h - u_4h) = ", res)

    # assert abs(res - 0.25) < 1e-2

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
def HYMLS_Bratu_problem(nx=8, interactive=False):

    try:
        from fvm import HYMLSInterface
        from PyTrilinos import Epetra
        from PyTrilinos import Teuchos
    except ImportError:
        pytest.skip("HYMLS not found")

    dim = 1
    dof = 1
    ny = 1
    nz = 1

    parameters = Teuchos.ParameterList()
    parameters.set('Bratu parameter', 0)
    parameters.set('Bordered Solver', True)
    parameters.set('Problem Type', 'Bratu problem')


    comm = Epetra.PyComm()
    interface = HYMLSInterface.Interface(comm, parameters, nx, ny, nz, dim, dof)
    m = interface.map

    continuation = Continuation(interface, parameters)

    x0 = HYMLSInterface.Vector(m)
    x0.PutScalar(0.0)
    x0 = continuation.newton(x0)

    target = 3
    ds = 0.1
    maxit = int(3.6 / ds * 2)

    (x, paras, u) = continuation.continuation(x0, 'Bratu parameter', target, ds, maxit)

    assert x.Norm2() > 0

    if not interactive:
        return

    x = gather(x)
    if comm.MyPID() == 0:
        print(x)

        # x = plot_utils.create_state_mtx(x, nx, ny, nz, dof)
        # plot_utils.plot_state(x[:, :, 0, 0], x[:, :, 0, 1], nx, ny)
    plt.plot(paras, u)
    plt.title('nx = ' + repr(nx))
    plt.xlabel('Bratu parameter C')
    plt.ylabel('Infinity Norm of u(x)')
    plt.show()

if __name__ == '__main__':
    parameters = {'Bordered Solver': True, 'Bratu parameter': 3, 'Problem Type': 'Bratu problem',
            'Use Iterative Solver': False, 'Use Preconditioner': True, 'Use LU Preconditioner': False,
            'Use ILU Preconditoner': False}

    N = 16
    test_second_order_convergence(parameters, N)
    # HYMLS_Bratu_problem(16)

