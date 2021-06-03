import numpy

from fvm import Continuation
from fvm import plot_utils
from fvm import Interface

import matplotlib.pyplot as plt

import cProfile
import pstats



def test_continuation(nx=4, interactive=False):
    dim = 3
    dof = 4
    ny = nx
    nz = nx

    parameters = {'Reynolds Number': 0}
    interface = Interface(parameters, nx, ny, nz, dim, dof)

    continuation = Continuation(interface, parameters)

    x0 = numpy.zeros(dof * nx * ny * nz)
    x0 = continuation.newton(x0)

    target = 100
    ds = 100
    maxit = 20
    x = continuation.continuation(x0, 'Reynolds Number', target, ds, maxit)

    assert numpy.linalg.norm(x) > 0

    print(x)
    if not interactive:
        return

    print(x)

    x = plot_utils.create_state_mtx(x, nx, ny, nz, dof)
    plot_utils.plot_state(x[:, ny // 2, :, 0], x[:, ny // 2, :, 2], nx, nz)


def continuation_semi_2D(nx=4, interactive=False):
    dim = 3
    dof = 4
    ny = nx
    nz = 1

    parameters = {'Reynolds Number': 0}
    interface = Interface(parameters, nx, ny, nz, dim, dof)

    continuation = Continuation(interface, parameters)

    x0 = numpy.zeros(dof * nx * ny * nz)
    x0 = continuation.newton(x0)

    target = 2000
    ds = 100
    maxit = 20
    x = continuation.continuation(x0, 'Reynolds Number', target, ds, maxit)

    assert numpy.linalg.norm(x) > 0

    if not interactive:
        return x

    print(x)

    x = plot_utils.create_state_mtx(x, nx, ny, nz, dof)
    plot_utils.plot_state(x[:, :, 0, 0], x[:, :, 0, 1], nx, ny)


def continuation_2D(nx=4, interactive=False):
    dim = 2
    dof = 3
    ny = nx
    nz = 1

    parameters = {'Reynolds Number': 0}
    interface = Interface(parameters, nx, ny, nz, dim, dof)

    continuation = Continuation(interface, parameters)

    x0 = numpy.zeros(dof * nx * ny * nz)
    x0 = continuation.newton(x0)

    target = 2000
    ds = 100
    maxit = 20
    x = continuation.continuation(x0, 'Reynolds Number', target, ds, maxit)

    assert numpy.linalg.norm(x) > 0

    if not interactive:
        return x

    print(x)

    x = plot_utils.create_state_mtx(x, nx, ny, nz, dof)
    plot_utils.plot_state(x[:, :, 0, 0], x[:, :, 0, 1], nx, ny)


def test_continuation_2D_equals():
    x1 = continuation_2D()
    x2 = continuation_semi_2D()

    dof1 = 3
    dof2 = 4

    assert numpy.linalg.norm(x1[0:-1:dof1] - x2[0:-1:dof2]) < 1e-2
    assert numpy.linalg.norm(x1[1:-1:dof1] - x2[1:-1:dof2]) < 1e-2
    assert numpy.linalg.norm(x1[2:-1:dof1] - x2[3:-1:dof2]) < 1e-2


#TODO wei
# C_c = 3.513830719
def test_continuation_Bratu_problem(para, ds, nx=4, interactive=False):
    dim = 1
    dof = 1
    ny = 1
    nz = 1

    parameters = para
    interface = Interface(parameters, nx, ny, nz, dim, dof)

    continuation = Continuation(interface, parameters)

    x0 = numpy.zeros(dof * (nx-1) * ny * nz)
    x0 = continuation.newton(x0)

    target = 3
    ds = ds
    maxit = int(4 / ds * 2) * 100

    (x, para, u, u_norm, C_v, iterations) = continuation.continuation(x0, 'Bratu parameter', target, ds, maxit)

    # print(para[200])
    # plt.plot(interface.discretization.x[0:15], u[200])
    # plt.xlabel('x')
    # plt.ylabel('u(x)')
    # plt.show()
    assert numpy.linalg.norm(x) > 0

    # if not interactive:
    #     return

    # print(x)
    return para, u, u_norm, C_v, iterations


if __name__ == '__main__':
    # ILU(0)
    # para = {'Bordered Solver': True, 'Bratu parameter': 0, 'Problem Type': 'Bratu problem', 'Use Iterative Solver': True, 'Use Preconditioner': True, 'Use LU Preconditioner': False, 'Use ILU Preconditioner': True}

    # No Precondtioner
    # para = {'Bordered Solver': False, 'Bratu parameter': 0, 'Problem Type': 'Bratu problem', 'Use Iterative Solver': True, 'Use Preconditioner': False, 'Use LU Preconditioner': True, 'Use ILU Preconditoner': False}

    # Direct solver
    para = {'Bordered Solver': True, 'Bratu parameter': 0, 'Problem Type': 'Bratu problem', 'Use Iterative Solver': False, 'Use Preconditioner': True, 'Use LU Preconditioner': False, 'Use ILU Preconditoner': False}

    N = 16

    ds = 0.1

    # (p1, u1, u_norm1, c1, i1) = test_continuation_Bratu_problem(para, ds, N, False)

    # (p2, u2, u_norm2, c2, i2) = test_continuation_Bratu_problem(para, ds, N*4, False)
    # (p3, u3, u_norm3, c3, i3) = test_continuation_Bratu_problem(para, ds, N*16, False)
    (p4, u4, u_norm4, c4, i4) = test_continuation_Bratu_problem(para, ds, N, False)

    # print('when nx = %d and ds=%e, the points that are close to the turning point is ' %(N, ds), c1)
    # print('when nx = %d and ds=%e, the number of newton corrector iterations is  ' % (N, ds), i1)

    # print('max p1=', max(p1))
    # print('max p2=', max(p2))
    # print('max p3=', max(p3))

    print('max p4=', max(p4))


    # plt.plot(p1, u_norm1, color='red', label='nx=%d' % N)
    # plt.plot(p2, u_norm2, color='blue', label='nx=%d' % (4*N))
    # plt.plot(p3, u_norm3, color='green', label='nx=%d' % (16*N))
    plt.plot(p4, u_norm4, color='red', label='nx=%d' % N)
    # #
    # #
    plt.legend()
    plt.xlabel('Bratu parameter C')
    plt.ylabel('Infinity Norm of u(x)')
    plt.show()



    # measure time
    # prof = cProfile.Profile()
    # prof.run('test_continuation_Bratu_problem(para, ds, N, False)')
    # prof.dump_stats('output.prof')
    #
    # stream1 = open('output_direct.txt', 'w')
    # stream2 = open('output_gmres.txt', 'w')
    #
    # stats1 = pstats.Stats('output.prof', stream=stream1) # dont't fully understand TODO
    # stats2 = pstats.Stats('output.prof', stream=stream2)
    #
    # stats1.strip_dirs().print_stats('SuperLU','solve', 'objects')
    # stats2.strip_dirs().print_stats('gmres', 1)
    #
    # stream1.close()
    # stream2.close()
    #
    # if para.get('Use Iterative Solver') is False:
    #     x = open('output_direct.txt', 'r+')
    # else:
    #     x = open('output_gmres.txt', 'r+')
    # data = x.readlines()[-3]
    # y = data.split()
    # print(y)


