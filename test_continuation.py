import continuation
import numpy
import plot_utils

def test_continuation():
    dof = 4
    nx = 4
    ny = nx
    nz = nx
    n = dof * nx * ny * nz

    interface = continuation.Interface(nx, ny, nz)

    x0 = numpy.zeros(dof * nx * ny * nz)
    x0 = continuation.newton(interface, x0, 0)

    l = 0
    target = 100
    ds = 100
    maxit = 20
    x = continuation.continuation(interface, x0, l, target, ds, maxit)

    assert numpy.linalg.norm(x) > 0

    # print(x)

    # x = plot_utils.get_state_mtx(x, nx, ny, nz, dof)
    # plot_utils.plot_state(x[:,ny//2,:,0], x[:,ny//2,:,2], nx, nz)

def test_continuation_2d():
    dof = 4
    nx = 8
    ny = nx
    nz = 1
    n = dof * nx * ny * nz

    interface = continuation.Interface(nx, ny, nz)

    x0 = numpy.zeros(dof * nx * ny * nz)
    x0 = continuation.newton(interface, x0, 0)

    l = 0
    target = 2000
    ds = 100
    maxit = 20
    x = continuation.continuation(interface, x0, l, target, ds, maxit)

    assert numpy.linalg.norm(x) > 0

    # print(x)

    # x = plot_utils.get_state_mtx(x, nx, ny, nz, dof)
    # plot_utils.plot_state(x[:,:,0,0], x[:,:,0,1], nx, ny)

if __name__ == '__main__':
    test_continuation()
    # test_continuation_2d()
