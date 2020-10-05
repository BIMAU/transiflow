import continuation
import numpy
import matplotlib.pyplot as plt

def get_state_mtx(x, nx, ny, nz, dof):
    state_mtx = numpy.zeros([nx, ny, nz, dof])
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                for d in range(dof):
                    state_mtx[i, j, k, d] = x[d + i * dof + j * dof * nx + k * dof * nx * ny]
    return state_mtx

def plot_state(u, v, nx, ny):
    psi = numpy.zeros([nx, ny])

    for i in range(nx):
        for j in range(ny):
            psiu = u[i, j]
            if j > 0:
                psiu += u[i, j-1]
            psiv = v[i, j]
            if i > 0:
                psiv += v[i-1, j]
            psi[i, j] = numpy.linalg.norm([psiu, psiv])

    e = numpy.arange(1/nx, 1+1/nx, 1/nx)
    x, y = numpy.meshgrid(e, e)

    fig1, ax1 = plt.subplots()
    cs = ax1.contourf(x, y, psi.transpose(), 15)
    fig1.colorbar(cs)
    plt.show()

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

    # x = get_state_mtx(x, nx, ny, nz, dof)
    # plot_state(x[:,ny//2,:,0], x[:,ny//2,:,2], nx, nz)

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

    # x = get_state_mtx(x, nx, ny, nz, dof)
    # plot_state(x[:,:,0,0], x[:,:,0,1], nx, ny)

if __name__ == '__main__':
    test_continuation()
    # test_continuation_2d()
