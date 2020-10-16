import numpy
import matplotlib.pyplot as plt

from fvm import create_state_mtx

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
