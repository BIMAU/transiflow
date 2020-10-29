import numpy

def create_state_mtx(state, nx, ny, nz, dof):
    state_mtx = numpy.zeros([nx, ny, nz, dof])
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                for d in range(dof):
                    state_mtx[i, j, k, d] = state[d + i * dof + j * dof * nx + k * dof * nx * ny]
    return state_mtx

def create_state_vec(state_mtx, nx, ny, nz, dof):
    state = numpy.zeros(nx * ny * nz * dof)

    row = 0
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                for d in range(dof):
                    state[row] = state_mtx[i, j, k, d]
                    row += 1
    return state

def create_uniform_coordinate_vector(nx):
    dx = 1 / nx
    return numpy.roll(numpy.arange(-dx, 1+2*dx, dx), -2)
