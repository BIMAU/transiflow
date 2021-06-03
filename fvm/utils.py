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

def create_uniform_coordinate_vector(start, end, nx):
    dx = (end - start) / nx
    # dx = (end - start) / (nx + 1)
    return numpy.roll(numpy.arange(start - dx, end + 2 * dx, dx), -2)


if __name__ == '__main__':
    start = 0
    end = 1
    nx = 5
    res = create_uniform_coordinate_vector(start, end, nx)
    print(res)
