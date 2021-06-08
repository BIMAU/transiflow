import numpy

from scipy import integrate

def create_state_mtx(state, nx, ny, nz, dof):
    '''Helper to create an (nx, ny, nz, dof) dimensional array out of a
    state vector that makes it easier to access the variables.'''

    state_mtx = numpy.zeros([nx, ny, nz, dof])
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                for d in range(dof):
                    state_mtx[i, j, k, d] = state[d + i * dof + j * dof * nx + k * dof * nx * ny]
    return state_mtx

def create_state_vec(state_mtx, nx, ny, nz, dof):
    '''Helper to create a state vector out of an array created with
    create_state_mtx().'''

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
    return numpy.roll(numpy.arange(start - dx, end + 2 * dx, dx), -2)

def create_stretched_coordinate_vector(start, end, nx, sigma):
    if start < 0 or end > 1:
        raise ValueError('Grid stretching currently only works for a [0, 1] domain')

    x = create_uniform_coordinate_vector(start, end, nx)
    x = 0.5 * (1 + numpy.tanh(2 * sigma * (x - 0.5)) / numpy.tanh(sigma))

    # Make cells inside and outside of the boundary the same size
    # to make sure the boundary conditions are applied at the boundary
    dx = x[0] - x[-1]
    if start == 0:
        x[-2] = x[-1] - dx
    if end == 1:
        x[-3] = x[-4] + dx
    return x

def compute_streamfunction(u, v, x, y):
    x, y = numpy.meshgrid(x, y)

    psiv = integrate.cumtrapz(v.T, x, axis=1, initial=0)
    psiu = integrate.cumtrapz(u.T, y, axis=0, initial=0)

    return ((-psiu + psiv[0]) + (psiv - psiu[:, 0][:, None])) / 2

def get_u_value(state, i, j, k, interface):
    '''Get the value of u at a grid point.'''

    nx = interface.discretization.nx
    ny = interface.discretization.ny
    nz = interface.discretization.nz
    dim = interface.discretization.dim
    dof = interface.discretization.dof

    state_mtx = create_state_mtx(state, nx, ny, nz, dof)

    y = interface.discretization.y
    dy1 = (y[i] - y[i-1]) / 2
    dy2 = (y[i+1] - y[i]) / 2

    u1 = (state_mtx[i, j, k, 0] * dy1 + state_mtx[i, j+1, k, 0] * dy2) / (dy1 + dy2)
    if dim == 2:
        return u1

    u2 = (state_mtx[i, j, k+1, 0] * dy1 + state_mtx[i, j+1, k+1, 0] * dy2) / (dy1 + dy2)

    z = interface.discretization.z
    dz1 = (z[i] - z[i-1]) / 2
    dz2 = (z[i+1] - z[i]) / 2

    return (u1 * dz1 + u2 * dz2) / (dz1 + dz2)

def get_v_value(state, i, j, k, interface):
    '''Get the value of v at a grid point.'''

    nx = interface.discretization.nx
    ny = interface.discretization.ny
    nz = interface.discretization.nz
    dim = interface.discretization.dim
    dof = interface.discretization.dof

    state_mtx = create_state_mtx(state, nx, ny, nz, dof)

    x = interface.discretization.x
    dx1 = (x[i] - x[i-1]) / 2
    dx2 = (x[i+1] - x[i]) / 2

    v1 = (state_mtx[i, j, k, 1] * dx1 + state_mtx[i+1, j, k, 1] * dx2) / (dx1 + dx2)
    if dim == 2:
        return v1

    v2 = (state_mtx[i, j, k+1, 1] * dx1 + state_mtx[i+1, j, k+1, 1] * dx2) / (dx1 + dx2)

    z = interface.discretization.z
    dz1 = (z[i] - z[i-1]) / 2
    dz2 = (z[i+1] - z[i]) / 2

    return (v1 * dz1 + v2 * dz2) / (dz1 + dz2)

def get_w_value(state, i, j, k, interface):
    '''Get the value of w at a grid point.'''

    nx = interface.discretization.nx
    ny = interface.discretization.ny
    nz = interface.discretization.nz
    dof = interface.discretization.dof

    state_mtx = create_state_mtx(state, nx, ny, nz, dof)

    x = interface.discretization.x
    dx1 = (x[i] - x[i-1]) / 2
    dx2 = (x[i+1] - x[i]) / 2

    w1 = (state_mtx[i, j, k, 2] * dx1 + state_mtx[i+1, j, k, 2] * dx2) / (dx1 + dx2)
    w2 = (state_mtx[i, j+1, k, 2] * dx1 + state_mtx[i+1, j+1, k, 2] * dx2) / (dx1 + dx2)

    y = interface.discretiyation.y
    dy1 = (y[i] - y[i-1]) / 2
    dy2 = (y[i+1] - y[i]) / 2

    return (w1 * dy1 + w2 * dy2) / (dy1 + dy2)
