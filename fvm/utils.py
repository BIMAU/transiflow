import numpy

from math import sqrt

def dot(x, y):
    try:
        return x.T.conj() @ y
    except AttributeError:
        return x.dot(y)

def norm(x):
    if len(x.shape) > 1:
        ret = numpy.zeros(x.shape[1])
        for i in range(x.shape[1]):
            ret[i] = sqrt(abs(dot(x[:, i], x[:, i])))
        return ret

    return sqrt(abs(dot(x, x)))

def create_state_mtx(state, nx=None, ny=None, nz=None, dof=None, interface=None):
    '''Helper to create an (nx, ny, nz, dof) dimensional array out of a
    state vector that makes it easier to access the variables.'''

    if interface:
        nx = interface.discretization.nx
        ny = interface.discretization.ny
        nz = interface.discretization.nz
        dof = interface.discretization.dof

    state_mtx = numpy.zeros([nx, ny, nz, dof])
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                for d in range(dof):
                    state_mtx[i, j, k, d] = state[d + i * dof + j * dof * nx + k * dof * nx * ny]
    return state_mtx

def create_padded_state_mtx(state, nx=None, ny=None, nz=None, dof=None,
                            x_periodic=True, y_periodic=True, z_periodic=True,
                            interface=None):
    '''Helper to create an (nx+2, ny+2, nz+2, dof) dimensional array out of a
    state vector that makes it easier to access the variables. The value from
    the other side of the domain is padded to each border.'''

    if interface:
        nx = interface.discretization.nx
        ny = interface.discretization.ny
        nz = interface.discretization.nz
        dof = interface.discretization.dof
        x_periodic = interface.discretization.x_periodic
        y_periodic = interface.discretization.y_periodic
        z_periodic = interface.discretization.z_periodic

    state_mtx = numpy.zeros([nx+2, ny+2, nz+2, dof])
    state_mtx[1:nx+1, 1:ny+1, 1:nz+1, :] = create_state_mtx(state, nx, ny, nz, dof)

    # Add extra borders for periodic boundary conditions
    if x_periodic:
        state_mtx[0, :, :, :] = state_mtx[nx, :, :, :]
        state_mtx[nx+1, :, :, :] = state_mtx[1, :, :, :]
    else:
        state_mtx[nx, :, :, 0] = 0

    if y_periodic:
        state_mtx[:, 0, :, :] = state_mtx[:, ny, :, :]
        state_mtx[:, ny+1, :, :] = state_mtx[:, 1, :, :]
    else:
        state_mtx[:, ny, :, 1] = 0

    if z_periodic:
        state_mtx[:, :, 0, :] = state_mtx[:, :, nz, :]
        state_mtx[:, :, nz+1, :] = state_mtx[:, :, 1, :]
    else:
        state_mtx[:, :, nz, 2] = 0

    return state_mtx

def create_state_vec(state_mtx, nx=None, ny=None, nz=None, dof=None, interface=None):
    '''Helper to create a state vector out of an array created with
    create_state_mtx().'''

    if interface:
        nx = interface.discretization.nx
        ny = interface.discretization.ny
        nz = interface.discretization.nz
        dof = interface.discretization.dof

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
    x = start + numpy.arange(-1, nx + 2) * dx
    return numpy.roll(x, -2)

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

def compute_velocity_magnitude(state, interface, axis=2):
    x = interface.discretization.x
    y = interface.discretization.y

    nx = interface.discretization.nx
    ny = interface.discretization.ny

    state_mtx = create_padded_state_mtx(state, interface=interface)
    u = state_mtx[1:, 1:, 1, 0]
    v = state_mtx[1:, 1:, 1, 1]
    w = state_mtx[1:, 1:, 1, 1] * 0

    if axis == 1:
        u = state_mtx[1:, 1, 1:, 0]
        v = state_mtx[1:, 1, 1:, 2]
        w = state_mtx[1:, 1, 1:, 1]
        y = interface.discretization.z
        ny = interface.discretization.nz

    m = numpy.zeros((nx, ny))

    # FIXME: This assumes zero or periodic boundaries
    for i in range(nx):
        for j in range(ny):
            dx0 = x[i] - x[i-1]
            dy0 = y[j] - y[j-1]
            dx1 = x[i+1] - x[i]
            dy1 = y[j+1] - y[j]

            ubar = u[i, j] * dy0 / (dy0 + dy1)
            ubar += u[i, j+1] * dy1 / (dy0 + dy1)

            vbar = v[i, j] * dx0 / (dx0 + dx1)
            vbar += v[i+1, j] * dx1 / (dx0 + dx1)

            wbar = w[i, j] * dx0 / (dx0 + dx1)
            wbar += w[i+1, j] * dx1 / (dx0 + dx1)

            m[i, j] = sqrt(ubar * ubar + vbar * vbar + wbar * wbar)

    return m

def compute_streamfunction(state, interface, axis=2):
    x = interface.discretization.x
    y = interface.discretization.y

    nx = interface.discretization.nx
    ny = interface.discretization.ny

    state_mtx = create_padded_state_mtx(state, interface=interface)
    u = state_mtx[1:, 1:, 1, 0]
    v = state_mtx[1:, 1:, 1, 1]

    if axis == 1:
        u = state_mtx[1:, 1, 1:, 0]
        v = state_mtx[1:, 1, 1:, 2]
        y = interface.discretization.z
        ny = interface.discretization.nz

    psiu = numpy.zeros((nx, ny))
    psiv = numpy.zeros((nx, ny))

    # Integration using the midpoint rule
    for i in range(nx):
        for j in range(ny):
            dx = x[i] - x[i-1]
            dy = y[j] - y[j-1]

            psiu[i, j] = v[i, j] * dx
            if i > 0:
                psiu[i, j] += psiu[i-1, j]

            psiv[i, j] = u[i, j] * dy
            if j > 0:
                psiv[i, j] += psiv[i, j-1]

    return (psiu - psiv) / 2

def compute_volume_averaged_kinetic_energy(state, interface):
    x = interface.discretization.x
    y = interface.discretization.y
    z = interface.discretization.z

    nx = interface.discretization.nx
    ny = interface.discretization.ny
    nz = interface.discretization.nz

    dim = interface.discretization.dim

    if nx <= 1:
        assert x[0] - x[-1] == 1

    if ny <= 1:
        assert y[0] - y[-1] == 1

    if nz <= 1:
        assert z[0] - z[-1] == 1

    state_mtx = create_padded_state_mtx(state, interface=interface)

    w = 0
    Ek = 0
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                dx = x[i] - x[i-1]
                dy = y[j] - y[j-1]
                dz = z[k] - z[k-1]

                u = (state_mtx[i+1, j+1, k+1, 0] + state_mtx[i, j+1, k+1, 0]) / 2
                v = (state_mtx[i+1, j+1, k+1, 1] + state_mtx[i+1, j, k+1, 1]) / 2
                if dim > 2:
                    w = (state_mtx[i+1, j+1, k+1, 2] + state_mtx[i+1, j+1, k, 2]) / 2

                Ek += (u * u + v * v + w * w) * dx * dy * dz

    return Ek / 2

def get_u_value(state, i, j, k, interface):
    '''Get the value of u at a grid point.'''

    state_mtx = create_padded_state_mtx(state, interface=interface)

    y = interface.discretization.y
    dy1 = (y[j] - y[j-1]) / 2
    dy2 = (y[j+1] - y[j]) / 2

    u1 = (state_mtx[i+1, j+1, k+1, 0] * dy1 + state_mtx[i+1, j+2, k+1, 0] * dy2) / (dy1 + dy2)
    u2 = (state_mtx[i+1, j+1, k+2, 0] * dy1 + state_mtx[i+1, j+2, k+2, 0] * dy2) / (dy1 + dy2)

    z = interface.discretization.z
    dz1 = (z[k] - z[k-1]) / 2
    dz2 = (z[k+1] - z[k]) / 2

    return (u1 * dz1 + u2 * dz2) / (dz1 + dz2)

def get_v_value(state, i, j, k, interface):
    '''Get the value of v at a grid point.'''

    state_mtx = create_padded_state_mtx(state, interface=interface)

    x = interface.discretization.x
    dx1 = (x[i] - x[i-1]) / 2
    dx2 = (x[i+1] - x[i]) / 2

    v1 = (state_mtx[i+1, j+1, k+1, 1] * dx1 + state_mtx[i+2, j+1, k+1, 1] * dx2) / (dx1 + dx2)
    v2 = (state_mtx[i+1, j+1, k+2, 1] * dx1 + state_mtx[i+2, j+1, k+2, 1] * dx2) / (dx1 + dx2)

    z = interface.discretization.z
    dz1 = (z[k] - z[k-1]) / 2
    dz2 = (z[k+1] - z[k]) / 2

    return (v1 * dz1 + v2 * dz2) / (dz1 + dz2)

def get_w_value(state, i, j, k, interface):
    '''Get the value of w at a grid point.'''

    state_mtx = create_padded_state_mtx(state, interface=interface)

    x = interface.discretization.x
    dx1 = (x[i] - x[i-1]) / 2
    dx2 = (x[i+1] - x[i]) / 2

    w1 = (state_mtx[i+1, j+1, k+1, 2] * dx1 + state_mtx[i+2, j+1, k+1, 2] * dx2) / (dx1 + dx2)
    w2 = (state_mtx[i+1, j+2, k+1, 2] * dx1 + state_mtx[i+2, j+2, k+1, 2] * dx2) / (dx1 + dx2)

    y = interface.discretization.y
    dy1 = (y[j] - y[j-1]) / 2
    dy2 = (y[j+1] - y[j]) / 2

    return (w1 * dy1 + w2 * dy2) / (dy1 + dy2)
