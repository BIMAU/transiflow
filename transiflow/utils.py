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
    for k, j, i, d in numpy.ndindex(nz, ny, nx, dof):
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
    for k, j, i, d in numpy.ndindex(nz, ny, nx, dof):
        state[row] = state_mtx[i, j, k, d]
        row += 1
    return state

def create_uniform_coordinate_vector(start, end, nx):
    dx = (end - start) / nx
    x = start + numpy.arange(-1, nx + 2) * dx
    return numpy.roll(x, -2)

def create_stretched_coordinate_vector(start, end, nx, sigma):
    x = create_uniform_coordinate_vector(0, 1, nx)
    x = 0.5 * (1 + numpy.tanh(2 * sigma * (x - 0.5)) / numpy.tanh(sigma))
    x = start + x * (end - start)

    # Make cells inside and outside of the boundary the same size
    # to make sure the boundary conditions are applied at the boundary
    dx = x[0] - x[-1]
    if start == 0:
        x[-2] = x[-1] - dx
    if end == 1:
        x[-3] = x[-4] + dx
    return x

def create_stretched_coordinate_vector2(start, end, nx, sigma):
    x = create_uniform_coordinate_vector(0, 1, nx)
    x = x - sigma * numpy.sin(2 * numpy.pi * x)
    x = start + x * (end - start)

    # Make cells inside and outside of the boundary the same size
    # to make sure the boundary conditions are applied at the boundary
    dx = x[0] - x[-1]
    if start == 0:
        x[-2] = x[-1] - dx
    if end == 1:
        x[-3] = x[-4] + dx
    return x

def compute_velocity_magnitude(state, interface, axis=2, position=None):
    nx = interface.discretization.nx
    ny = interface.discretization.ny
    nz = interface.discretization.nz

    x = interface.discretization.x
    y = interface.discretization.y
    z = interface.discretization.z

    state_mtx = create_padded_state_mtx(state, interface=interface)

    # FIXME: This assumes zero or periodic boundaries
    if axis == 0:
        m = numpy.zeros((ny, nz))

        center = nx // 2 - 1
        if position:
            center = numpy.argmin(numpy.abs(x - position))

        print('Using center: %e at %d' % (x[center], center))

        for j, k in numpy.ndindex(ny, nz):
            u = get_u_value(state_mtx, center, j, k, interface)
            v = get_v_value(state_mtx, center, j, k, interface)
            w = get_w_value(state_mtx, center, j, k, interface)
            m[j, k] = sqrt(u * u + v * v + w * w)

        return m

    if axis == 1:
        m = numpy.zeros((nx, nz))

        center = ny // 2 - 1
        if position:
            center = numpy.argmin(numpy.abs(y - position))

        print('Using center: %e at %d' % (y[center], center))

        for i, k in numpy.ndindex(nx, nz):
            u = get_u_value(state_mtx, i, center, k, interface)
            v = get_v_value(state_mtx, i, center, k, interface)
            w = get_w_value(state_mtx, i, center, k, interface)
            m[i, k] = sqrt(u * u + v * v + w * w)

        return m

    m = numpy.zeros((nx, ny))

    center = nz // 2 - 1
    if position:
        center = numpy.argmin(numpy.abs(z - position))

    print('Using center: %e at %d' % (z[center], center))

    for i, j in numpy.ndindex(nx, ny):
        u = get_u_value(state_mtx, i, j, center, interface)
        v = get_v_value(state_mtx, i, j, center, interface)

        w = 0
        if interface.discretization.dim > 2:
            w = get_w_value(state_mtx, i, j, center, interface)

        m[i, j] = sqrt(u * u + v * v + w * w)

    return m

def compute_streamfunction(state, interface, axis=2):
    x = interface.discretization.x
    y = interface.discretization.y

    nx = interface.discretization.nx
    ny = interface.discretization.ny

    state_mtx = create_padded_state_mtx(state, interface=interface)

    center = interface.discretization.nz // 2 + 1
    u = state_mtx[1:, 1:, center, 0]
    v = state_mtx[1:, 1:, center, 1]

    if axis == 1:
        center = interface.discretization.ny // 2 + 1
        u = state_mtx[1:, center, 1:, 0]
        v = state_mtx[1:, center, 1:, 2]
        y = interface.discretization.z
        ny = interface.discretization.nz

    psiu = numpy.zeros((nx, ny))
    psiv = numpy.zeros((nx, ny))

    # Integration using the midpoint rule
    for i, j in numpy.ndindex(nx, ny):
        dx = x[i] - x[i-1]
        dy = y[j] - y[j-1]

        psiu[i, j] = v[i, j] * dx
        if i > 0:
            psiu[i, j] += psiu[i-1, j]

        psiv[i, j] = u[i, j] * dy
        if j > 0:
            psiv[i, j] += psiv[i, j-1]

    return (psiu - psiv) / 2

def compute_vorticity(state, interface, axis=2):
    x = interface.discretization.x
    y = interface.discretization.y

    nx = interface.discretization.nx
    ny = interface.discretization.ny

    state_mtx = create_padded_state_mtx(state, interface=interface)

    center = interface.discretization.nz // 2 + 1
    u = state_mtx[1:, 1:, center, 0]
    v = state_mtx[1:, 1:, center, 1]

    assert axis == 2

    zeta = numpy.zeros((nx, ny))

    # Integration using the midpoint rule
    for i, j in numpy.ndindex(nx, ny):
        dx = (x[i+1] - x[i-1]) / 2
        dy = (y[j+1] - y[j-1]) / 2

        if i < nx - 1:
            zeta[i, j] += v[i+1, j] / dx
            zeta[i, j] -= v[i, j] / dx

        if j < ny - 1:
            zeta[i, j] += u[i, j+1] / dy
            zeta[i, j] -= u[i, j] / dy

    return zeta

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
    for i, j, k in numpy.ndindex(nx, ny, nz):
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

    if len(state.shape) < 4:
        state_mtx = create_padded_state_mtx(state, interface=interface)
    else:
        state_mtx = state

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

    if len(state.shape) < 4:
        state_mtx = create_padded_state_mtx(state, interface=interface)
    else:
        state_mtx = state

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

    if len(state.shape) < 4:
        state_mtx = create_padded_state_mtx(state, interface=interface)
    else:
        state_mtx = state

    x = interface.discretization.x
    dx1 = (x[i] - x[i-1]) / 2
    dx2 = (x[i+1] - x[i]) / 2

    w1 = (state_mtx[i+1, j+1, k+1, 2] * dx1 + state_mtx[i+2, j+1, k+1, 2] * dx2) / (dx1 + dx2)
    w2 = (state_mtx[i+1, j+2, k+1, 2] * dx1 + state_mtx[i+2, j+2, k+1, 2] * dx2) / (dx1 + dx2)

    y = interface.discretization.y
    dy1 = (y[j] - y[j-1]) / 2
    dy2 = (y[j+1] - y[j]) / 2

    return (w1 * dy1 + w2 * dy2) / (dy1 + dy2)
