import numpy

class CrsMatrix:
    def __init__(self, coA=None, jcoA=None, begA=None):
        self.coA = coA
        self.jcoA = jcoA
        self.begA = begA

def create_uniform_coordinate_vector(nx):
    dx = 1 / nx
    return numpy.roll(numpy.arange(-dx, 1+2*dx, dx), -2)

def linear_part(nx, ny, nz, dof, Re, Ra=0, Pr=0):
    x = create_uniform_coordinate_vector(nx)
    y = create_uniform_coordinate_vector(ny)
    z = create_uniform_coordinate_vector(nz)

    derivatives = Derivatives(nx, ny, nz, dof)

    atom = 1 / Re * (derivatives.u_xx(x, y, z) + derivatives.u_yy(x, y, z) + derivatives.u_zz(x, y, z) \
                  +  derivatives.v_xx(x, y, z) + derivatives.v_yy(x, y, z) + derivatives.v_zz(x, y, z) \
                  +  derivatives.w_xx(x, y, z) + derivatives.w_yy(x, y, z) + derivatives.w_zz(x, y, z)) \
        - (derivatives.p_x(x, y, z) + derivatives.p_y(x, y, z) + derivatives.p_z(x, y, z)) \
        + derivatives.div(x, y, z)

    if Ra:
        atom += Ra * derivatives.forward_average_z(x, y, z)

    if Pr:
        atom += 1 / Pr * (derivatives.T_xx(x, y, z) + derivatives.T_yy(x, y, z) + derivatives.T_zz(x, y, z))

    return atom

def boundaries(atom, nx, ny, nz, dof):
    boundary_conditions = BoundaryConditions(nx, ny, nz, dof)

    boundary_conditions.dirichlet_east(atom)
    boundary_conditions.dirichlet_west(atom)
    frc = boundary_conditions.dirichlet_north(atom)
    boundary_conditions.dirichlet_south(atom)
    frc += boundary_conditions.dirichlet_top(atom)
    boundary_conditions.dirichlet_bottom(atom)
    return frc

def create_state_mtx(state, nx, ny, nz, dof):
    state_mtx = numpy.zeros([nx, ny, nz, dof])
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                for d in range(dof):
                    state_mtx[i, j, k, d] = state[d + i * dof + j * dof * nx + k * dof * nx * ny]
    return state_mtx

def convection(state, nx, ny, nz, dof):
    x = create_uniform_coordinate_vector(nx)
    y = create_uniform_coordinate_vector(ny)
    z = create_uniform_coordinate_vector(nz)

    state_mtx = create_state_mtx(state, nx, ny, nz, dof)

    derivatives = Derivatives(nx, ny, nz, dof)
    return derivatives.convection(state_mtx, x, y, z)

def assemble(atom, nx, ny, nz, dof):
    ''' Assemble the Jacobian. Optimized version of

    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                for d1 in range(dof):
                    for z in range(3):
                        for y in range(3):
                            for x in range(3):
                                for d2 in range(dof):
                                    if abs(atom[i, j, k, d1, d2, x, y, z]) > 1e-14:
                                       jcoA[idx] = row + (x-1) * dof + (y-1) * nx * dof + (z-1) * nx * ny * dof + d2 - d1
                                       coA[idx] = atom[i, j, k, d1, d2, x, y, z]
                                       idx += 1
                    row += 1
                    begA[row] = idx
    '''

    row = 0
    idx = 0
    n = nx * ny * nz * dof
    coA = numpy.zeros(27*n)
    jcoA = numpy.zeros(27*n, dtype=int)
    begA = numpy.zeros(n+1, dtype=int)

    # Check where values are nonzero in the atoms
    configs = []
    for z in range(3):
        for y in range(3):
            for x in range(3):
                for d2 in range(dof):
                    if numpy.any(atom[:, :, :, :, d2, x, y, z]):
                        configs.append([d2, x, y, z])

    # Iterate only over configurations with values in there
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                for d1 in range(dof):
                    for config in configs:
                        if abs(atom[i, j, k, d1, config[0], config[1], config[2], config[3]]) > 1e-14:
                            jcoA[idx] = row + (config[1]-1) * dof + (config[2]-1) * nx * dof + (config[3]-1) * nx * ny * dof + config[0] - d1
                            coA[idx] = atom[i, j, k, d1, config[0], config[1], config[2], config[3]]
                            idx += 1
                    row += 1
                    begA[row] = idx
    return CrsMatrix(coA, jcoA, begA)

def rhs(state, atom, nx, ny, nz, dof):
    ''' Assemble the right-hand side. Optimized version of

    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                for d1 in range(dof):
                    for z in range(3):
                        for y in range(3):
                            for x in range(3):
                                for d2 in range(dof):
                                    if abs(atom[i, j, k, d1, d2, x, y, z]) > 1e-14:
                                        offset = row + (x-1) * dof + (y-1) * nx * dof + (z-1) * nx * ny * dof + d2 - d1
                                        out[row] -= atom[i, j, k, d1, d2, x, y, z] * state[offset]
                    row += 1
    '''

    row = 0
    n = nx * ny * nz * dof

    # Put the state in shifted matrix form
    state_mtx = numpy.zeros([nx+2, ny+2, nz+2, dof])
    state_mtx[1:nx+1, 1:ny+1, 1:nz+1, :] = create_state_mtx(state, nx, ny, nz, dof)

    # Add up all contributions without iterating over the domain
    out_mtx = numpy.zeros([nx, ny, nz, dof])
    for k in range(3):
        for j in range(3):
            for i in range(3):
                for d1 in range(dof):
                    for d2 in range(dof):
                        out_mtx[:, :, :, d1] -= atom[:, :, :, d1, d2, i, j, k] * state_mtx[i:(i+nx), j:(j+ny), k:(k+nz), d2]

    # Put the output in vector form
    out = numpy.zeros(n)
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                for d1 in range(dof):
                    out[row] = out_mtx[i, j, k, d1]
                    row += 1
    return out

class BoundaryConditions:

    def __init__(self, nx, ny, nz, dof):
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.dof = dof

    def dirichlet_east(self, atom):
        # At the boundary u[i] = 0, v[i] + v[i+1] = 2*V similar for w. So v[i+1] = -v[i]+2*V.
        atom[self.nx-1, :, :, :, :, 1, :, :] -= atom[self.nx-1, :, :, :, :, 2, :, :]
        atom[self.nx-1, :, :, :, 0, 1, :, :] = 0
        atom[self.nx-1, :, :, 0, :, :, :, :] = 0
        atom[self.nx-1, :, :, :, :, 2, :, :] = 0
        atom[self.nx-1, :, :, 0, 0, 1, 1, 1] = -1
        # TODO: Do we need this?
        atom[self.nx-2, :, :, 0, 0, 2, :, :] = 0

    def dirichlet_west(self, atom):
        # At the boundary u[i-1] = 0, v[i-1] + v[i] = 0 similar for w. So v[i-1] = -v[i].
        atom[0, :, :, :, 0, 0, :, :] = 0
        atom[0, :, :, :, :, 1, :, :] -= atom[0, :, :, :, :, 0, :, :]
        atom[0, :, :, :, :, 0, :, :] = 0

    def dirichlet_north(self, atom):
        # At the boundary v[i] = 0, u[i] + u[i+1] = 2*U similar for w. So u[i+1] = -u[i]+2*U.
        atom[:, self.ny-1, :, :, :, :, 1, :] -= atom[:, self.ny-1, :, :, :, :, 2, :]
        atom[:, self.ny-1, :, :, 1, :, 1, :] = 0
        frc = self.ldc_forcing_north(atom)
        atom[:, self.ny-1, :, 1, :, :, :, :] = 0
        atom[:, self.ny-1, :, :, :, :, 2, :] = 0
        atom[:, self.ny-1, :, 1, 1, 1, 1, 1] = -1
        # TODO: Do we need this?
        atom[:, self.ny-2, :, 1, 1, :, 2, :] = 0
        return frc

    def dirichlet_south(self, atom):
        # At the boundary v[i-1] = 0, u[i-1] + u[i] = 0 similar for w. So u[i-1] = -u[i].
        atom[:, 0, :, :, 1, :, 0, :] = 0
        atom[:, 0, :, :, :, :, 1, :] -= atom[:, 0, :, :, :, :, 0, :]
        atom[:, 0, :, :, :, :, 0, :] = 0

    def dirichlet_top(self, atom):
        # At the boundary w[i] = 0, u[i] + u[i+1] = 2*U similar for v. So u[i+1] = -u[i]+2*U.
        atom[:, :, self.nz-1, :, :, :, :, 1] -= atom[:, :, self.nz-1, :, :, :, :, 2]
        atom[:, :, self.nz-1, :, 2, :, :, 1] = 0
        frc = self.ldc_forcing_top(atom)
        atom[:, :, self.nz-1, 2, :, :, :, :] = 0
        atom[:, :, self.nz-1, :, :, :, :, 2] = 0
        atom[:, :, self.nz-1, 2, 2, 1, 1, 1] = -1
        # TODO: Do we need this?
        atom[:, :, self.nz-2, 2, 2, :, :, 2] = 0
        return frc

    def dirichlet_bottom(self, atom):
        # At the boundary w[i-1] = 0, u[i-1] + u[i] = 0 similar for v. So u[i-1] = -u[i].
        atom[:, :, 0, :, 2, :, :, 0] = 0
        atom[:, :, 0, :, :, :, :, 1] -= atom[:, :, 0, :, :, :, :, 0]
        atom[:, :, 0, :, :, :, :, 0] = 0

    def ldc_forcing_top(self, atom):
        n = self.nx * self.ny * self.nz * self.dof
        out = numpy.zeros(n)

        if self.nz <= 1:
            return out

        k = self.nz-1
        z = 2
        for j in range(self.ny):
            for i in range(self.nx):
                for y in range(3):
                    for x in range(3):
                        offset = i * self.dof + j * self.nx * self.dof + k * self.nx * self.ny * self.dof + 1
                        out[offset] += 2 * atom[i, j, k, 1, 0, x, y, z]
                        offset = i * self.dof + j * self.nx * self.dof + k * self.nx * self.ny * self.dof
                        out[offset] += 2 * atom[i, j, k, 0, 0, x, y, z]
        return out

    def ldc_forcing_north(self, atom):
        n = self.nx * self.ny * self.nz * self.dof
        out = numpy.zeros(n)

        if self.nz > 1:
            return out

        j = self.ny-1
        y = 2
        for k in range(self.nz):
            for i in range(self.nx):
                for z in range(3):
                    for x in range(3):
                        offset = i * self.dof + j * self.nx * self.dof + k * self.nx * self.ny * self.dof
                        out[offset] += 2 * atom[i, j, k, 0, 0, x, y, z]
                        offset = i * self.dof + j * self.nx * self.dof + k * self.nx * self.ny * self.dof + 2
                        out[offset] += 2 * atom[i, j, k, 2, 0, x, y, z]
        return out

class Derivatives:

    def __init__(self, nx, ny, nz, dof):
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.dof = dof

    @staticmethod
    def _u_xx(atom, i, j, k, x, y, z):
        # distance between u[i] and u[i-1]
        dx = x[i] - x[i-1]
        # distance between u[i+1] and u[i]
        dxp1 = x[i+1] - x[i]
        # volume size in the y direction
        dy = y[j] - y[j-1]
        # volume size in the z direction
        dz = z[k] - z[k-1]

        # second order finite difference
        atom[0] = 1 / dx * dy * dz
        atom[2] = 1 / dxp1 * dy * dz
        atom[1] = -atom[0] - atom[2]

    def u_xx(self, x, y, z):
        atom = numpy.zeros([self.nx, self.ny, self.nz, self.dof, self.dof, 3, 3, 3])
        for i in range(self.nx):
            for j in range(self.ny):
                for k in range(self.nz):
                    Derivatives._u_xx(atom[i, j, k, 0, 0, :, 1, 1], i, j, k, x, y, z)
        return atom

    def v_yy(self, x, y, z):
        atom = numpy.zeros([self.nx, self.ny, self.nz, self.dof, self.dof, 3, 3, 3])
        for i in range(self.nx):
            for j in range(self.ny):
                for k in range(self.nz):
                    Derivatives._u_xx(atom[i, j, k, 1, 1, 1, :, 1], j, i, k, y, x, z)
        return atom

    def w_zz(self, x, y, z):
        atom = numpy.zeros([self.nx, self.ny, self.nz, self.dof, self.dof, 3, 3, 3])
        for i in range(self.nx):
            for j in range(self.ny):
                for k in range(self.nz):
                    Derivatives._u_xx(atom[i, j, k, 2, 2, 1, 1, :], k, j, i, z, y, x)
        return atom

    @staticmethod
    def _u_yy(atom, i, j, k, x, y, z):
        # distance between u[j] and u[j-1]
        dy = (y[j] - y[j-2]) / 2
        # distance between u[j+1] and u[j]
        dyp1 = (y[j+1] - y[j-1]) / 2
        # volume size in the x direction
        dx = (x[i+1] - x[i-1]) / 2
        # volume size in the z direction
        dz = z[k] - z[k-1]

        # second order finite difference
        atom[0] = 1 / dy * dx * dz
        atom[2] = 1 / dyp1 * dx * dz
        atom[1] = -atom[0] - atom[2]

    def u_yy(self, x, y, z):
        atom = numpy.zeros([self.nx, self.ny, self.nz, self.dof, self.dof, 3, 3, 3])
        for i in range(self.nx):
            for j in range(self.ny):
                for k in range(self.nz):
                    Derivatives._u_yy(atom[i, j, k, 0, 0, 1, :, 1], i, j, k, x, y, z)
        return atom

    def v_xx(self, x, y, z):
        atom = numpy.zeros([self.nx, self.ny, self.nz, self.dof, self.dof, 3, 3, 3])
        for i in range(self.nx):
            for j in range(self.ny):
                for k in range(self.nz):
                    Derivatives._u_yy(atom[i, j, k, 1, 1, :, 1, 1], j, i, k, y, x, z)
        return atom

    def w_yy(self, x, y, z):
        atom = numpy.zeros([self.nx, self.ny, self.nz, self.dof, self.dof, 3, 3, 3])
        for i in range(self.nx):
            for j in range(self.ny):
                for k in range(self.nz):
                    Derivatives._u_yy(atom[i, j, k, 2, 2, 1, :, 1], k, j, i, z, y, x)
        return atom

    @staticmethod
    def _u_zz(atom, i, j, k, x, y, z):
        # distance between u[k] and u[k-1]
        dz = (z[k] - z[k-2]) / 2
        # distance between u[k+1] and u[k]
        dzp1 = (z[k+1] - z[k-1]) / 2
        # volume size in the x direction
        dx = (x[i+1] - x[i-1]) / 2
        # volume size in the y direction
        dy = y[j] - y[j-1]

        # second order finite difference
        atom[0] = 1 / dz * dx * dy
        atom[2] = 1 / dzp1 * dx * dy
        atom[1] = -atom[0] - atom[2]

    def u_zz(self, x, y, z):
        atom = numpy.zeros([self.nx, self.ny, self.nz, self.dof, self.dof, 3, 3, 3])
        for i in range(self.nx):
            for j in range(self.ny):
                for k in range(self.nz):
                    Derivatives._u_zz(atom[i, j, k, 0, 0, 1, 1, :], i, j, k, x, y, z)
        return atom

    def v_zz(self, x, y, z):
        atom = numpy.zeros([self.nx, self.ny, self.nz, self.dof, self.dof, 3, 3, 3])
        for i in range(self.nx):
            for j in range(self.ny):
                for k in range(self.nz):
                    Derivatives._u_zz(atom[i, j, k, 1, 1, 1, 1, :], j, i, k, y, x, z)
        return atom

    def w_xx(self, x, y, z):
        atom = numpy.zeros([self.nx, self.ny, self.nz, self.dof, self.dof, 3, 3, 3])
        for i in range(self.nx):
            for j in range(self.ny):
                for k in range(self.nz):
                    Derivatives._u_zz(atom[i, j, k, 2, 2, :, 1, 1], k, j, i, z, y, x)
        return atom

    @staticmethod
    def _T_xx(atom, i, j, k, x, y, z):
        # distance between u[i] and u[i-1]
        dx = (x[i] - x[i-2]) / 2
        # distance between u[i+1] and u[i]
        dxp1 = (x[i+1] - x[i-1]) / 2
        # volume size in the y direction
        dy = y[j] - y[j-1]
        # volume size in the z direction
        dz = z[k] - z[k-1]

        # second order finite difference
        atom[0] = 1 / dx * dy * dz
        atom[2] = 1 / dxp1 * dy * dz
        atom[1] = -atom[0] - atom[2]

    def T_xx(self, x, y, z):
        atom = numpy.zeros([self.nx, self.ny, self.nz, self.dof, self.dof, 3, 3, 3])
        for i in range(self.nx):
            for j in range(self.ny):
                for k in range(self.nz):
                    Derivatives._T_xx(atom[i, j, k, 4, 4, :, 1, 1], i, j, k, x, y, z)
        return atom

    def T_yy(self, x, y, z):
        atom = numpy.zeros([self.nx, self.ny, self.nz, self.dof, self.dof, 3, 3, 3])
        for i in range(self.nx):
            for j in range(self.ny):
                for k in range(self.nz):
                    Derivatives._T_xx(atom[i, j, k, 4, 4, 1, :, 1], j, i, k, y, x, z)
        return atom

    def T_zz(self, x, y, z):
        atom = numpy.zeros([self.nx, self.ny, self.nz, self.dof, self.dof, 3, 3, 3])
        for i in range(self.nx):
            for j in range(self.ny):
                for k in range(self.nz):
                    Derivatives._T_xx(atom[i, j, k, 4, 4, 1, 1, :], k, j, i, z, y, x)
        return atom

    @staticmethod
    def _forward_u_x(atom, i, j, k, x, y, z):
        # volume size in the y direction
        dy = y[j] - y[j-1]
        # volume size in the z direction
        dz = z[k] - z[k-1]

        # forward difference
        atom[2] = dy * dz
        atom[1] = -atom[2]

    def p_x(self, x, y, z):
        atom = numpy.zeros([self.nx, self.ny, self.nz, self.dof, self.dof, 3, 3, 3])
        for i in range(self.nx):
            for j in range(self.ny):
                for k in range(self.nz):
                    Derivatives._forward_u_x(atom[i, j, k, 0, 3, :, 1, 1], i, j, k, x, y, z)
        return atom

    def p_y(self, x, y, z):
        atom = numpy.zeros([self.nx, self.ny, self.nz, self.dof, self.dof, 3, 3, 3])
        for i in range(self.nx):
            for j in range(self.ny):
                for k in range(self.nz):
                    Derivatives._forward_u_x(atom[i, j, k, 1, 3, 1, :, 1], j, i, k, y, x, z)
        return atom

    def p_z(self, x, y, z):
        atom = numpy.zeros([self.nx, self.ny, self.nz, self.dof, self.dof, 3, 3, 3])
        for i in range(self.nx):
            for j in range(self.ny):
                for k in range(self.nz):
                    Derivatives._forward_u_x(atom[i, j, k, 2, 3, 1, 1, :], k, j, i, z, y, x)
        return atom

    @staticmethod
    def _backward_u_x(atom, i, j, k, x, y, z):
        # volume size in the y direction
        dy = y[j] - y[j-1]
        # volume size in the z direction
        dz = z[k] - z[k-1]

        # backward difference
        atom[1] = dy * dz
        atom[0] = -atom[1]

    def u_x(self, x, y, z):
        atom = numpy.zeros([self.nx, self.ny, self.nz, self.dof, self.dof, 3, 3, 3])
        for i in range(self.nx):
            for j in range(self.ny):
                for k in range(self.nz):
                    Derivatives._backward_u_x(atom[i, j, k, 3, 0, :, 1, 1], i, j, k, x, y, z)
        return atom

    def v_y(self, x, y, z):
        atom = numpy.zeros([self.nx, self.ny, self.nz, self.dof, self.dof, 3, 3, 3])
        for i in range(self.nx):
            for j in range(self.ny):
                for k in range(self.nz):
                    Derivatives._backward_u_x(atom[i, j, k, 3, 1, 1, :, 1], j, i, k, y, x, z)
        return atom

    def w_z(self, x, y, z):
        atom = numpy.zeros([self.nx, self.ny, self.nz, self.dof, self.dof, 3, 3, 3])
        for i in range(self.nx):
            for j in range(self.ny):
                for k in range(self.nz):
                    Derivatives._backward_u_x(atom[i, j, k, 3, 2, 1, 1, :], k, j, i, z, y, x)
        return atom

    @staticmethod
    def _backward_u_y(atom, i, j, k, x, y, z):
        # volume size in the x direction
        dx = (x[i+1] - x[i-1]) / 2
        # volume size in the z direction
        dz = z[k] - z[k-1]

        # backward difference
        atom[1] = dx * dz
        atom[0] = -atom[1]

    @staticmethod
    def _backward_u_z(atom, i, j, k, x, y, z):
        # volume size in the x direction
        dx = (x[i+1] - x[i-1]) / 2
        # volume size in the y direction
        dy = y[k] - y[k-1]

        # backward difference
        atom[1] = dx * dy
        atom[0] = -atom[1]

    def div(self, x, y, z):
        return self.u_x(x, y, z) + self.v_y(x, y, z) + self.w_z(x, y, z)

    @staticmethod
    def _convection_v_u_unoptimized(atomJ, atomF, MxV, MyU, bil, varV, varU, ny, j):
        ''' This method is left here for documentation purposes and
        can be used when iterating over the entire grid'''

        bw = 1 # forward average, backward difference
        if varU == varV:
            bw = 0 # backward average, forward difference

        for d1 in range(2):
            j2 = max(min(j + d1 - bw, ny - 1), 0)

            coef1 = MxV[j2] * bil[j, 3 + varV, varU, d1]
            if abs(coef1) > 1e-15:
                for d2 in range(2):
                    coef2 = bil[j2, varU, varV, d2]
                    if abs(coef2) > 1e-15:
                        idx = [1, 1, 1]
                        idx[varV] += d1 - bw
                        idx[varV] += d2 - 1 + bw
                        atomF[idx[0], idx[1], idx[2]] -= coef1 * coef2

            coef1 = MyU[j2] * bil[j, 3 + varV, varU, d1]
            if abs(coef1) > 1e-15:
                for d2 in range(2):
                    coef2 = bil[j2, varV, varU, d2]
                    if abs(coef2) > 1e-15:
                        idx = [1, 1, 1]
                        idx[varV] += d1 - bw
                        idx[varU] += d2 - 1 + bw
                        atomJ[idx[0], idx[1], idx[2]] -= coef1 * coef2

    @staticmethod
    def _convection_u_v(atomJ, atomF, averages, bil, varU, varV, nx, i):
        bw = 1 # forward average, backward difference
        if varU == varV:
            bw = 0 # backward average, forward difference

        for d1 in range(2):
            i2 = max(min(i + d1 - bw, nx - 1), 0)

            coef1 = averages[i2, :, :, varU, varV] * bil[i, :, :, 3 + varU, varV, d1]
            if numpy.any(coef1):
                for d2 in range(2):
                    coef2 = bil[i2, :, :, varV, varU, d2]
                    if numpy.any(coef2):
                        idx = [1, 1, 1]
                        idx[varU] += d1 - bw
                        idx[varU] += d2 - 1 + bw
                        atomF[i, :, :, varV, varV, idx[0], idx[1], idx[2]] -= coef1 * coef2

            coef1 = averages[i2, :, :, varV, varU] * bil[i, :, :, 3 + varU, varV, d1]
            if numpy.any(coef1):
                for d2 in range(2):
                    coef2 = bil[i2, :, :, varU, varV, d2]
                    if numpy.any(coef2):
                        idx = [1, 1, 1]
                        idx[varU] += d1 - bw
                        idx[varV] += d2 - 1 + bw
                        atomJ[i, :, :, varV, varU, idx[0], idx[1], idx[2]] -= coef1 * coef2

    @staticmethod
    def _convection_v_u(atomJ, atomF, averages, bil, varV, varU, ny, j):
        bw = 1 # forward average, backward difference
        if varV == varU:
            bw = 0 # backward average, forward difference

        for d1 in range(2):
            j2 = max(min(j + d1 - bw, ny - 1), 0)

            coef1 = averages[:, j2, :, varV, varU] * bil[:, j, :, 3 + varV, varU, d1]
            if numpy.any(coef1):
                for d2 in range(2):
                    coef2 = bil[:, j2, :, varU, varV, d2]
                    if numpy.any(coef2):
                        idx = [1, 1, 1]
                        idx[varV] += d1 - bw
                        idx[varV] += d2 - 1 + bw
                        atomF[:, j, :, varU, varU, idx[0], idx[1], idx[2]] -= coef1 * coef2

            coef1 = averages[:, j2, :, varU, varV] * bil[:, j, :, 3 + varV, varU, d1]
            if numpy.any(coef1):
                for d2 in range(2):
                    coef2 = bil[:, j2, :, varV, varU, d2]
                    if numpy.any(coef2):
                        idx = [1, 1, 1]
                        idx[varV] += d1 - bw
                        idx[varU] += d2 - 1 + bw
                        atomJ[:, j, :, varU, varV, idx[0], idx[1], idx[2]] -= coef1 * coef2

    @staticmethod
    def _convection_w_u(atomJ, atomF, averages, bil, varW, varU, nz, k):
        bw = 1 # forward average, backward difference
        if varW == varU:
            bw = 0 # backward average, forward difference

        for d1 in range(2):
            k2 = max(min(k + d1 - bw, nz - 1), 0)

            coef1 = averages[:, :, k2, varW, varU] * bil[:, :, k, 3 + varW, varU, d1]
            if numpy.any(coef1):
                for d2 in range(2):
                    coef2 = bil[:, :, k2, varU, varW, d2]
                    if numpy.any(coef2):
                        idx = [1, 1, 1]
                        idx[varW] += d1 - bw
                        idx[varW] += d2 - 1 + bw
                        atomF[:, :, k, varU, varU, idx[0], idx[1], idx[2]] -= coef1 * coef2

            coef1 = averages[:, :, k2, varU, varW] * bil[:, :, k, 3 + varW, varU, d1]
            if numpy.any(coef1):
                for d2 in range(2):
                    coef2 = bil[:, :, k2, varW, varU, d2]
                    if numpy.any(coef2):
                        idx = [1, 1, 1]
                        idx[varW] += d1 - bw
                        idx[varU] += d2 - 1 + bw
                        atomJ[:, :, k, varU, varW, idx[0], idx[1], idx[2]] -= coef1 * coef2

    def convection_u_u(self, atomJ, atomF, averages, bil):
        for i in range(self.nx):
            Derivatives._convection_u_v(atomJ, atomF, averages, bil, 0, 0, self.nx, i)

    def convection_v_u(self, atomJ, atomF, averages, bil):
        for j in range(self.ny):
            Derivatives._convection_v_u(atomJ, atomF, averages, bil, 1, 0, self.ny, j)

    def convection_w_u(self, atomJ, atomF, averages, bil):
        for k in range(self.nz):
            Derivatives._convection_w_u(atomJ, atomF, averages, bil, 2, 0, self.nz, k)

    def convection_u_v(self, atomJ, atomF, averages, bil):
        for i in range(self.nx):
            Derivatives._convection_u_v(atomJ, atomF, averages, bil, 0, 1, self.nx, i)

    def convection_v_v(self, atomJ, atomF, averages, bil):
        for j in range(self.ny):
            Derivatives._convection_v_u(atomJ, atomF, averages, bil, 1, 1, self.ny, j)

    def convection_w_v(self, atomJ, atomF, averages, bil):
        for k in range(self.nz):
            Derivatives._convection_w_u(atomJ, atomF, averages, bil, 2, 1, self.nz, k)

    def convection_u_w(self, atomJ, atomF, averages, bil):
        for i in range(self.nx):
            Derivatives._convection_u_v(atomJ, atomF, averages, bil, 0, 2, self.nx, i)

    def convection_v_w(self, atomJ, atomF, averages, bil):
        for j in range(self.ny):
            Derivatives._convection_v_u(atomJ, atomF, averages, bil, 1, 2, self.ny, j)

    def convection_w_w(self, atomJ, atomF, averages, bil):
        for k in range(self.nz):
            Derivatives._convection_w_u(atomJ, atomF, averages, bil, 2, 2, self.nz, k)

    def convection(self, state, x, y, z):
        bil = numpy.zeros([self.nx, self.ny, self.nz, 6, 3, 2])

        convective_term = ConvectiveTerm(self.nx, self.ny, self.nz)

        convective_term.averages(bil)

        convective_term.u_x(bil[:, :, :, 3, :, :], x, y, z) # tMxUMxU
        convective_term.u_y(bil[:, :, :, 4, :, :], x, y, z) # tMxVMyU
        convective_term.u_z(bil[:, :, :, 5, :, :], x, y, z) # tMxWMzU
        convective_term.v_x(bil[:, :, :, 3, :, :], x, y, z) # tMyUMxV
        convective_term.v_y(bil[:, :, :, 4, :, :], x, y, z) # tMyVMyV
        convective_term.v_z(bil[:, :, :, 5, :, :], x, y, z) # tMyWMzV
        convective_term.w_x(bil[:, :, :, 3, :, :], x, y, z) # tMzUMxW
        convective_term.w_y(bil[:, :, :, 4, :, :], x, y, z) # tMzVMyW
        convective_term.w_z(bil[:, :, :, 5, :, :], x, y, z) # tMzWMzW

        convective_term.dirichlet_east(bil)
        convective_term.dirichlet_west(bil)
        convective_term.dirichlet_north(bil)
        convective_term.dirichlet_south(bil)
        convective_term.dirichlet_top(bil)
        convective_term.dirichlet_bottom(bil)

        averages = numpy.zeros([self.nx, self.ny, self.nz, 3, 3])

        convective_term.MxU(averages, bil, state)
        convective_term.MxV(averages, bil, state)
        convective_term.MxW(averages, bil, state)
        convective_term.MyU(averages, bil, state)
        convective_term.MyV(averages, bil, state)
        convective_term.MyW(averages, bil, state)
        convective_term.MzU(averages, bil, state)
        convective_term.MzV(averages, bil, state)
        convective_term.MzW(averages, bil, state)

        atomJ = numpy.zeros([self.nx, self.ny, self.nz, self.dof, self.dof, 3, 3, 3])
        atomF = numpy.zeros([self.nx, self.ny, self.nz, self.dof, self.dof, 3, 3, 3])

        self.convection_u_u(atomJ, atomF, averages, bil)
        self.convection_u_v(atomJ, atomF, averages, bil)
        self.convection_u_w(atomJ, atomF, averages, bil)
        self.convection_v_u(atomJ, atomF, averages, bil)
        self.convection_v_v(atomJ, atomF, averages, bil)
        self.convection_v_w(atomJ, atomF, averages, bil)
        self.convection_w_u(atomJ, atomF, averages, bil)
        self.convection_w_v(atomJ, atomF, averages, bil)
        self.convection_w_w(atomJ, atomF, averages, bil)

        atomJ += atomF

        return (atomJ, atomF)

    @staticmethod
    def _forward_average_x(atom, i, j, k, x, y, z):
        # volume size in the x direction
        dx = (x[i+1] - x[i-1]) / 2
        # volume size in the y direction
        dy = y[j] - y[j-1]
        # volume size in the z direction
        dz = z[k] - z[k-1]

        # forward average
        atom[1] = dx * dy * dz / 2
        atom[2] = atom[1]

    def forward_average_z(self, x, y, z):
        atom = numpy.zeros([self.nx, self.ny, self.nz, self.dof, self.dof, 3, 3, 3])
        for i in range(self.nx):
            for j in range(self.ny):
                for k in range(self.nz):
                    Derivatives._forward_average_x(atom[i, j, k, 4, 2, 1, 1, :], k, j, i, z, y, x)
        return atom


class ConvectiveTerm:

    def __init__(self, nx, ny, nz):
        self.nx = nx
        self.ny = ny
        self.nz = nz

    def average(self, bil):
        bil[:, :, :, :] = 1/2

    def averages(self, bil):
        self.average(bil[:, :, :, 0, 0, :]) # tMxU
        self.average(bil[:, :, :, 1, 1, :]) # tMyV
        self.average(bil[:, :, :, 2, 2, :]) # tMzW
        self.average(bil[:, :, :, 1, 0, :]) # tMxV
        self.average(bil[:, :, :, 2, 0, :]) # tMxW
        self.average(bil[:, :, :, 0, 1, :]) # tMyU
        self.average(bil[:, :, :, 2, 1, :]) # tMyW
        self.average(bil[:, :, :, 0, 2, :]) # tMzU
        self.average(bil[:, :, :, 1, 2, :]) # tMzV

    @staticmethod
    def _state_average(atom, bil, state, i):
        for d2 in range(2):
            coef = bil[d2]
            if abs(coef) > 1e-15:
                atom[0] += coef * state[i+d2]

    def MxU(self, atom, bil, state):
        for i in range(self.nx):
            for j in range(self.ny):
                for k in range(self.nz):
                    ConvectiveTerm._state_average(atom[i, j, k, 0, 0:1], bil[i, j, k, 0, 0, :], state[:, j, k, 0], i-1)

    def MxV(self, atom, bil, state):
        for i in range(self.nx):
            for j in range(self.ny):
                for k in range(self.nz):
                    ConvectiveTerm._state_average(atom[i, j, k, 1, 0:1], bil[i, j, k, 1, 0, :], state[:, j, k, 1], i)

    def MxW(self, atom, bil, state):
        for i in range(self.nx):
            for j in range(self.ny):
                for k in range(self.nz):
                    ConvectiveTerm._state_average(atom[i, j, k, 2, 0:1], bil[i, j, k, 2, 0, :], state[:, j, k, 2], i)

    def MyU(self, atom, bil, state):
        for i in range(self.nx):
            for j in range(self.ny):
                for k in range(self.nz):
                    ConvectiveTerm._state_average(atom[i, j, k, 0, 1:2], bil[i, j, k, 0, 1, :], state[i, :, k, 0], j)

    def MyV(self, atom, bil, state):
        for i in range(self.nx):
            for j in range(self.ny):
                for k in range(self.nz):
                    ConvectiveTerm._state_average(atom[i, j, k, 1, 1:2], bil[i, j, k, 1, 1, :], state[i, :, k, 1], j-1)

    def MyW(self, atom, bil, state):
        for i in range(self.nx):
            for j in range(self.ny):
                for k in range(self.nz):
                    ConvectiveTerm._state_average(atom[i, j, k, 2, 1:2], bil[i, j, k, 2, 1, :], state[i, :, k, 2], j)

    def MzU(self, atom, bil, state):
        for i in range(self.nx):
            for j in range(self.ny):
                for k in range(self.nz):
                    ConvectiveTerm._state_average(atom[i, j, k, 0, 2:3], bil[i, j, k, 0, 2, :], state[i, j, :, 0], k)

    def MzV(self, atom, bil, state):
        for i in range(self.nx):
            for j in range(self.ny):
                for k in range(self.nz):
                    ConvectiveTerm._state_average(atom[i, j, k, 1, 2:3], bil[i, j, k, 1, 2, :], state[i, j, :, 1], k)

    def MzW(self, atom, bil, state):
        for i in range(self.nx):
            for j in range(self.ny):
                for k in range(self.nz):
                    ConvectiveTerm._state_average(atom[i, j, k, 2, 2:3], bil[i, j, k, 2, 2, :], state[i, j, :, 2], k-1)

    def u_x(self, atom, x, y, z):
        for i in range(self.nx):
            for j in range(self.ny):
                for k in range(self.nz):
                    Derivatives._backward_u_x(atom[i, j, k, 0, :], i, j, k, x, y, z)

    def v_y(self, atom, x, y, z):
        for i in range(self.nx):
            for j in range(self.ny):
                for k in range(self.nz):
                    Derivatives._backward_u_x(atom[i, j, k, 1, :], j, i, k, y, x, z)

    def w_z(self, atom, x, y, z):
        for i in range(self.nx):
            for j in range(self.ny):
                for k in range(self.nz):
                    Derivatives._backward_u_x(atom[i, j, k, 2, :], k, j, i, z, y, x)

    def u_y(self, atom, x, y, z):
        for i in range(self.nx):
            for j in range(self.ny):
                for k in range(self.nz):
                    Derivatives._backward_u_y(atom[i, j, k, 0, :], i, j, k, x, y, z)

    def v_x(self, atom, x, y, z):
        for i in range(self.nx):
            for j in range(self.ny):
                for k in range(self.nz):
                    Derivatives._backward_u_y(atom[i, j, k, 1, :], j, i, k, y, x, z)

    def w_y(self, atom, x, y, z):
        for i in range(self.nx):
            for j in range(self.ny):
                for k in range(self.nz):
                    Derivatives._backward_u_y(atom[i, j, k, 2, :], k, j, i, z, y, x)

    def u_z(self, atom, x, y, z):
        for i in range(self.nx):
            for j in range(self.ny):
                for k in range(self.nz):
                    Derivatives._backward_u_z(atom[i, j, k, 0, :], i, j, k, x, y, z)

    def v_z(self, atom, x, y, z):
        for i in range(self.nx):
            for j in range(self.ny):
                for k in range(self.nz):
                    Derivatives._backward_u_z(atom[i, j, k, 1, :], j, i, k, y, x, z)

    def w_x(self, atom, x, y, z):
        for i in range(self.nx):
            for j in range(self.ny):
                for k in range(self.nz):
                    Derivatives._backward_u_z(atom[i, j, k, 2, :], k, j, i, z, y, x)

    def dirichlet_east(self, atom):
        atom[self.nx-1, :, :, 3, 0, :] = 0
        atom[self.nx-1, :, :, 1, 0, :] = 0
        atom[self.nx-1, :, :, 2, 0, :] = 0
        atom[self.nx-1, :, :, 0, 1, :] = 0
        atom[self.nx-1, :, :, 0, 2, :] = 0
        atom[self.nx-1, :, :, 0, 0, 1] = 0
        atom[self.nx-1, :, :, 3, 1, 1] = 0
        atom[self.nx-1, :, :, 3, 2, 1] = 0

    def dirichlet_west(self, atom):
        atom[0, :, :, 0, 0, 0] = 0
        atom[0, :, :, 3, 1, 0] = 0
        atom[0, :, :, 3, 2, 0] = 0

    def dirichlet_north(self, atom):
        atom[:, self.ny-1, :, 4, 1, :] = 0
        atom[:, self.ny-1, :, 0, 1, :] = 0
        atom[:, self.ny-1, :, 2, 1, :] = 0
        atom[:, self.ny-1, :, 1, 0, :] = 0
        atom[:, self.ny-1, :, 1, 0, :] = 0
        atom[:, self.ny-1, :, 1, 1, 1] = 0
        atom[:, self.ny-1, :, 4, 0, 1] = 0
        atom[:, self.ny-1, :, 4, 2, 1] = 0

    def dirichlet_south(self, atom):
        atom[:, 0, :, 1, 1, 0] = 0
        atom[:, 0, :, 4, 0, 0] = 0
        atom[:, 0, :, 4, 2, 0] = 0

    def dirichlet_top(self, atom):
        atom[:, :, self.nz-1, 5, 2, :] = 0
        atom[:, :, self.nz-1, 0, 2, :] = 0
        atom[:, :, self.nz-1, 1, 2, :] = 0
        atom[:, :, self.nz-1, 2, 0, :] = 0
        atom[:, :, self.nz-1, 2, 1, :] = 0
        atom[:, :, self.nz-1, 2, 2, 1] = 0
        atom[:, :, self.nz-1, 5, 0, 1] = 0
        atom[:, :, self.nz-1, 5, 1, 1] = 0

    def dirichlet_bottom(self, atom):
        atom[:, :, 0, 2, 2, 0] = 0
        atom[:, :, 0, 5, 0, 0] = 0
        atom[:, :, 0, 5, 1, 0] = 0
