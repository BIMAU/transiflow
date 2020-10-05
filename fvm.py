import numpy

class CrsMatrix:
    def __init__(self, coA=None, jcoA=None, begA=None):
        self.coA = coA
        self.jcoA = jcoA
        self.begA = begA

def create_uniform_coordinate_vector(nx):
    dx = 1 / nx
    return numpy.roll(numpy.arange(-dx, 1+2*dx, dx), -2)

def linear_part(Re, nx, ny, nz):
    x = create_uniform_coordinate_vector(nx)
    y = create_uniform_coordinate_vector(ny)
    z = create_uniform_coordinate_vector(nz)

    return 1 / Re * (Derivatives.u_xx(nx, ny, nz, x, y, z) + Derivatives.u_yy(nx, ny, nz, x, y, z) + Derivatives.u_zz(nx, ny, nz, x, y, z) \
                  + Derivatives.v_xx(nx, ny, nz, x, y, z) + Derivatives.v_yy(nx, ny, nz, x, y, z) + Derivatives.v_zz(nx, ny, nz, x, y, z) \
                  + Derivatives.w_xx(nx, ny, nz, x, y, z) + Derivatives.w_yy(nx, ny, nz, x, y, z) + Derivatives.w_zz(nx, ny, nz, x, y, z)) \
                  - (Derivatives.p_x(nx, ny, nz, x, y, z) + Derivatives.p_y(nx, ny, nz, x, y, z) + Derivatives.p_z(nx, ny, nz, x, y, z)) \
                  + Derivatives.div(nx, ny, nz, x, y, z)

def boundaries(atom, nx, ny, nz):
    BoundaryConditions.dirichlet_east(atom, nx, ny, nz)
    BoundaryConditions.dirichlet_west(atom, nx, ny, nz)
    frc = BoundaryConditions.dirichlet_north(atom, nx, ny, nz)
    BoundaryConditions.dirichlet_south(atom, nx, ny, nz)
    frc += BoundaryConditions.dirichlet_top(atom, nx, ny, nz)
    BoundaryConditions.dirichlet_bottom(atom, nx, ny, nz)
    return frc

def convection(state, nx, ny, nz):
    x = create_uniform_coordinate_vector(nx)
    y = create_uniform_coordinate_vector(ny)
    z = create_uniform_coordinate_vector(nz)

    dof = 4
    state_mtx = numpy.zeros([nx, ny, nz, dof])
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                for d in range(dof):
                    state_mtx[i, j, k, d] = state[d + i * dof + j * dof * nx + k * dof * nx * ny]

    return Derivatives.convection(state_mtx, nx, ny, nz, x, y, z)

def assemble(atom, nx, ny, nz):
    dof = 4
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

def rhs(state, atom, nx, ny, nz):
    dof = 4
    row = 0
    n = nx * ny * nz * dof

    # Put the state in matrix form
    state_mtx = numpy.zeros([nx+2, ny+2, nz+2, dof])
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                for d in range(dof):
                    state_mtx[i+1, j+1, k+1, d] = state[d + i * dof + j * dof * nx + k * dof * nx * ny]

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

def ldc_forcing_top(atom, nx, ny, nz):
    dof = 4
    n = nx * ny * nz * dof
    out = numpy.zeros(n)

    if nz <= 1:
        return out

    k = nz-1
    z = 2
    for j in range(ny):
        for i in range(nx):
            for y in range(3):
                for x in range(3):
                    offset = i * dof + j * nx * dof + k * nx * ny * dof + 1
                    out[offset] += 2 * atom[i, j, k, 1, 0, x, y, z]
                    offset = i * dof + j * nx * dof + k * nx * ny * dof
                    out[offset] += 2 * atom[i, j, k, 0, 0, x, y, z]
    return out

def ldc_forcing_north(atom, nx, ny, nz):
    dof = 4
    n = nx * ny * nz * dof
    out = numpy.zeros(n)

    if nz > 1:
        return out

    j = ny-1
    y = 2
    for k in range(nz):
        for i in range(nx):
            for z in range(3):
                for x in range(3):
                    offset = i * dof + j * nx * dof + k * nx * ny * dof
                    out[offset] += 2 * atom[i, j, k, 0, 0, x, y, z]
                    offset = i * dof + j * nx * dof + k * nx * ny * dof + 2
                    out[offset] += 2 * atom[i, j, k, 2, 0, x, y, z]
    return out

class BoundaryConditions:
    @staticmethod
    def dirichlet_east(atom, nx, ny, nz):
        # At the boundary u[i] = 0, v[i] + v[i+1] = 2*V similar for w. So v[i+1] = -v[i]+2*V.
        atom[nx-1, 0:ny, 0:nz, :, [1,2], 1, :, :] -= atom[nx-1, 0:ny, 0:nz, :, [1,2], 2, :, :]
        atom[nx-1, 0:ny, 0:nz, :, 0, 1, :, :] = 0
        atom[nx-1, 0:ny, 0:nz, 0, :, :, :, :] = 0
        atom[nx-1, 0:ny, 0:nz, :, :, 2, :, :] = 0
        atom[nx-1, 0:ny, 0:nz, 0, 0, 1, 1, 1] = -1
        # TODO: Do we need this?
        atom[nx-2, 0:ny, 0:nz, 0, 0, 2, :, :] = 0

    @staticmethod
    def dirichlet_west(atom, nx, ny, nz):
        # At the boundary u[i-1] = 0, v[i-1] + v[i] = 0 similar for w. So v[i-1] = -v[i].
        atom[0, 0:ny, 0:nz, :, [1,2], 1, :, :] -= atom[0, 0:ny, 0:nz, :, [1,2], 0, :, :]
        atom[0, 0:ny, 0:nz, :, :, 0, :, :] = 0

    @staticmethod
    def dirichlet_north(atom, nx, ny, nz):
        # At the boundary v[i] = 0, u[i] + u[i+1] = 2*U similar for w. So u[i+1] = -u[i]+2*U.
        atom[0:nx, ny-1, 0:nz, :, [0,2], :, 1, :] -= atom[0:nx, ny-1, 0:nz, :, [0,2], :, 2, :]
        atom[0:nx, ny-1, 0:nz, :, 1, :, 1, :] = 0
        frc = ldc_forcing_north(atom, nx, ny, nz)
        atom[0:nx, ny-1, 0:nz, 1, :, :, :, :] = 0
        atom[0:nx, ny-1, 0:nz, :, :, :, 2, :] = 0
        atom[0:nx, ny-1, 0:nz, 1, 1, 1, 1, 1] = -1
        # TODO: Do we need this?
        atom[0:nx, ny-2, 0:nz, 1, 1, :, 2, :] = 0
        return frc

    @staticmethod
    def dirichlet_south(atom, nx, ny, nz):
        # At the boundary v[i-1] = 0, u[i-1] + u[i] = 0 similar for w. So u[i-1] = -u[i].
        atom[0:nx, 0, 0:nz, :, [0,2], :, 1, :] -= atom[0:nx, 0, 0:nz, :, [0,2], :, 0, :]
        atom[0:nx, 0, 0:nz, :, :, :, 0, :] = 0

    @staticmethod
    def dirichlet_top(atom, nx, ny, nz):
        # At the boundary w[i] = 0, u[i] + u[i+1] = 2*U similar for v. So u[i+1] = -u[i]+2*U.
        atom[0:nx, 0:ny, nz-1, :, [0,1], :, :, 1] -= atom[0:nx, 0:ny, nz-1, :, [0,1], :, :, 2]
        atom[0:nx, 0:ny, nz-1, :, 2, :, :, 1] = 0
        frc = ldc_forcing_top(atom, nx, ny, nz)
        atom[0:nx, 0:ny, nz-1, 2, :, :, :, :] = 0
        atom[0:nx, 0:ny, nz-1, :, :, :, :, 2] = 0
        atom[0:nx, 0:ny, nz-1, 2, 2, 1, 1, 1] = -1
        # TODO: Do we need this?
        atom[0:nx, 0:ny, nz-2, 2, 2, :, :, 2] = 0
        return frc

    @staticmethod
    def dirichlet_bottom(atom, nx, ny, nz):
        # At the boundary w[i-1] = 0, u[i-1] + u[i] = 0 similar for v. So u[i-1] = -u[i].
        atom[0:nx, 0:ny, 0, :, [0,1], :, :, 1] -= atom[0:nx, 0:ny, 0, :, [0,1], :, :, 0]
        atom[0:nx, 0:ny, 0, :, :, :, :, 0] = 0

class Derivatives:
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

    @staticmethod
    def u_xx(nx, ny, nz, x, y, z):
        atom = numpy.zeros([nx, ny, nz, 4, 4, 3, 3, 3])
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    Derivatives._u_xx(atom[i, j, k, 0, 0, :, 1, 1], i, j, k, x, y, z)
        return atom

    @staticmethod
    def v_yy(nx, ny, nz, x, y, z):
        atom = numpy.zeros([nx, ny, nz, 4, 4, 3, 3, 3])
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    Derivatives._u_xx(atom[i, j, k, 1, 1, 1, :, 1], j, i, k, y, x, z)
        return atom

    @staticmethod
    def w_zz(nx, ny, nz, x, y, z):
        atom = numpy.zeros([nx, ny, nz, 4, 4, 3, 3, 3])
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
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

    @staticmethod
    def u_yy(nx, ny, nz, x, y, z):
        atom = numpy.zeros([nx, ny, nz, 4, 4, 3, 3, 3])
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    Derivatives._u_yy(atom[i, j, k, 0, 0, 1, :, 1], i, j, k, x, y, z)
        return atom

    @staticmethod
    def v_xx(nx, ny, nz, x, y, z):
        atom = numpy.zeros([nx, ny, nz, 4, 4, 3, 3, 3])
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    Derivatives._u_yy(atom[i, j, k, 1, 1, :, 1, 1], j, i, k, y, x, z)
        return atom

    @staticmethod
    def w_yy(nx, ny, nz, x, y, z):
        atom = numpy.zeros([nx, ny, nz, 4, 4, 3, 3, 3])
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
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

    @staticmethod
    def u_zz(nx, ny, nz, x, y, z):
        atom = numpy.zeros([nx, ny, nz, 4, 4, 3, 3, 3])
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    Derivatives._u_zz(atom[i, j, k, 0, 0, 1, 1, :], i, j, k, x, y, z)
        return atom

    @staticmethod
    def v_zz(nx, ny, nz, x, y, z):
        atom = numpy.zeros([nx, ny, nz, 4, 4, 3, 3, 3])
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    Derivatives._u_zz(atom[i, j, k, 1, 1, 1, 1, :], j, i, k, y, x, z)
        return atom

    @staticmethod
    def w_xx(nx, ny, nz, x, y, z):
        atom = numpy.zeros([nx, ny, nz, 4, 4, 3, 3, 3])
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    Derivatives._u_zz(atom[i, j, k, 2, 2, :, 1, 1], k, j, i, z, y, x)
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

    @staticmethod
    def p_x(nx, ny, nz, x, y, z):
        atom = numpy.zeros([nx, ny, nz, 4, 4, 3, 3, 3])
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    Derivatives._forward_u_x(atom[i, j, k, 0, 3, :, 1, 1], i, j, k, x, y, z)
        return atom

    @staticmethod
    def p_y(nx, ny, nz, x, y, z):
        atom = numpy.zeros([nx, ny, nz, 4, 4, 3, 3, 3])
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    Derivatives._forward_u_x(atom[i, j, k, 1, 3, 1, :, 1], j, i, k, y, x, z)
        return atom

    @staticmethod
    def p_z(nx, ny, nz, x, y, z):
        atom = numpy.zeros([nx, ny, nz, 4, 4, 3, 3, 3])
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
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

    @staticmethod
    def u_x(nx, ny, nz, x, y, z):
        atom = numpy.zeros([nx, ny, nz, 4, 4, 3, 3, 3])
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    Derivatives._backward_u_x(atom[i, j, k, 3, 0, :, 1, 1], i, j, k, x, y, z)
        return atom

    @staticmethod
    def v_y(nx, ny, nz, x, y, z):
        atom = numpy.zeros([nx, ny, nz, 4, 4, 3, 3, 3])
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    Derivatives._backward_u_x(atom[i, j, k, 3, 1, 1, :, 1], j, i, k, y, x, z)
        return atom

    @staticmethod
    def w_z(nx, ny, nz, x, y, z):
        atom = numpy.zeros([nx, ny, nz, 4, 4, 3, 3, 3])
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    Derivatives._backward_u_x(atom[i, j, k, 3, 2, 1, 1, :], k, j, i, z, y, x)
        return atom

    @staticmethod
    def _backward_u_y(atom, i, j, k, x, y, z):
        # volume size in the y direction
        dx = (x[i+1] - x[i-1]) / 2
        # volume size in the z direction
        dz = z[k] - z[k-1]

        # backward difference
        atom[1] = dx * dz
        atom[0] = -atom[1]

    @staticmethod
    def _backward_u_z(atom, i, j, k, x, y, z):
        # volume size in the y direction
        dx = (x[i+1] - x[i-1]) / 2
        # volume size in the z direction
        dy = y[k] - y[k-1]

        # backward difference
        atom[1] = dx * dy
        atom[0] = -atom[1]

    @staticmethod
    def div(nx, ny, nz, x, y, z):
        return Derivatives.u_x(nx, ny, nz, x, y, z) + Derivatives.v_y(nx, ny, nz, x, y, z) + Derivatives.w_z(nx, ny, nz, x, y, z)

    @staticmethod
    def _convection_v_u(atomJ, atomF, MxV, MyU, bil, varV, varU, ny, j):
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
    def convection_u_u(atomJ, atomF, averages, bil, nx, ny, nz):
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    Derivatives._convection_v_u(atomJ[i, j, k, 0, 0, :, :, :], atomF[i, j, k, 0, 0, :, :, :],
                                                averages[:, j, k, 0, 0], averages[:, j, k, 0, 0],
                                                bil[:, j, k, :, :, :], 0, 0, nx, i)

    @staticmethod
    def convection_v_u(atomJ, atomF, averages, bil, nx, ny, nz):
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    Derivatives._convection_v_u(atomJ[i, j, k, 0, 1, :, :, :], atomF[i, j, k, 0, 0, :, :, :],
                                                averages[i, :, k, 1, 0], averages[i, :, k, 0, 1],
                                                bil[i, :, k, :, :, :], 1, 0, ny, j)

    @staticmethod
    def convection_w_u(atomJ, atomF, averages, bil, nx, ny, nz):
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    Derivatives._convection_v_u(atomJ[i, j, k, 0, 2, :, :, :], atomF[i, j, k, 0, 0, :, :, :],
                                                averages[i, j, :, 2, 0], averages[i, j, :, 0, 2],
                                                bil[i, j, :, :, :, :], 2, 0, nz, k)

    @staticmethod
    def convection_u_v(atomJ, atomF, averages, bil, nx, ny, nz):
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    Derivatives._convection_v_u(atomJ[i, j, k, 1, 0, :, :, :], atomF[i, j, k, 1, 1, :, :, :],
                                                averages[:, j, k, 0, 1], averages[:, j, k, 1, 0],
                                                bil[:, j, k, :, :, :], 0, 1, nx, i)

    @staticmethod
    def convection_v_v(atomJ, atomF, averages, bil, nx, ny, nz):
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    Derivatives._convection_v_u(atomJ[i, j, k, 1, 1, :, :, :], atomF[i, j, k, 1, 1, :, :, :],
                                                averages[i, :, k, 1, 1], averages[i, :, k, 1, 1],
                                                bil[i, :, k, :, :, :], 1, 1, ny, j)

    @staticmethod
    def convection_w_v(atomJ, atomF, averages, bil, nx, ny, nz):
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    Derivatives._convection_v_u(atomJ[i, j, k, 1, 2, :, :, :], atomF[i, j, k, 1, 1, :, :, :],
                                                averages[i, j, :, 2, 1], averages[i, j, :, 1, 2],
                                                bil[i, j, :, :, :, :], 2, 1, nz, k)

    @staticmethod
    def convection_u_w(atomJ, atomF, averages, bil, nx, ny, nz):
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    Derivatives._convection_v_u(atomJ[i, j, k, 2, 0, :, :, :], atomF[i, j, k, 2, 2, :, :, :],
                                                averages[:, j, k, 0, 2], averages[:, j, k, 2, 0],
                                                bil[:, j, k, :, :, :], 0, 2, nx, i)

    @staticmethod
    def convection_v_w(atomJ, atomF, averages, bil, nx, ny, nz):
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    Derivatives._convection_v_u(atomJ[i, j, k, 2, 1, :, :, :], atomF[i, j, k, 2, 2, :, :, :],
                                                averages[i, :, k, 1, 2], averages[i, :, k, 2, 1],
                                                bil[i, :, k, :, :, :], 1, 2, ny, j)

    @staticmethod
    def convection_w_w(atomJ, atomF, averages, bil, nx, ny, nz):
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    Derivatives._convection_v_u(atomJ[i, j, k, 2, 2, :, :, :], atomF[i, j, k, 2, 2, :, :, :],
                                                averages[i, j, :, 2, 2], averages[i, j, :, 2, 2],
                                                bil[i, j, :, :, :, :], 2, 2, nz, k)

    @staticmethod
    def convection(state, nx, ny, nz, x, y, z):
        bil = numpy.zeros([nx, ny, nz, 6, 3, 2])

        ConvectiveTerm.averages(bil)

        ConvectiveTerm.u_x(bil[:, :, :, 3, :, :], nx, ny, nz, x, y, z) # tMxUMxU
        ConvectiveTerm.u_y(bil[:, :, :, 4, :, :], nx, ny, nz, x, y, z) # tMxVMyU
        ConvectiveTerm.u_z(bil[:, :, :, 5, :, :], nx, ny, nz, x, y, z) # tMxWMzU
        ConvectiveTerm.v_x(bil[:, :, :, 3, :, :], nx, ny, nz, x, y, z) # tMyUMxV
        ConvectiveTerm.v_y(bil[:, :, :, 4, :, :], nx, ny, nz, x, y, z) # tMyVMyV
        ConvectiveTerm.v_z(bil[:, :, :, 5, :, :], nx, ny, nz, x, y, z) # tMyWMzV
        ConvectiveTerm.w_x(bil[:, :, :, 3, :, :], nx, ny, nz, x, y, z) # tMzUMxW
        ConvectiveTerm.w_y(bil[:, :, :, 4, :, :], nx, ny, nz, x, y, z) # tMzVMyW
        ConvectiveTerm.w_z(bil[:, :, :, 5, :, :], nx, ny, nz, x, y, z) # tMzWMzW

        ConvectiveTerm.dirichlet_east(bil, nx, ny, nz)
        ConvectiveTerm.dirichlet_west(bil, nx, ny, nz)
        ConvectiveTerm.dirichlet_north(bil, nx, ny, nz)
        ConvectiveTerm.dirichlet_south(bil, nx, ny, nz)
        ConvectiveTerm.dirichlet_top(bil, nx, ny, nz)
        ConvectiveTerm.dirichlet_bottom(bil, nx, ny, nz)

        averages = numpy.zeros([nx, ny, nz, 3, 3])

        ConvectiveTerm.MxU(averages, bil, state, nx, ny, nz)
        ConvectiveTerm.MxV(averages, bil, state, nx, ny, nz)
        ConvectiveTerm.MxW(averages, bil, state, nx, ny, nz)
        ConvectiveTerm.MyU(averages, bil, state, nx, ny, nz)
        ConvectiveTerm.MyV(averages, bil, state, nx, ny, nz)
        ConvectiveTerm.MyW(averages, bil, state, nx, ny, nz)
        ConvectiveTerm.MzU(averages, bil, state, nx, ny, nz)
        ConvectiveTerm.MzV(averages, bil, state, nx, ny, nz)
        ConvectiveTerm.MzW(averages, bil, state, nx, ny, nz)

        atomJ = numpy.zeros([nx, ny, nz, 4, 4, 3, 3, 3])
        atomF = numpy.zeros([nx, ny, nz, 4, 4, 3, 3, 3])

        Derivatives.convection_u_u(atomJ, atomF, averages, bil, nx, ny, nz)
        Derivatives.convection_u_v(atomJ, atomF, averages, bil, nx, ny, nz)
        Derivatives.convection_u_w(atomJ, atomF, averages, bil, nx, ny, nz)
        Derivatives.convection_v_u(atomJ, atomF, averages, bil, nx, ny, nz)
        Derivatives.convection_v_v(atomJ, atomF, averages, bil, nx, ny, nz)
        Derivatives.convection_v_w(atomJ, atomF, averages, bil, nx, ny, nz)
        Derivatives.convection_w_u(atomJ, atomF, averages, bil, nx, ny, nz)
        Derivatives.convection_w_v(atomJ, atomF, averages, bil, nx, ny, nz)
        Derivatives.convection_w_w(atomJ, atomF, averages, bil, nx, ny, nz)

        atomJ += atomF

        return (atomJ, atomF)


class ConvectiveTerm:

    @staticmethod
    def average(bil):
        bil[:, :, :, :] = 1/2

    @staticmethod
    def averages(bil):
        ConvectiveTerm.average(bil[:, :, :, 0, 0, :]) # tMxU
        ConvectiveTerm.average(bil[:, :, :, 1, 1, :]) # tMyV
        ConvectiveTerm.average(bil[:, :, :, 2, 2, :]) # tMzW
        ConvectiveTerm.average(bil[:, :, :, 1, 0, :]) # tMxV
        ConvectiveTerm.average(bil[:, :, :, 2, 0, :]) # tMxW
        ConvectiveTerm.average(bil[:, :, :, 0, 1, :]) # tMyU
        ConvectiveTerm.average(bil[:, :, :, 2, 1, :]) # tMyW
        ConvectiveTerm.average(bil[:, :, :, 0, 2, :]) # tMzU
        ConvectiveTerm.average(bil[:, :, :, 1, 2, :]) # tMzV

    @staticmethod
    def _state_average(atom, bil, state, i):
        for d2 in range(2):
            coef = bil[d2]
            if abs(coef) > 1e-15:
                atom[0] += coef * state[i+d2]

    @staticmethod
    def MxU(atom, bil, state, nx, ny, nz):
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    ConvectiveTerm._state_average(atom[i, j, k, 0, 0:1], bil[i, j, k, 0, 0, :], state[:, j, k, 0], i-1)

    @staticmethod
    def MxV(atom, bil, state, nx, ny, nz):
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    ConvectiveTerm._state_average(atom[i, j, k, 1, 0:1], bil[i, j, k, 1, 0, :], state[:, j, k, 1], i)

    @staticmethod
    def MxW(atom, bil, state, nx, ny, nz):
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    ConvectiveTerm._state_average(atom[i, j, k, 2, 0:1], bil[i, j, k, 2, 0, :], state[:, j, k, 2], i)

    @staticmethod
    def MyU(atom, bil, state, nx, ny, nz):
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    ConvectiveTerm._state_average(atom[i, j, k, 0, 1:2], bil[i, j, k, 0, 1, :], state[i, :, k, 0], j)

    @staticmethod
    def MyV(atom, bil, state, nx, ny, nz):
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    ConvectiveTerm._state_average(atom[i, j, k, 1, 1:2], bil[i, j, k, 1, 1, :], state[i, :, k, 1], j-1)
  
    @staticmethod
    def MyW(atom, bil, state, nx, ny, nz):
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    ConvectiveTerm._state_average(atom[i, j, k, 2, 1:2], bil[i, j, k, 2, 1, :], state[i, :, k, 2], j)

    @staticmethod
    def MzU(atom, bil, state, nx, ny, nz):
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    ConvectiveTerm._state_average(atom[i, j, k, 0, 2:3], bil[i, j, k, 0, 2, :], state[i, j, :, 0], k)

    @staticmethod
    def MzV(atom, bil, state, nx, ny, nz):
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    ConvectiveTerm._state_average(atom[i, j, k, 1, 2:3], bil[i, j, k, 1, 2, :], state[i, j, :, 1], k)

    @staticmethod
    def MzW(atom, bil, state, nx, ny, nz):
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    ConvectiveTerm._state_average(atom[i, j, k, 2, 2:3], bil[i, j, k, 2, 2, :], state[i, j, :, 2], k-1)

    @staticmethod
    def u_x(atom, nx, ny, nz, x, y, z):
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    Derivatives._backward_u_x(atom[i, j, k, 0, :], i, j, k, x, y, z)

    @staticmethod
    def v_y(atom, nx, ny, nz, x, y, z):
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    Derivatives._backward_u_x(atom[i, j, k, 1, :], j, i, k, y, x, z)

    @staticmethod
    def w_z(atom, nx, ny, nz, x, y, z):
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    Derivatives._backward_u_x(atom[i, j, k, 2, :], k, j, i, z, y, x)

    @staticmethod
    def u_y(atom, nx, ny, nz, x, y, z):
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    Derivatives._backward_u_y(atom[i, j, k, 0, :], i, j, k, x, y, z)

    @staticmethod
    def v_x(atom, nx, ny, nz, x, y, z):
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    Derivatives._backward_u_y(atom[i, j, k, 1, :], j, i, k, y, x, z)

    @staticmethod
    def w_y(atom, nx, ny, nz, x, y, z):
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    Derivatives._backward_u_y(atom[i, j, k, 2, :], k, j, i, z, y, x)

    @staticmethod
    def u_z(atom, nx, ny, nz, x, y, z):
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    Derivatives._backward_u_z(atom[i, j, k, 0, :], i, j, k, x, y, z)

    @staticmethod
    def v_z(atom, nx, ny, nz, x, y, z):
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    Derivatives._backward_u_z(atom[i, j, k, 1, :], j, i, k, y, x, z)

    @staticmethod
    def w_x(atom, nx, ny, nz, x, y, z):
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    Derivatives._backward_u_z(atom[i, j, k, 2, :], k, j, i, z, y, x)

    @staticmethod
    def dirichlet_east(atom, nx, ny, nz):
        atom[nx-1, 0:ny, 0:nz, 3, 0, :] = 0
        atom[nx-1, 0:ny, 0:nz, 1, 0, :] = 0
        atom[nx-1, 0:ny, 0:nz, 2, 0, :] = 0
        atom[nx-1, 0:ny, 0:nz, 0, 1, :] = 0
        atom[nx-1, 0:ny, 0:nz, 0, 2, :] = 0
        atom[nx-1, 0:ny, 0:nz, 0, 0, 1] = 0
        atom[nx-1, 0:ny, 0:nz, 3, 1, 1] = 0
        atom[nx-1, 0:ny, 0:nz, 3, 2, 1] = 0

    @staticmethod
    def dirichlet_west(atom, nx, ny, nz):
        atom[0, 0:ny, 0:nz, 0, 0, 0] = 0
        atom[0, 0:ny, 0:nz, 3, 1, 0] = 0
        atom[0, 0:ny, 0:nz, 3, 2, 0] = 0

    @staticmethod
    def dirichlet_north(atom, nx, ny, nz):
        atom[0:nx, ny-1, 0:nz, 4, 1, :] = 0
        atom[0:nx, ny-1, 0:nz, 0, 1, :] = 0
        atom[0:nx, ny-1, 0:nz, 2, 1, :] = 0
        atom[0:nx, ny-1, 0:nz, 1, 0, :] = 0
        atom[0:nx, ny-1, 0:nz, 1, 0, :] = 0
        atom[0:nx, ny-1, 0:nz, 1, 1, 1] = 0
        atom[0:nx, ny-1, 0:nz, 4, 0, 1] = 0
        atom[0:nx, ny-1, 0:nz, 4, 2, 1] = 0

    @staticmethod
    def dirichlet_south(atom, nx, ny, nz):
        atom[0:nx, 0, 0:nz, 1, 1, 0] = 0
        atom[0:nx, 0, 0:nz, 4, 0, 0] = 0
        atom[0:nx, 0, 0:nz, 4, 2, 0] = 0

    @staticmethod
    def dirichlet_top(atom, nx, ny, nz):
        atom[0:nx, 0:ny, nz-1, 5, 2, :] = 0
        atom[0:nx, 0:ny, nz-1, 0, 2, :] = 0
        atom[0:nx, 0:ny, nz-1, 1, 2, :] = 0
        atom[0:nx, 0:ny, nz-1, 2, 0, :] = 0
        atom[0:nx, 0:ny, nz-1, 2, 1, :] = 0
        atom[0:nx, 0:ny, nz-1, 2, 2, 1] = 0
        atom[0:nx, 0:ny, nz-1, 5, 0, 1] = 0
        atom[0:nx, 0:ny, nz-1, 5, 1, 1] = 0

    @staticmethod
    def dirichlet_bottom(atom, nx, ny, nz):
        atom[0:nx, 0:ny, 0, 2, 2, 0] = 0
        atom[0:nx, 0:ny, 0, 5, 0, 0] = 0
        atom[0:nx, 0:ny, 0, 5, 1, 0] = 0
