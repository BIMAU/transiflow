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
                  + (Derivatives.u_x(nx, ny, nz, x, y, z) + Derivatives.u_y(nx, ny, nz, x, y, z) + Derivatives.u_z(nx, ny, nz, x, y, z))

def boundaries(atom, nx, ny, nz):
    BoundaryConditions.dirichlet_east(atom, nx, ny, nz)
    BoundaryConditions.dirichlet_west(atom, nx, ny, nz)
    BoundaryConditions.dirichlet_north(atom, nx, ny, nz)
    BoundaryConditions.dirichlet_south(atom, nx, ny, nz)
    BoundaryConditions.dirichlet_top(atom, nx, ny, nz)
    BoundaryConditions.dirichlet_bottom(atom, nx, ny, nz)

def assemble(atom, nx, ny, nz):
    dof = 4
    row = 0
    idx = 0
    n = nx * ny * nz * dof
    coA = numpy.zeros(27*n)
    jcoA = numpy.zeros(27*n, dtype=int)
    begA = numpy.zeros(n+1, dtype=int)
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
    return CrsMatrix(coA, jcoA, begA)

class BoundaryConditions:
    @staticmethod
    def dirichlet_east(atom, nx, ny, nz):
        # At the boundary u[i] = 0, v[i] + v[i+1] = 2*V similar for w. So v[i+1] = -v[i]+2*V.
        atom[nx-1, 0:ny, 0:nz, [1,2,3], 1, 1, :, :] -= atom[nx-1, 0:ny, 0:nz, [1,2,3], 1, 2, :, :]
        atom[nx-1, 0:ny, 0:nz, [1,2,3], 2, 1, :, :] -= atom[nx-1, 0:ny, 0:nz, [1,2,3], 2, 2, :, :]
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
        atom[0:nx, ny-1, 0:nz, [0,2,3], 0, :, 1, :] -= atom[0:nx, ny-1, 0:nz, [0,2,3], 0, :, 2, :]
        atom[0:nx, ny-1, 0:nz, [0,2,3], 2, :, 1, :] -= atom[0:nx, ny-1, 0:nz, [0,2,3], 2, :, 2, :]
        atom[0:nx, ny-1, 0:nz, :, 1, :, 1, :] = 0
        atom[0:nx, ny-1, 0:nz, 1, :, :, :, :] = 0
        atom[0:nx, ny-1, 0:nz, :, :, :, 2, :] = 0
        atom[0:nx, ny-1, 0:nz, 1, 1, 1, 1, 1] = -1
        # TODO: Do we need this?
        atom[0:nx, ny-2, 0:nz, 1, 1, :, 2, :] = 0

    @staticmethod
    def dirichlet_south(atom, nx, ny, nz):
        # At the boundary v[i-1] = 0, u[i-1] + u[i] = 0 similar for w. So u[i-1] = -u[i].
        atom[0:nx, 0, 0:nz, :, [0,2], :, 1, :] -= atom[0:nx, 0, 0:nz, :, [0,2], :, 0, :]
        atom[0:nx, 0, 0:nz, :, :, :, 0, :] = 0

    @staticmethod
    def dirichlet_top(atom, nx, ny, nz):
        # At the boundary w[i] = 0, u[i] + u[i+1] = 2*U similar for v. So u[i+1] = -u[i]+2*U.
        atom[0:nx, 0:ny, nz-1, [0,1,3], 0, :, :, 1] -= atom[0:nx, 0:ny, nz-1, [0,1,3], 0, :, :, 2]
        atom[0:nx, 0:ny, nz-1, [0,1,3], 1, :, :, 1] -= atom[0:nx, 0:ny, nz-1, [0,1,3], 1, :, :, 2]
        atom[0:nx, 0:ny, nz-1, :, 2, :, :, 1] = 0
        atom[0:nx, 0:ny, nz-1, 2, :, :, :, :] = 0
        atom[0:nx, 0:ny, nz-1, :, :, :, :, 2] = 0
        atom[0:nx, 0:ny, nz-1, 2, 2, 1, 1, 1] = -1
        # TODO: Do we need this?
        atom[0:nx, 0:ny, nz-2, 2, 2, :, :, 2] = 0

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
    def _p_x(atom, i, j, k, x, y, z):
        # volume size in the y direction
        dy = y[j] - y[j-1]
        # volume size in the z direction
        dz = z[k] - z[k-1]

        # second order finite difference
        atom[2] = dy * dz
        atom[1] = -atom[2]

    @staticmethod
    def p_x(nx, ny, nz, x, y, z):
        atom = numpy.zeros([nx, ny, nz, 4, 4, 3, 3, 3])
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    Derivatives._p_x(atom[i, j, k, 0, 3, :, 1, 1], i, j, k, x, y, z)
        return atom

    @staticmethod
    def p_y(nx, ny, nz, x, y, z):
        atom = numpy.zeros([nx, ny, nz, 4, 4, 3, 3, 3])
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    Derivatives._p_x(atom[i, j, k, 1, 3, 1, :, 1], j, i, k, y, x, z)
        return atom

    @staticmethod
    def p_z(nx, ny, nz, x, y, z):
        atom = numpy.zeros([nx, ny, nz, 4, 4, 3, 3, 3])
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    Derivatives._p_x(atom[i, j, k, 2, 3, 1, 1, :], k, j, i, z, y, x)
        return atom

    @staticmethod
    def _u_x(atom, i, j, k, x, y, z):
        # volume size in the y direction
        dy = y[j] - y[j-1]
        # volume size in the z direction
        dz = z[k] - z[k-1]

        # second order finite difference
        atom[1] = dy * dz
        atom[0] = -atom[1]

    @staticmethod
    def u_x(nx, ny, nz, x, y, z):
        atom = numpy.zeros([nx, ny, nz, 4, 4, 3, 3, 3])
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    Derivatives._u_x(atom[i, j, k, 3, 0, :, 1, 1], i, j, k, x, y, z)
        return atom

    @staticmethod
    def u_y(nx, ny, nz, x, y, z):
        atom = numpy.zeros([nx, ny, nz, 4, 4, 3, 3, 3])
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    Derivatives._u_x(atom[i, j, k, 3, 1, 1, :, 1], j, i, k, y, x, z)
        return atom

    @staticmethod
    def u_z(nx, ny, nz, x, y, z):
        atom = numpy.zeros([nx, ny, nz, 4, 4, 3, 3, 3])
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    Derivatives._u_x(atom[i, j, k, 3, 2, 1, 1, :], k, j, i, z, y, x)
        return atom
