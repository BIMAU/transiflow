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
        atom[1] = -atom[0] + atom[2]

    @staticmethod
    def u_xx(atom, nx, ny, nz, x, y, z):
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    Derivatives._u_xx(atom[i, j, k, :, 1, 1], i, j, k, x, y, z)

    @staticmethod
    def v_yy(atom, nx, ny, nz, x, y, z):
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    Derivatives._u_xx(atom[i, j, k, 1, :, 1], j, i, k, y, x, z)

    @staticmethod
    def w_zz(atom, nx, ny, nz, x, y, z):
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    Derivatives._u_xx(atom[i, j, k, 1, 1, :], k, j, i, z, y, x)
