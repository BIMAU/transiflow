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
    def u_yy(atom, nx, ny, nz, x, y, z):
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    Derivatives._u_yy(atom[i, j, k, :, 1, 1], i, j, k, x, y, z)

    @staticmethod
    def v_xx(atom, nx, ny, nz, x, y, z):
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    Derivatives._u_yy(atom[i, j, k, 1, :, 1], j, i, k, y, x, z)

    @staticmethod
    def w_yy(atom, nx, ny, nz, x, y, z):
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    Derivatives._u_yy(atom[i, j, k, 1, 1, :], k, j, i, z, y, x)

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
    def u_zz(atom, nx, ny, nz, x, y, z):
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    Derivatives._u_zz(atom[i, j, k, :, 1, 1], i, j, k, x, y, z)

    @staticmethod
    def v_zz(atom, nx, ny, nz, x, y, z):
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    Derivatives._u_zz(atom[i, j, k, 1, :, 1], j, i, k, y, x, z)

    @staticmethod
    def w_xx(atom, nx, ny, nz, x, y, z):
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    Derivatives._u_zz(atom[i, j, k, 1, 1, :], k, j, i, z, y, x)

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
    def p_x(atom, nx, ny, nz, x, y, z):
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    Derivatives._p_x(atom[i, j, k, :, 1, 1], i, j, k, x, y, z)

    @staticmethod
    def p_y(atom, nx, ny, nz, x, y, z):
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    Derivatives._p_x(atom[i, j, k, 1, :, 1], j, i, k, y, x, z)

    @staticmethod
    def p_z(atom, nx, ny, nz, x, y, z):
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    Derivatives._p_x(atom[i, j, k, 1, 1, :], k, j, i, z, y, x)

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
    def u_x(atom, nx, ny, nz, x, y, z):
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    Derivatives._u_x(atom[i, j, k, :, 1, 1], i, j, k, x, y, z)

    @staticmethod
    def u_y(atom, nx, ny, nz, x, y, z):
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    Derivatives._u_x(atom[i, j, k, 1, :, 1], j, i, k, y, x, z)

    @staticmethod
    def u_z(atom, nx, ny, nz, x, y, z):
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    Derivatives._u_x(atom[i, j, k, 1, 1, :], k, j, i, z, y, x)
