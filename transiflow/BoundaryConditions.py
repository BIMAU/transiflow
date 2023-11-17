import numpy

from .utils import create_state_vec

class BoundaryConditions:

    def __init__(self, nx, ny, nz, dim, dof, x, y, z):
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.dim = dim
        self.dof = dof

        self.x = x
        self.y = y
        self.z = z

    def no_slip_east(self, atom):
        '''At the boundary u[i] = 0, v[i] + v[i+1] = 2*V similar for w. So v[i+1] = -v[i]+2*V.'''
        atom[self.nx-1, :, :, :, :, 1, :, :] -= atom[self.nx-1, :, :, :, :, 2, :, :]
        atom[self.nx-1, :, :, :, 0, 1, :, :] = 0
        atom[self.nx-1, :, :, 0, :, :, :, :] = 0
        atom[self.nx-1, :, :, :, :, 2, :, :] = 0
        atom[self.nx-1, :, :, 0, 0, 1, 1, 1] = -1
        # TODO: Do we need this?
        atom[self.nx-2, :, :, 0, 0, 2, :, :] = 0

    def no_slip_west(self, atom):
        '''At the boundary u[i-1] = 0, v[i-1] + v[i] = 0 similar for w. So v[i-1] = -v[i].'''
        atom[0, :, :, :, 0, 0, :, :] = 0
        atom[0, :, :, :, :, 1, :, :] -= atom[0, :, :, :, :, 0, :, :]
        atom[0, :, :, :, :, 0, :, :] = 0

    def no_slip_north(self, atom):
        '''At the boundary v[i] = 0, u[i] + u[i+1] = 2*U similar for w. So u[i+1] = -u[i]+2*U.'''
        atom[:, self.ny-1, :, :, :, :, 1, :] -= atom[:, self.ny-1, :, :, :, :, 2, :]
        atom[:, self.ny-1, :, :, 1, :, 1, :] = 0
        atom[:, self.ny-1, :, 1, :, :, :, :] = 0
        atom[:, self.ny-1, :, :, :, :, 2, :] = 0
        atom[:, self.ny-1, :, 1, 1, 1, 1, 1] = -1
        # TODO: Do we need this?
        atom[:, self.ny-2, :, 1, 1, :, 2, :] = 0

    def no_slip_south(self, atom):
        '''At the boundary v[i-1] = 0, u[i-1] + u[i] = 0 similar for w. So u[i-1] = -u[i].'''
        atom[:, 0, :, :, 1, :, 0, :] = 0
        atom[:, 0, :, :, :, :, 1, :] -= atom[:, 0, :, :, :, :, 0, :]
        atom[:, 0, :, :, :, :, 0, :] = 0

    def no_slip_top(self, atom):
        '''At the boundary w[i] = 0, u[i] + u[i+1] = 2*U similar for v. So u[i+1] = -u[i]+2*U.'''
        atom[:, :, self.nz-1, :, :, :, :, 1] -= atom[:, :, self.nz-1, :, :, :, :, 2]
        atom[:, :, self.nz-1, :, 2, :, :, 1] = 0
        atom[:, :, self.nz-1, 2, :, :, :, :] = 0
        atom[:, :, self.nz-1, :, :, :, :, 2] = 0
        atom[:, :, self.nz-1, 2, 2, 1, 1, 1] = -1
        # TODO: Do we need this?
        atom[:, :, self.nz-2, 2, 2, :, :, 2] = 0

    def no_slip_bottom(self, atom):
        '''At the boundary w[i-1] = 0, u[i-1] + u[i] = 0 similar for v. So u[i-1] = -u[i].'''
        atom[:, :, 0, :, 2, :, :, 0] = 0
        atom[:, :, 0, :, :, :, :, 1] -= atom[:, :, 0, :, :, :, :, 0]
        atom[:, :, 0, :, :, :, :, 0] = 0

    def free_slip_east(self, atom):
        '''At the boundary u[i] = 0, v[i] - v[i+1] = h*V similar for w. So v[i+1] = v[i]+h*V.'''
        atom[self.nx-1, :, :, :, :, 1, :, :] += atom[self.nx-1, :, :, :, :, 2, :, :]
        atom[self.nx-1, :, :, :, 0, 1, :, :] = 0
        atom[self.nx-1, :, :, 0, :, :, :, :] = 0
        atom[self.nx-1, :, :, :, :, 2, :, :] = 0
        atom[self.nx-1, :, :, 0, 0, 1, 1, 1] = -1
        # TODO: Do we need this?
        atom[self.nx-2, :, :, 0, 0, 2, :, :] = 0

    def free_slip_west(self, atom):
        '''At the boundary u[i-1] = 0, v[i-1] - v[i] = 0 similar for w. So v[i-1] = v[i].'''
        atom[0, :, :, :, 0, 0, :, :] = 0
        atom[0, :, :, :, :, 1, :, :] += atom[0, :, :, :, :, 0, :, :]
        atom[0, :, :, :, :, 0, :, :] = 0

    def free_slip_north(self, atom):
        '''At the boundary v[i] = 0, u[i] - u[i+1] = h*U similar for w. So u[i+1] = u[i]+h*U.'''
        atom[:, self.ny-1, :, :, :, :, 1, :] += atom[:, self.ny-1, :, :, :, :, 2, :]
        atom[:, self.ny-1, :, :, 1, :, 1, :] = 0
        atom[:, self.ny-1, :, 1, :, :, :, :] = 0
        atom[:, self.ny-1, :, :, :, :, 2, :] = 0
        atom[:, self.ny-1, :, 1, 1, 1, 1, 1] = -1
        # TODO: Do we need this?
        atom[:, self.ny-2, :, 1, 1, :, 2, :] = 0

    def free_slip_south(self, atom):
        '''At the boundary v[i-1] = 0, u[i-1] - u[i] = 0 similar for w. So u[i-1] = u[i].'''
        atom[:, 0, :, :, 1, :, 0, :] = 0
        atom[:, 0, :, :, :, :, 1, :] += atom[:, 0, :, :, :, :, 0, :]
        atom[:, 0, :, :, :, :, 0, :] = 0

    def free_slip_top(self, atom):
        '''At the boundary w[i] = 0, u[i] - u[i+1] = h*U similar for v. So u[i+1] = u[i]+h*U.'''
        atom[:, :, self.nz-1, :, :, :, :, 1] += atom[:, :, self.nz-1, :, :, :, :, 2]
        atom[:, :, self.nz-1, :, 2, :, :, 1] = 0
        atom[:, :, self.nz-1, 2, :, :, :, :] = 0
        atom[:, :, self.nz-1, :, :, :, :, 2] = 0
        atom[:, :, self.nz-1, 2, 2, 1, 1, 1] = -1
        # TODO: Do we need this?
        atom[:, :, self.nz-2, 2, 2, :, :, 2] = 0

    def free_slip_bottom(self, atom):
        '''At the boundary w[i-1] = 0, u[i-1] - u[i] = 0 similar for v. So u[i-1] = u[i].'''
        atom[:, :, 0, :, 2, :, :, 0] = 0
        atom[:, :, 0, :, :, :, :, 1] += atom[:, :, 0, :, :, :, :, 0]
        atom[:, :, 0, :, :, :, :, 0] = 0

    def moving_lid_east(self, atom, velocity):
        frc = self._constant_forcing_east(atom[:, :, :, :, 1, :, :, :], 1, 2 * velocity)
        if self.dim == 3 and self.nz > 1:
            frc += self._constant_forcing_east(atom[:, :, :, :, 1, :, :, :], 2, 2 * velocity)

        self.no_slip_east(atom)

        return frc

    def moving_lid_west(self, atom, velocity):
        frc = self._constant_forcing_west(atom[:, :, :, :, 1, :, :, :], 1, 2 * velocity)
        if self.dim == 3 and self.nz > 1:
            frc += self._constant_forcing_west(atom[:, :, :, :, 1, :, :, :], 2, 2 * velocity)

        self.no_slip_west(atom)

        return frc

    def moving_lid_north(self, atom, velocity):
        frc = self._constant_forcing_north(atom[:, :, :, :, 0, :, :, :], 0, 2 * velocity)

        self.no_slip_north(atom)

        return frc

    def moving_lid_south(self, atom, velocity):
        frc = self._constant_forcing_south(atom[:, :, :, :, 0, :, :, :], 0, 2 * velocity)

        self.no_slip_south(atom)

        return frc

    def moving_lid_top(self, atom, velocity):
        frc = self._constant_forcing_top(atom[:, :, :, :, 0, :, :, :], 0, 2 * velocity) + \
            self._constant_forcing_top(atom[:, :, :, :, 0, :, :, :], 1, 2 * velocity)

        self.no_slip_top(atom)

        return frc

    def moving_lid_bottom(self, atom, velocity):
        frc = self._constant_forcing_bottom(atom[:, :, :, :, 0, :, :, :], 0, 2 * velocity) + \
            self._constant_forcing_bottom(atom[:, :, :, :, 0, :, :, :], 1, 2 * velocity)

        self.no_slip_bottom(atom)

        return frc

# TODO: These methods are untested with nonzero heatflux and stretched grids

    def temperature_east(self, atom, temperature):
        '''T[i] + T[i+1] = 2 * Tb'''
        frc = self._constant_forcing_east(atom[:, :, :, :, self.dim+1, :, :, :], self.dim+1, 2 * temperature)
        atom[self.nx-1, :, :, self.dim+1, self.dim+1, 1, :, :] -= atom[self.nx-1, :, :, self.dim+1, self.dim+1, 2, :, :]
        atom[self.nx-1, :, :, self.dim+1, self.dim+1, 2, :, :] = 0

        return frc

    def temperature_west(self, atom, temperature):
        '''T[i] + T[i-1] = 2 * Tb'''
        frc = self._constant_forcing_west(atom[:, :, :, :, self.dim+1, :, :, :], self.dim+1, 2 * temperature)
        atom[0, :, :, self.dim+1, self.dim+1, 1, :, :] -= atom[0, :, :, self.dim+1, self.dim+1, 0, :, :]
        atom[0, :, :, self.dim+1, self.dim+1, 0, :, :] = 0

        return frc

    def temperature_north(self, atom, temperature):
        '''T[j] + T[j+1] = 2 * Tb'''
        frc = self._constant_forcing_north(atom[:, :, :, :, self.dim+1, :, :, :], self.dim+1, 2 * temperature)
        atom[:, self.ny-1, :, self.dim+1, self.dim+1, :, 1, :] -= atom[:, self.ny-1, :, self.dim+1, self.dim+1, :, 2, :]
        atom[:, self.ny-1, :, self.dim+1, self.dim+1, :, 2, :] = 0

        return frc

    def temperature_south(self, atom, temperature):
        '''T[j] + T[j-1] = 2 * Tb'''
        frc = self._constant_forcing_south(atom[:, :, :, :, self.dim+1, :, :, :], self.dim+1, 2 * temperature)
        atom[:, 0, :, self.dim+1, self.dim+1, :, 1, :] -= atom[:, 0, :, self.dim+1, self.dim+1, :, 0, :]
        atom[:, 0, :, self.dim+1, self.dim+1, :, 0, :] = 0

        return frc

    def temperature_top(self, atom, temperature):
        '''T[k] + T[k+1] = 2 * Tb'''
        frc = self._constant_forcing_top(atom[:, :, :, :, self.dim+1, :, :, :], self.dim+1, 2 * temperature)
        atom[:, :, self.nz-1, self.dim+1, self.dim+1, :, :, 1] -= atom[:, :, self.nz-1, self.dim+1, self.dim+1, :, :, 2]
        atom[:, :, self.nz-1, self.dim+1, self.dim+1, :, :, 2] = 0

        return frc

    def temperature_bottom(self, atom, temperature):
        '''T[k] + T[k-1] = 2 * Tb'''
        frc = self._constant_forcing_bottom(atom[:, :, :, :, self.dim+1, :, :, :], self.dim+1, 2 * temperature)
        atom[:, :, 0, self.dim+1, self.dim+1, :, :, 1] -= atom[:, :, 0, self.dim+1, self.dim+1, :, :, 0]
        atom[:, :, 0, self.dim+1, self.dim+1, :, :, 0] = 0

        return frc

    def heatflux_east(self, atom, heatflux, biot=0.0):
        '''T[i+1] - T[i] + h * Bi * (T[i+1] + T[i]) / 2 = h * Tbc, h = (x[i+1] - x[i-1]) / 2'''
        h = (self.x[self.nx] - self.x[self.nx-2]) / 2

        c = 1 + h * biot / 2
        frc = self._constant_forcing_east(atom[:, :, :, :, self.dim+1, :, :, :], self.dim+1, -heatflux * h / c)

        c = (1 - h * biot / 2) / c
        atom[self.nx-1, :, :, self.dim+1, self.dim+1, 1, :, :] += c * atom[self.nx-1, :, :, self.dim+1, self.dim+1, 2, :, :]
        atom[self.nx-1, :, :, self.dim+1, self.dim+1, 2, :, :] = 0

        return frc

    def heatflux_west(self, atom, heatflux, biot=0.0):
        '''T[i] - T[i-1] + h * Bi * (T[i] + T[i-1]) / 2 = h * Tbc, h = (x[i] - x[i-2]) / 2
        (west boundary does not start at x = 0)'''
        h = (self.x[0] - self.x[-2]) / 2

        c = 1 - h * biot / 2
        frc = self._constant_forcing_west(atom[:, :, :, :, self.dim+1, :, :, :], self.dim+1, -heatflux * h / c)

        c = (1 + h * biot / 2) / c
        atom[0, :, :, self.dim+1, self.dim+1, 1, :, :] += c * atom[0, :, :, self.dim+1, self.dim+1, 0, :, :]
        atom[0, :, :, self.dim+1, self.dim+1, 0, :, :] = 0

        return frc

    def heatflux_north(self, atom, heatflux, biot=0.0):
        '''T[j+1] - T[j] + h * Bi * (T[j+1] + T[j]) / 2 = h * Tbc, h = (y[j+1] - y[j-1]) / 2'''
        h = (self.y[self.ny] - self.y[self.ny-2]) / 2

        c = 1 + h * biot / 2
        frc = self._constant_forcing_north(atom[:, :, :, :, self.dim+1, :, :, :], self.dim+1, -heatflux * h / c)

        c = (1 - h * biot / 2) / c
        atom[:, self.ny-1, :, self.dim+1, self.dim+1, :, 1, :] += c * atom[:, self.ny-1, :, self.dim+1, self.dim+1, :, 2, :]
        atom[:, self.ny-1, :, self.dim+1, self.dim+1, :, 2, :] = 0

        return frc

    def heatflux_south(self, atom, heatflux, biot=0.0):
        '''T[j] - T[j-1] + h * Bi * (T[j] + T[j-1]) / 2 = h * Tbc, h = (y[j] - y[j-2]) / 2
        (south boundary does not start at y = 0)'''
        h = (self.y[0] - self.y[-2]) / 2

        c = 1 - h * biot / 2
        frc = self._constant_forcing_south(atom[:, :, :, :, self.dim+1, :, :, :], self.dim+1, -heatflux * h / c)

        c = (1 + h * biot / 2) / c
        atom[:, 0, :, self.dim+1, self.dim+1, :, 1, :] += c * atom[:, 0, :, self.dim+1, self.dim+1, :, 0, :]
        atom[:, 0, :, self.dim+1, self.dim+1, :, 0, :] = 0

        return frc

    def heatflux_top(self, atom, heatflux, biot=0.0):
        '''T[k+1] - T[k] + h * Bi * (T[k+1] + T[k]) / 2 = h * Tbc, h = (z[k+1] - z[k-1]) / 2'''
        h = (self.z[self.nz] - self.z[self.nz-2]) / 2

        c = 1 + h * biot / 2
        frc = self._constant_forcing_top(atom[:, :, :, :, self.dim+1, :, :, :], self.dim+1, -heatflux * h / c)

        c = (1 - h * biot / 2) / c
        atom[:, :, self.nz-1, self.dim+1, self.dim+1, :, :, 1] += c * atom[:, :, self.nz-1, self.dim+1, self.dim+1, :, :, 2]
        atom[:, :, self.nz-1, self.dim+1, self.dim+1, :, :, 2] = 0

        return frc

    def heatflux_bottom(self, atom, heatflux, biot=0.0):
        '''T[k] - T[k-1] + h * Bi * (T[k] + T[k-1]) / 2 = h * Tbc, h = (z[k] - z[k-2]) / 2
        (bottom boundary does not start at z = 0)'''
        h = (self.z[0] - self.z[-2]) / 2

        c = 1 - h * biot / 2
        frc = self._constant_forcing_bottom(atom[:, :, :, :, self.dim+1, :, :, :], self.dim+1, -heatflux * h / c)

        c = (1 + h * biot / 2) / c
        atom[:, :, 0, self.dim+1, self.dim+1, :, :, 1] += c * atom[:, :, 0, self.dim+1, self.dim+1, :, :, 0]
        atom[:, :, 0, self.dim+1, self.dim+1, :, :, 0] = 0

        return frc

    def _constant_forcing(self, atom, nx, ny, var, value):
        frc = numpy.zeros((nx, ny, self.dof))
        for j in range(ny):
            for i in range(nx):
                for y in range(3):
                    for x in range(3):
                        frc[i, j, var] += atom[i, j, var, x, y] * value
        return frc

    def _constant_forcing_east(self, atom, var, value):
        frc = numpy.zeros((self.nx, self.ny, self.nz, self.dof))
        frc[self.nx-1, :, :, :] = self._constant_forcing(atom[self.nx-1, :, :, :, 2, :, :], self.ny, self.nz, var, value)
        return create_state_vec(frc, self.nx, self.ny, self.nz, self.dof)

    def _constant_forcing_west(self, atom, var, value):
        frc = numpy.zeros((self.nx, self.ny, self.nz, self.dof))
        frc[0, :, :, :] = self._constant_forcing(atom[0, :, :, :, 0, :, :], self.ny, self.nz, var, value)
        return create_state_vec(frc, self.nx, self.ny, self.nz, self.dof)

    def _constant_forcing_north(self, atom, var, value):
        frc = numpy.zeros((self.nx, self.ny, self.nz, self.dof))
        frc[:, self.ny-1, :, :] = self._constant_forcing(atom[:, self.ny-1, :, :, :, 2, :], self.nx, self.nz, var, value)
        return create_state_vec(frc, self.nx, self.ny, self.nz, self.dof)

    def _constant_forcing_south(self, atom, var, value):
        frc = numpy.zeros((self.nx, self.ny, self.nz, self.dof))
        frc[:, 0, :, :] = self._constant_forcing(atom[:, 0, :, :, :, 0, :], self.nx, self.nz, var, value)
        return create_state_vec(frc, self.nx, self.ny, self.nz, self.dof)

    def _constant_forcing_top(self, atom, var, value):
        frc = numpy.zeros((self.nx, self.ny, self.nz, self.dof))
        frc[:, :, self.nz-1, :] = self._constant_forcing(atom[:, :, self.nz-1, :, :, :, 2], self.nx, self.ny, var, value)
        return create_state_vec(frc, self.nx, self.ny, self.nz, self.dof)

    def _constant_forcing_bottom(self, atom, var, value):
        frc = numpy.zeros((self.nx, self.ny, self.nz, self.dof))
        frc[:, :, 0, :] = self._constant_forcing(atom[:, :, 0, :, :, :, 0], self.nx, self.ny, var, value)
        return create_state_vec(frc, self.nx, self.ny, self.nz, self.dof)
