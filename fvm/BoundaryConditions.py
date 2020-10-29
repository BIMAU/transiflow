import numpy

from .utils import create_state_vec

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
        atom[:, self.ny-1, :, 1, :, :, :, :] = 0
        atom[:, self.ny-1, :, :, :, :, 2, :] = 0
        atom[:, self.ny-1, :, 1, 1, 1, 1, 1] = -1
        # TODO: Do we need this?
        atom[:, self.ny-2, :, 1, 1, :, 2, :] = 0

    def dirichlet_south(self, atom):
        # At the boundary v[i-1] = 0, u[i-1] + u[i] = 0 similar for w. So u[i-1] = -u[i].
        atom[:, 0, :, :, 1, :, 0, :] = 0
        atom[:, 0, :, :, :, :, 1, :] -= atom[:, 0, :, :, :, :, 0, :]
        atom[:, 0, :, :, :, :, 0, :] = 0

    def dirichlet_top(self, atom):
        # At the boundary w[i] = 0, u[i] + u[i+1] = 2*U similar for v. So u[i+1] = -u[i]+2*U.
        atom[:, :, self.nz-1, :, :, :, :, 1] -= atom[:, :, self.nz-1, :, :, :, :, 2]
        atom[:, :, self.nz-1, :, 2, :, :, 1] = 0
        atom[:, :, self.nz-1, 2, :, :, :, :] = 0
        atom[:, :, self.nz-1, :, :, :, :, 2] = 0
        atom[:, :, self.nz-1, 2, 2, 1, 1, 1] = -1
        # TODO: Do we need this?
        atom[:, :, self.nz-2, 2, 2, :, :, 2] = 0

    def dirichlet_bottom(self, atom):
        # At the boundary w[i-1] = 0, u[i-1] + u[i] = 0 similar for v. So u[i-1] = -u[i].
        atom[:, :, 0, :, 2, :, :, 0] = 0
        atom[:, :, 0, :, :, :, :, 1] -= atom[:, :, 0, :, :, :, :, 0]
        atom[:, :, 0, :, :, :, :, 0] = 0

    def moving_lid_north(self, atom, velocity):
        frc = self._constant_forcing_north(atom[:, :, :, :, 0, :, :, :], 0, 2 * velocity) + \
            self._constant_forcing_north(atom[:, :, :, :, 0, :, :, :], 2, 2 * velocity)

        self.dirichlet_north(atom)

        return frc

    def moving_lid_top(self, atom, velocity):
        frc = self._constant_forcing_top(atom[:, :, :, :, 0, :, :, :], 0, 2 * velocity) + \
            self._constant_forcing_top(atom[:, :, :, :, 0, :, :, :], 1, 2 * velocity)

        self.dirichlet_top(atom)

        return frc

    def temperature_east(self, atom, temperature):
        '''T[i] + T[i+1] = 2 * Tb'''
        frc = self._constant_forcing_east(atom[:, :, :, :, 4, :, :, :], 4, 2 * temperature)

        self.dirichlet_east(atom)

        return frc

    def temperature_west(self, atom, temperature):
        '''T[i] + T[i-1] = 2 * Tb'''
        frc = self._constant_forcing_west(atom[:, :, :, :, 4, :, :, :], 4, 2 * temperature)

        self.dirichlet_west(atom)

        return frc

# TODO: These methods are untested with nonzero heatflux

    def heatflux_east(self, atom, x, y, z, heatflux):
        '''T[i+1] - T[i] = h * Tbc, h = (x[i+1] - x[i-1]) / 2'''
        frc = self._constant_forcing_east(atom[:, :, :, :, 4, :, :, :], 4, - heatflux * (x[self.nx+1] - x[self.nx-1]) / 2)

        atom[self.nx-1, :, :, 4, 4, 1, :, :] += 2 * atom[self.nx-1, :, :, 4, 4, 2, :, :]
        self.dirichlet_east(atom)

        return frc

    def heatflux_west(self, atom, x, y, z, heatflux):
        '''T[i] - T[i-1] = h * Tbc, h = (x[i] - x[i-2]) / 2 (west boundary does not start at x = 0)'''
        frc = self._constant_forcing_west(atom[:, :, :, :, 4, :, :, :], 4, - heatflux * (x[0] - x[-2]) / 2)

        atom[0, :, :, 4, 4, 1, :, :] += 2 * atom[0, :, :, 4, 4, 0, :, :]
        self.dirichlet_west(atom)

        return frc

    def heatflux_north(self, atom, x, y, z, heatflux):
        '''T[j+1] - T[j] = h * Tbc, h = (y[j+1] - y[j-1]) / 2'''
        frc = self._constant_forcing_north(atom[:, :, :, :, 4, :, :, :], 4, - heatflux * (y[self.nx+1] - y[self.nx-1]) / 2)

        atom[:, self.ny-1, :, 4, 4, :, 1, :] += 2 * atom[:, self.ny-1, :, 4, 4, :, 2, :]
        self.dirichlet_north(atom)

        return frc

    def heatflux_south(self, atom, x, y, z, heatflux):
        '''T[j] - T[j-1] = h * Tbc, h = (y[j] - y[j-2]) / 2 (south boundary does not start at y = 0)'''
        frc = self._constant_forcing_south(atom[:, :, :, :, 4, :, :, :], 4, - heatflux * (y[0] - y[-2]) / 2)

        atom[:, 0, :, 4, 4, :, 1, :] += 2 * atom[:, 0, :, 4, 4, :, 0, :]
        self.dirichlet_south(atom)

        return frc

    def heatflux_top(self, atom, x, y, z, heatflux):
        '''T[k+1] - T[k] = h * Tbc, h = (z[k+1] - z[k-1]) / 2'''
        frc = self._constant_forcing_top(atom[:, :, :, :, 4, :, :, :], 4, - heatflux * (z[self.nx+1] - z[self.nx-1]) / 2)

        atom[:, :, self.nz-1, 4, 4, :, :, 1] += 2 * atom[:, :, self.nz-1, 4, 4, :, :, 2]
        self.dirichlet_top(atom)

        return frc

    def heatflux_bottom(self, atom, x, y, z, heatflux):
        '''T[k] - T[k-1] = h * Tbc, h = (z[k] - z[k-2]) / 2 (bottom boundary does not start at z = 0)'''
        frc = self._constant_forcing_bottom(atom[:, :, :, :, 4, :, :, :], 4, - heatflux * (z[0] - z[-2]) / 2)

        atom[:, :, 0, 4, 4, :, :, 1] += 2 * atom[:, :, 0, 4, 4, :, :, 0]
        self.dirichlet_bottom(atom)

        return frc

    def _constant_forcing(self, atom, nx, ny, var, value):
        frc = numpy.zeros([nx, ny, self.dof])
        for j in range(ny):
            for i in range(nx):
                for y in range(3):
                    for x in range(3):
                        frc[i, j, var] += atom[i, j, var, x, y] * value
        return frc

    def _constant_forcing_east(self, atom, var, value):
        frc = numpy.zeros([self.nx, self.ny, self.nz, self.dof])
        frc[self.nx-1, :, :] = self._constant_forcing(atom[self.nx-1, :, :, :, 2, :, :], self.ny, self.nz, var, value)
        return create_state_vec(frc, self.nx, self.ny, self.nz, self.dof)

    def _constant_forcing_west(self, atom, var, value):
        frc = numpy.zeros([self.nx, self.ny, self.nz, self.dof])
        frc[0, :, :] = self._constant_forcing(atom[0, :, :, :, 0, :, :], self.ny, self.nz, var, value)
        return create_state_vec(frc, self.nx, self.ny, self.nz, self.dof)

    def _constant_forcing_north(self, atom, var, value):
        frc = numpy.zeros([self.nx, self.ny, self.nz, self.dof])
        frc[:, self.ny-1, :] = self._constant_forcing(atom[:, self.ny-1, :, :, :, 2, :], self.nx, self.nz, var, value)
        return create_state_vec(frc, self.nx, self.ny, self.nz, self.dof)

    def _constant_forcing_south(self, atom, var, value):
        frc = numpy.zeros([self.nx, self.ny, self.nz, self.dof])
        frc[:, 0, :] = self._constant_forcing(atom[:, 0, :, :, :, 0, :], self.nx, self.nz, var, value)
        return create_state_vec(frc, self.nx, self.ny, self.nz, self.dof)

    def _constant_forcing_top(self, atom, var, value):
        frc = numpy.zeros([self.nx, self.ny, self.nz, self.dof])
        frc[:, :, self.nz-1] = self._constant_forcing(atom[:, :, self.nz-1, :, :, :, 2], self.nx, self.ny, var, value)
        return create_state_vec(frc, self.nx, self.ny, self.nz, self.dof)

    def _constant_forcing_bottom(self, atom, var, value):
        frc = numpy.zeros([self.nx, self.ny, self.nz, self.dof])
        frc[:, :, 0] = self._constant_forcing(atom[:, :, 0, :, :, :, 0], self.nx, self.ny, var, value)
        return create_state_vec(frc, self.nx, self.ny, self.nz, self.dof)
