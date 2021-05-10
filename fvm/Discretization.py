import numpy
import copy

from fvm import utils
from fvm import BoundaryConditions
from fvm import CrsMatrix

class Discretization:

    def __init__(self, parameters, nx, ny, nz, dim, dof, x=None, y=None, z=None):
        self.parameters = copy.copy(parameters)

        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.dim = dim
        self.dof = dof

        if 'Grid Stretching Factor' in self.parameters.keys():
            self.x = utils.create_stretched_coordinate_vector(
                self.parameters.get('xmin', 0.0), self.parameters.get('xmax', 1.0), self.nx,
                self.parameters.get('Grid Stretching Factor', 1.5)) if x is None else x
            self.y = utils.create_stretched_coordinate_vector(
                self.parameters.get('ymin', 0.0), self.parameters.get('ymax', 1.0), self.ny,
                self.parameters.get('Grid Stretching Factor', 1.5)) if y is None else y

            # TODO: Maybe force this if dim = 2?
            self.z = utils.create_stretched_coordinate_vector(
                self.parameters.get('zmin', 0.0), self.parameters.get('zmax', 1.0), self.nz,
                self.parameters.get('Grid Stretching Factor', 1.5)) if z is None else z
        else:
            self.x = utils.create_uniform_coordinate_vector(
                self.parameters.get('xmin', 0.0), self.parameters.get('xmax', 1.0), self.nx) if x is None else x
            self.y = utils.create_uniform_coordinate_vector(
                self.parameters.get('ymin', 0.0), self.parameters.get('ymax', 1.0), self.ny) if y is None else y

            # TODO: Maybe force this if dim = 2?
            self.z = utils.create_uniform_coordinate_vector(
                self.parameters.get('zmin', 0.0), self.parameters.get('zmax', 1.0), self.nz) if z is None else z

        self.frc = None
        self.atom = None
        self.recompute_linear_part = True

    def set_parameter(self, name, value):
        self.parameters[name] = value
        self.recompute_linear_part = True

    def get_parameter(self, name, default=0):
        return self.parameters.get(name, default)

    def linear_part(self):
        if self.dim == 2:
            return self._linear_part_2D()
        return self._linear_part_3D()

    def _linear_part_2D(self):
        Re = self.get_parameter('Reynolds Number')
        Ra = self.get_parameter('Rayleigh Number')

        if Re == 0:
            Re = 1

        atom = 1 / Re * (self.u_xx() + self.u_yy()
                         + self.v_xx() + self.v_yy()) \
            - (self.p_x() + self.p_y()) \
            + self.div()

        if self.dof > 3:
            atom += self.T_xx() + self.T_yy()
            atom += Ra * self.forward_average_T_y()

        return atom

    def _linear_part_3D(self):
        Re = self.get_parameter('Reynolds Number')
        Ra = self.get_parameter('Rayleigh Number')

        if Re == 0:
            Re = 1

        atom = 1 / Re * (self.u_xx() + self.u_yy() + self.u_zz()
                         + self.v_xx() + self.v_yy() + self.v_zz()
                         + self.w_xx() + self.w_yy() + self.w_zz()) \
            - (self.p_x() + self.p_y() + self.p_z()) \
            + self.div()

        if self.dof > 4:
            atom += self.T_xx() + self.T_yy() + self.T_zz()
            if self.nz > 1:
                atom += Ra * self.forward_average_T_z()
            else:
                atom += Ra * self.forward_average_T_y()

        return atom

    def nonlinear_part(self, state):
        state_mtx = utils.create_state_mtx(state, self.nx, self.ny, self.nz, self.dof)

        Re = self.get_parameter('Reynolds Number')
        if Re == 0:
            state_mtx[:, :, :, :] = 0

        if self.dim == 2:
            return self.convection_2D(state_mtx)
        return self.convection_3D(state_mtx)

    def rhs(self, state):
        if self.recompute_linear_part:
            self.atom = self.linear_part()
            self.frc = self.boundaries(self.atom)
            self.recompute_linear_part = False

        atomJ, atomF = self.nonlinear_part(state)
        atomF += self.atom

        return self.assemble_rhs(state, atomF) + self.frc

    def jacobian(self, state):
        if self.recompute_linear_part:
            self.atom = self.linear_part()
            self.frc = self.boundaries(self.atom)
            self.recompute_linear_part = False

        atomJ, atomF = self.nonlinear_part(state)
        atomJ += self.atom

        return self.assemble_jacobian(atomJ)

    def mass_matrix(self):
        atom = self.mass_x() + self.mass_y()
        if self.dim == 3:
            atom += self.mass_z()
        if self.dof > self.dim + 1:
            Pr = self.get_parameter('Prandtl Number', 1.0)
            atom /= Pr
            atom += self.mass_T()
        return self.assemble_mass_matrix(atom)

    def assemble_rhs(self, state, atom):
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

        # Put the state in shifted matrix form
        state_mtx = numpy.zeros([self.nx+2, self.ny+2, self.nz+2, self.dof])
        state_mtx[1:self.nx+1, 1:self.ny+1, 1:self.nz+1, :] = utils.create_state_mtx(
            state, self.nx, self.ny, self.nz, self.dof)

        # Add extra borders for periodic boundary conditions
        state_mtx[0, 1:self.ny+1, 1:self.nz+1, :] = state_mtx[self.nx, 1:self.ny+1, 1:self.nz+1, :]
        state_mtx[self.nx+1, 1:self.ny+1, 1:self.nz+1, :] = state_mtx[1, 1:self.ny+1, 1:self.nz+1, :]
        state_mtx[1:self.nx+1, 0, 1:self.nz+1, :] = state_mtx[1:self.nx+1, self.ny, 1:self.nz+1, :]
        state_mtx[1:self.nx+1, self.ny+1, 1:self.nz+1, :] = state_mtx[1:self.nx+1, 1, 1:self.nz+1, :]
        state_mtx[1:self.nx+1, 1:self.ny+1, 0, :] = state_mtx[1:self.nx+1, 1:self.ny+1, self.nz, :]
        state_mtx[1:self.nx+1, 1:self.ny+1, self.nz+1, :] = state_mtx[1:self.nx+1, 1:self.ny+1, 1, :]

        # Add up all contributions without iterating over the domain
        out_mtx = numpy.zeros([self.nx, self.ny, self.nz, self.dof])
        for k in range(3):
            for j in range(3):
                for i in range(3):
                    for d1 in range(self.dof):
                        for d2 in range(self.dof):
                            out_mtx[:, :, :, d1] += atom[:, :, :, d1, d2, i, j, k] \
                                * state_mtx[i:(i+self.nx), j:(j+self.ny), k:(k+self.nz), d2]

        return utils.create_state_vec(out_mtx, self.nx, self.ny, self.nz, self.dof)

    def assemble_jacobian(self, atom):
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
        n = self.nx * self.ny * self.nz * self.dof
        coA = numpy.zeros(27*n)
        jcoA = numpy.zeros(27*n, dtype=int)
        begA = numpy.zeros(n+1, dtype=int)

        # Check where values are nonzero in the atoms
        configs = []
        for z in range(3):
            for y in range(3):
                for x in range(3):
                    for d2 in range(self.dof):
                        if numpy.any(atom[:, :, :, :, d2, x, y, z]):
                            configs.append([d2, x, y, z])

        # Iterate only over configurations with values in there
        for k in range(self.nz):
            for j in range(self.ny):
                for i in range(self.nx):
                    for d1 in range(self.dof):
                        for config in configs:
                            if abs(atom[i, j, k, d1, config[0], config[1], config[2], config[3]]) > 1e-14:
                                jcoA[idx] = ((i + config[1] - 1) % self.nx) * self.dof \
                                    + ((j + config[2] - 1) % self.ny) * self.nx * self.dof + \
                                    + ((k + config[3] - 1) % self.nz) * self.nx * self.ny * self.dof + config[0]
                                coA[idx] = atom[i, j, k, d1, config[0], config[1], config[2], config[3]]
                                idx += 1
                        row += 1
                        begA[row] = idx
        return CrsMatrix(coA, jcoA, begA)

    def assemble_mass_matrix(self, atom):
        ''' Assemble the mass matrix.'''

        row = 0
        idx = 0
        n = self.nx * self.ny * self.nz * self.dof
        coA = numpy.zeros(n)
        jcoA = numpy.zeros(n, dtype=int)
        begA = numpy.zeros(n+1, dtype=int)

        for k in range(self.nz):
            for j in range(self.ny):
                for i in range(self.nx):
                    for d1 in range(self.dof):
                        if abs(atom[i, j, k, d1]) > 1e-14:
                            jcoA[idx] = (i + (j + k * self.ny) * self.nx) * self.dof + d1
                            coA[idx] = atom[i, j, k, d1]
                            idx += 1
                        row += 1
                        begA[row] = idx
        return CrsMatrix(coA, jcoA, begA)

    @staticmethod
    def _problem_type_equals(first, second):
        return first.lower() == second.lower()

    def boundaries(self, atom):
        boundary_conditions = BoundaryConditions(self.nx, self.ny, self.nz, self.dim, self.dof, self.x, self.y, self.z)
        problem_type = self.get_parameter('Problem Type', 'Lid-driven cavity')

        frc = numpy.zeros(self.nx * self.ny * self.nz * self.dof)

        if Discretization._problem_type_equals(problem_type, 'Lid-driven cavity'):
            v = self.get_parameter('Lid Velocity', 1)
            boundary_conditions.no_slip_east(atom)
            boundary_conditions.no_slip_west(atom)

            boundary_conditions.no_slip_south(atom)
            if self.dim == 2 or self.nz <= 1:
                frc += boundary_conditions.moving_lid_north(atom, v)
                return frc

            boundary_conditions.no_slip_north(atom)

            boundary_conditions.no_slip_bottom(atom)
            frc += boundary_conditions.moving_lid_top(atom, v)
        elif Discretization._problem_type_equals(problem_type, 'Rayleigh-Benard'):
            frc += boundary_conditions.heatflux_east(atom, 0)
            frc += boundary_conditions.heatflux_west(atom, 0)
            boundary_conditions.no_slip_east(atom)
            boundary_conditions.no_slip_west(atom)

            if self.dim == 2 or self.nz <= 1:
                Bi = self.get_parameter('Biot Number')
                frc += boundary_conditions.heatflux_north(atom, 0, Bi)
                frc += boundary_conditions.temperature_south(atom, 1)
                boundary_conditions.free_slip_north(atom)
                boundary_conditions.no_slip_south(atom)
                return frc

            frc += boundary_conditions.heatflux_north(atom, 0)
            frc += boundary_conditions.heatflux_south(atom, 0)
            boundary_conditions.no_slip_north(atom)
            boundary_conditions.no_slip_south(atom)

            frc += boundary_conditions.temperature_top(atom, 0)
            frc += boundary_conditions.temperature_bottom(atom, 0)
            boundary_conditions.no_slip_top(atom)
            boundary_conditions.no_slip_bottom(atom)
        elif Discretization._problem_type_equals(problem_type, 'Differentially heated cavity'):
            frc += boundary_conditions.temperature_east(atom, -1/2)
            frc += boundary_conditions.temperature_west(atom, 1/2)
            boundary_conditions.no_slip_east(atom)
            boundary_conditions.no_slip_west(atom)

            frc += boundary_conditions.heatflux_north(atom, 0)
            frc += boundary_conditions.heatflux_south(atom, 0)
            boundary_conditions.no_slip_north(atom)
            boundary_conditions.no_slip_south(atom)

            if self.dim > 2 and self.nz > 1:
                frc += boundary_conditions.heatflux_top(atom, 0)
                frc += boundary_conditions.heatflux_bottom(atom, 0)
                boundary_conditions.no_slip_top(atom)
                boundary_conditions.no_slip_bottom(atom)
        else:
            raise Exception('Invalid problem type %s' % problem_type)

        return frc

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

    def u_xx(self):
        atom = numpy.zeros([self.nx, self.ny, self.nz, self.dof, self.dof, 3, 3, 3])
        for i in range(self.nx):
            for j in range(self.ny):
                for k in range(self.nz):
                    Discretization._u_xx(atom[i, j, k, 0, 0, :, 1, 1], i, j, k, self.x, self.y, self.z)
        return atom

    def v_yy(self):
        atom = numpy.zeros([self.nx, self.ny, self.nz, self.dof, self.dof, 3, 3, 3])
        for i in range(self.nx):
            for j in range(self.ny):
                for k in range(self.nz):
                    Discretization._u_xx(atom[i, j, k, 1, 1, 1, :, 1], j, i, k, self.y, self.x, self.z)
        return atom

    def w_zz(self):
        atom = numpy.zeros([self.nx, self.ny, self.nz, self.dof, self.dof, 3, 3, 3])
        for i in range(self.nx):
            for j in range(self.ny):
                for k in range(self.nz):
                    Discretization._u_xx(atom[i, j, k, 2, 2, 1, 1, :], k, j, i, self.z, self.y, self.x)
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

    def u_yy(self):
        atom = numpy.zeros([self.nx, self.ny, self.nz, self.dof, self.dof, 3, 3, 3])
        for i in range(self.nx):
            for j in range(self.ny):
                for k in range(self.nz):
                    Discretization._u_yy(atom[i, j, k, 0, 0, 1, :, 1], i, j, k, self.x, self.y, self.z)
        return atom

    def v_xx(self):
        atom = numpy.zeros([self.nx, self.ny, self.nz, self.dof, self.dof, 3, 3, 3])
        for i in range(self.nx):
            for j in range(self.ny):
                for k in range(self.nz):
                    Discretization._u_yy(atom[i, j, k, 1, 1, :, 1, 1], j, i, k, self.y, self.x, self.z)
        return atom

    def w_yy(self):
        atom = numpy.zeros([self.nx, self.ny, self.nz, self.dof, self.dof, 3, 3, 3])
        for i in range(self.nx):
            for j in range(self.ny):
                for k in range(self.nz):
                    Discretization._u_yy(atom[i, j, k, 2, 2, 1, :, 1], k, j, i, self.z, self.y, self.x)
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

    def u_zz(self):
        atom = numpy.zeros([self.nx, self.ny, self.nz, self.dof, self.dof, 3, 3, 3])
        for i in range(self.nx):
            for j in range(self.ny):
                for k in range(self.nz):
                    Discretization._u_zz(atom[i, j, k, 0, 0, 1, 1, :], i, j, k, self.x, self.y, self.z)
        return atom

    def v_zz(self):
        atom = numpy.zeros([self.nx, self.ny, self.nz, self.dof, self.dof, 3, 3, 3])
        for i in range(self.nx):
            for j in range(self.ny):
                for k in range(self.nz):
                    Discretization._u_zz(atom[i, j, k, 1, 1, 1, 1, :], j, i, k, self.y, self.x, self.z)
        return atom

    def w_xx(self):
        atom = numpy.zeros([self.nx, self.ny, self.nz, self.dof, self.dof, 3, 3, 3])
        for i in range(self.nx):
            for j in range(self.ny):
                for k in range(self.nz):
                    Discretization._u_zz(atom[i, j, k, 2, 2, :, 1, 1], k, j, i, self.z, self.y, self.x)
        return atom

    @staticmethod
    def _T_xx(atom, i, j, k, x, y, z):
        # distance between T[i] and T[i-1]
        dx = (x[i] - x[i-2]) / 2
        # distance between T[i+1] and T[i]
        dxp1 = (x[i+1] - x[i-1]) / 2
        # volume size in the y direction
        dy = y[j] - y[j-1]
        # volume size in the z direction
        dz = z[k] - z[k-1]

        # second order finite difference
        atom[0] = 1 / dx * dy * dz
        atom[2] = 1 / dxp1 * dy * dz
        atom[1] = -atom[0] - atom[2]

    def T_xx(self):
        atom = numpy.zeros([self.nx, self.ny, self.nz, self.dof, self.dof, 3, 3, 3])
        for i in range(self.nx):
            for j in range(self.ny):
                for k in range(self.nz):
                    Discretization._T_xx(atom[i, j, k, self.dim+1, self.dim+1, :, 1, 1], i, j, k, self.x, self.y, self.z)
        return atom

    def T_yy(self):
        atom = numpy.zeros([self.nx, self.ny, self.nz, self.dof, self.dof, 3, 3, 3])
        for i in range(self.nx):
            for j in range(self.ny):
                for k in range(self.nz):
                    Discretization._T_xx(atom[i, j, k, self.dim+1, self.dim+1, 1, :, 1], j, i, k, self.y, self.x, self.z)
        return atom

    def T_zz(self):
        atom = numpy.zeros([self.nx, self.ny, self.nz, self.dof, self.dof, 3, 3, 3])
        for i in range(self.nx):
            for j in range(self.ny):
                for k in range(self.nz):
                    Discretization._T_xx(atom[i, j, k, self.dim+1, self.dim+1, 1, 1, :], k, j, i, self.z, self.y, self.x)
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

    def p_x(self):
        atom = numpy.zeros([self.nx, self.ny, self.nz, self.dof, self.dof, 3, 3, 3])
        for i in range(self.nx):
            for j in range(self.ny):
                for k in range(self.nz):
                    Discretization._forward_u_x(atom[i, j, k, 0, self.dim, :, 1, 1], i, j, k, self.x, self.y, self.z)
        return atom

    def p_y(self):
        atom = numpy.zeros([self.nx, self.ny, self.nz, self.dof, self.dof, 3, 3, 3])
        for i in range(self.nx):
            for j in range(self.ny):
                for k in range(self.nz):
                    Discretization._forward_u_x(atom[i, j, k, 1, self.dim, 1, :, 1], j, i, k, self.y, self.x, self.z)
        return atom

    def p_z(self):
        atom = numpy.zeros([self.nx, self.ny, self.nz, self.dof, self.dof, 3, 3, 3])
        for i in range(self.nx):
            for j in range(self.ny):
                for k in range(self.nz):
                    Discretization._forward_u_x(atom[i, j, k, 2, self.dim, 1, 1, :], k, j, i, self.z, self.y, self.x)
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

    def u_x(self):
        atom = numpy.zeros([self.nx, self.ny, self.nz, self.dof, self.dof, 3, 3, 3])
        for i in range(self.nx):
            for j in range(self.ny):
                for k in range(self.nz):
                    Discretization._backward_u_x(atom[i, j, k, self.dim, 0, :, 1, 1], i, j, k, self.x, self.y, self.z)
        return atom

    def v_y(self):
        atom = numpy.zeros([self.nx, self.ny, self.nz, self.dof, self.dof, 3, 3, 3])
        for i in range(self.nx):
            for j in range(self.ny):
                for k in range(self.nz):
                    Discretization._backward_u_x(atom[i, j, k, self.dim, 1, 1, :, 1], j, i, k, self.y, self.x, self.z)
        return atom

    def w_z(self):
        atom = numpy.zeros([self.nx, self.ny, self.nz, self.dof, self.dof, 3, 3, 3])
        for i in range(self.nx):
            for j in range(self.ny):
                for k in range(self.nz):
                    Discretization._backward_u_x(atom[i, j, k, self.dim, 2, 1, 1, :], k, j, i, self.z, self.y, self.x)
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
        dy = y[j] - y[j-1]

        # backward difference
        atom[1] = dx * dy
        atom[0] = -atom[1]

    def div(self):
        if self.dim == 2:
            return self.u_x() + self.v_y()
        return self.u_x() + self.v_y() + self.w_z()

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

    def forward_average_T_y(self):
        atom = numpy.zeros([self.nx, self.ny, self.nz, self.dof, self.dof, 3, 3, 3])
        for i in range(self.nx):
            for j in range(self.ny):
                for k in range(self.nz):
                    Discretization._forward_average_x(atom[i, j, k, 1, self.dim+1, 1, :, 1], j, i, k, self.y, self.x, self.z)
        return atom

    def forward_average_T_z(self):
        atom = numpy.zeros([self.nx, self.ny, self.nz, self.dof, self.dof, 3, 3, 3])
        for i in range(self.nx):
            for j in range(self.ny):
                for k in range(self.nz):
                    Discretization._forward_average_x(atom[i, j, k, 2, self.dim+1, 1, 1, :], k, j, i, self.z, self.y, self.x)
        return atom

    @staticmethod
    def _backward_average_x(atom, i, j, k, x, y, z):
        # volume size in the x direction
        dx = (x[i+1] - x[i-1]) / 2
        # volume size in the y direction
        dy = y[j] - y[j-1]
        # volume size in the z direction
        dz = z[k] - z[k-1]

        # forward average
        atom[0] = dx * dy * dz / 2
        atom[1] = atom[0]

    def backward_average_v_y(self):
        atom = numpy.zeros([self.nx, self.ny, self.nz, self.dof, self.dof, 3, 3, 3])
        for i in range(self.nx):
            for j in range(self.ny):
                for k in range(self.nz):
                    Discretization._backward_average_x(atom[i, j, k, self.dim+1, 1, 1, :, 1], j, i, k, self.y, self.x, self.z)
        return atom

    def backward_average_w_z(self):
        atom = numpy.zeros([self.nx, self.ny, self.nz, self.dof, self.dof, 3, 3, 3])
        for i in range(self.nx):
            for j in range(self.ny):
                for k in range(self.nz):
                    Discretization._backward_average_x(atom[i, j, k, self.dim+1, 2, 1, 1, :], k, j, i, self.z, self.y, self.x)
        return atom

    @staticmethod
    def _mass_x(atom, i, j, k, x, y, z):
        # volume size in the x direction
        dx = (x[i+1] - x[i-1]) / 2
        # volume size in the y direction
        dy = y[j] - y[j-1]
        # volume size in the z direction
        dz = z[k] - z[k-1]

        atom[0] = dx * dy * dz

    def mass_x(self):
        atom = numpy.zeros([self.nx, self.ny, self.nz, self.dof])
        for i in range(self.nx):
            for j in range(self.ny):
                for k in range(self.nz):
                    Discretization._mass_x(atom[i, j, k, 0:1], i, j, k, self.x, self.y, self.z)
        return atom

    def mass_y(self):
        atom = numpy.zeros([self.nx, self.ny, self.nz, self.dof])
        for i in range(self.nx):
            for j in range(self.ny):
                for k in range(self.nz):
                    Discretization._mass_x(atom[i, j, k, 1:2], j, i, k, self.y, self.x, self.z)
        return atom

    def mass_z(self):
        atom = numpy.zeros([self.nx, self.ny, self.nz, self.dof])
        for i in range(self.nx):
            for j in range(self.ny):
                for k in range(self.nz):
                    Discretization._mass_x(atom[i, j, k, 2:3], k, j, i, self.z, self.y, self.x)
        return atom

    @staticmethod
    def _mass_T(atom, i, j, k, x, y, z):
        # volume size in the x direction
        dx = x[i] - x[i-1]
        # volume size in the y direction
        dy = y[j] - y[j-1]
        # volume size in the z direction
        dz = z[k] - z[k-1]

        atom[0] = dx * dy * dz

    def mass_T(self):
        atom = numpy.zeros([self.nx, self.ny, self.nz, self.dof])
        for i in range(self.nx):
            for j in range(self.ny):
                for k in range(self.nz):
                    Discretization._mass_T(atom[i, j, k, self.dim+1:self.dim+2], i, j, k, self.x, self.y, self.z)
        return atom

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
    def _convection_u_v(atomJ, atomF, averages, weighted_averages, bil, varU, varV, dim, nx, i):
        for d1 in range(3):
            i2 = i + d1 - 1

            v_x = bil[i, :, :, 2, varU, varV, d1]
            if not numpy.any(v_x):
                continue

            coef1 = weighted_averages[i2, :, :, varU, varV] * v_x
            for d2 in range(3):
                coef2 = bil[i2, :, :, 0, varV, varU, d2]
                if numpy.any(coef2):
                    idx = [1, 1, 1]
                    idx[varU] += d1 - 1
                    idx[varU] += d2 - 1
                    atomF[i, :, :, varV, varV, idx[0], idx[1], idx[2]] -= coef1 * coef2

            coef1 = averages[i2, :, :, varV, varU] * v_x
            for d2 in range(3):
                coef2 = bil[i2, :, :, 1, varU, varV, d2]
                if numpy.any(coef2):
                    idx = [1, 1, 1]
                    idx[varU] += d1 - 1
                    idx[varV if varV < dim else varU] += d2 - 1
                    atomJ[i, :, :, varV, varU, idx[0], idx[1], idx[2]] -= coef1 * coef2

    @staticmethod
    def _convection_v_u(atomJ, atomF, averages, weighted_averages, bil, varV, varU, dim, ny, j):
        for d1 in range(3):
            j2 = j + d1 - 1

            u_y = bil[:, j, :, 2, varV, varU, d1]
            if not numpy.any(u_y):
                continue

            coef1 = weighted_averages[:, j2, :, varV, varU] * u_y
            for d2 in range(3):
                coef2 = bil[:, j2, :, 0, varU, varV, d2]
                if numpy.any(coef2):
                    idx = [1, 1, 1]
                    idx[varV] += d1 - 1
                    idx[varV] += d2 - 1
                    atomF[:, j, :, varU, varU, idx[0], idx[1], idx[2]] -= coef1 * coef2

            coef1 = averages[:, j2, :, varU, varV] * u_y
            for d2 in range(3):
                coef2 = bil[:, j2, :, 1, varV, varU, d2]
                if numpy.any(coef2):
                    idx = [1, 1, 1]
                    idx[varV] += d1 - 1
                    idx[varU if varU < dim else varV] += d2 - 1
                    atomJ[:, j, :, varU, varV, idx[0], idx[1], idx[2]] -= coef1 * coef2

    @staticmethod
    def _convection_w_u(atomJ, atomF, averages, weighted_averages, bil, varW, varU, dim, nz, k):
        for d1 in range(3):
            k2 = k + d1 - 1

            u_z = bil[:, :, k, 2, varW, varU, d1]
            if not numpy.any(u_z):
                continue

            coef1 = weighted_averages[:, :, k2, varW, varU] * u_z
            for d2 in range(3):
                coef2 = bil[:, :, k2, 0, varU, varW, d2]
                if numpy.any(coef2):
                    idx = [1, 1, 1]
                    idx[varW] += d1 - 1
                    idx[varW] += d2 - 1
                    atomF[:, :, k, varU, varU, idx[0], idx[1], idx[2]] -= coef1 * coef2

            coef1 = averages[:, :, k2, varU, varW] * u_z
            for d2 in range(3):
                coef2 = bil[:, :, k2, 1, varW, varU, d2]
                if numpy.any(coef2):
                    idx = [1, 1, 1]
                    idx[varW] += d1 - 1
                    idx[varU if varU < dim else varW] += d2 - 1
                    atomJ[:, :, k, varU, varW, idx[0], idx[1], idx[2]] -= coef1 * coef2

    def convection_u_u(self, atomJ, atomF, averages, weighted_averages, bil):
        for i in range(self.nx):
            Discretization._convection_u_v(atomJ, atomF, averages, weighted_averages, bil, 0, 0, self.dim, self.nx, i)

    def convection_v_u(self, atomJ, atomF, averages, weighted_averages, bil):
        for j in range(self.ny):
            Discretization._convection_v_u(atomJ, atomF, averages, weighted_averages, bil, 1, 0, self.dim, self.ny, j)

    def convection_w_u(self, atomJ, atomF, averages, weighted_averages, bil):
        for k in range(self.nz):
            Discretization._convection_w_u(atomJ, atomF, averages, weighted_averages, bil, 2, 0, self.dim, self.nz, k)

    def convection_u_v(self, atomJ, atomF, averages, weighted_averages, bil):
        for i in range(self.nx):
            Discretization._convection_u_v(atomJ, atomF, averages, weighted_averages, bil, 0, 1, self.dim, self.nx, i)

    def convection_v_v(self, atomJ, atomF, averages, weighted_averages, bil):
        for j in range(self.ny):
            Discretization._convection_v_u(atomJ, atomF, averages, weighted_averages, bil, 1, 1, self.dim, self.ny, j)

    def convection_w_v(self, atomJ, atomF, averages, weighted_averages, bil):
        for k in range(self.nz):
            Discretization._convection_w_u(atomJ, atomF, averages, weighted_averages, bil, 2, 1, self.dim, self.nz, k)

    def convection_u_w(self, atomJ, atomF, averages, weighted_averages, bil):
        for i in range(self.nx):
            Discretization._convection_u_v(atomJ, atomF, averages, weighted_averages, bil, 0, 2, self.dim, self.nx, i)

    def convection_v_w(self, atomJ, atomF, averages, weighted_averages, bil):
        for j in range(self.ny):
            Discretization._convection_v_u(atomJ, atomF, averages, weighted_averages, bil, 1, 2, self.dim, self.ny, j)

    def convection_w_w(self, atomJ, atomF, averages, weighted_averages, bil):
        for k in range(self.nz):
            Discretization._convection_w_u(atomJ, atomF, averages, weighted_averages, bil, 2, 2, self.dim, self.nz, k)

    def convection_T_u(self, atomJ, atomF, averages, weighted_averages, bil):
        for i in range(self.nx):
            Discretization._convection_u_v(atomJ, atomF, averages, weighted_averages, bil, 0, self.dim+1, self.dim, self.nx, i)

    def convection_T_v(self, atomJ, atomF, averages, weighted_averages, bil):
        for j in range(self.ny):
            Discretization._convection_v_u(atomJ, atomF, averages, weighted_averages, bil, 1, self.dim+1, self.dim, self.ny, j)

    def convection_T_w(self, atomJ, atomF, averages, weighted_averages, bil):
        for k in range(self.nz):
            Discretization._convection_w_u(atomJ, atomF, averages, weighted_averages, bil, 2, self.dim+1, self.dim, self.nz, k)

    def convection_2D(self, state):
        bil = numpy.zeros([self.nx, self.ny, self.nz, 3, self.dof, self.dof, 3])
        averages = numpy.zeros([self.nx, self.ny, self.nz, self.dof, self.dof])
        weighted_averages = numpy.zeros([self.nx, self.ny, self.nz, self.dof, self.dof])

        convective_term = ConvectiveTerm(self.nx, self.ny, self.nz, self.dim, self.x, self.y, self.z)

        convective_term.backward_average_x(bil[:, :, :, :, 0, :, :], averages[:, :, :, 0, :],
                                           weighted_averages[:, :, :, 0, :], state[:, :, :, 0]) # tMxU
        convective_term.forward_average_x(bil[:, :, :, :, 1, :, :], averages[:, :, :, 1, :],
                                          weighted_averages[:, :, :, 1, :], state[:, :, :, 1]) # tMxV
        convective_term.forward_average_y(bil[:, :, :, :, 0, :, :], averages[:, :, :, 0, :],
                                          weighted_averages[:, :, :, 0, :], state[:, :, :, 0]) # tMyU
        convective_term.backward_average_y(bil[:, :, :, :, 1, :, :], averages[:, :, :, 1, :],
                                           weighted_averages[:, :, :, 1, :], state[:, :, :, 1]) # tMyV

        if self.dof > self.dim+1:
            convective_term.forward_average_x(bil[:, :, :, :, self.dim+1, :, :], averages[:, :, :, self.dim+1, :],
                                              weighted_averages[:, :, :, self.dim+1, :], state[:, :, :, self.dim+1]) # tMxT
            convective_term.forward_average_y(bil[:, :, :, :, self.dim+1, :, :], averages[:, :, :, self.dim+1, :],
                                              weighted_averages[:, :, :, self.dim+1, :], state[:, :, :, self.dim+1]) # tMyT
            convective_term.value_u(bil[:, :, :, :, :, self.dim+1, :], averages[:, :, :, :, self.dim+1],
                                    weighted_averages[:, :, :, :, self.dim+1], state)
            convective_term.value_v(bil[:, :, :, :, :, self.dim+1, :], averages[:, :, :, :, self.dim+1],
                                    weighted_averages[:, :, :, :, self.dim+1], state)

        convective_term.u_x(bil) # tMxUMxU
        convective_term.u_y(bil) # tMxVMyU
        convective_term.v_x(bil) # tMyUMxV
        convective_term.v_y(bil) # tMyVMyV

        if self.dof > self.dim + 1:
            convective_term.T_x(bil)
            convective_term.T_y(bil)

        convective_term.boundary_east(bil)
        convective_term.boundary_west(bil)
        convective_term.boundary_north(bil)
        convective_term.boundary_south(bil)

        atomJ = numpy.zeros([self.nx, self.ny, self.nz, self.dof, self.dof, 3, 3, 3])
        atomF = numpy.zeros([self.nx, self.ny, self.nz, self.dof, self.dof, 3, 3, 3])

        self.convection_u_u(atomJ, atomF, averages, weighted_averages, bil)
        self.convection_u_v(atomJ, atomF, averages, weighted_averages, bil)
        self.convection_v_u(atomJ, atomF, averages, weighted_averages, bil)
        self.convection_v_v(atomJ, atomF, averages, weighted_averages, bil)

        if self.dof > self.dim + 1:
            Pr = self.get_parameter('Prandtl Number', 1.0)
            atomJ /= Pr
            atomF /= Pr

            self.convection_T_u(atomJ, atomF, averages, weighted_averages, bil)
            self.convection_T_v(atomJ, atomF, averages, weighted_averages, bil)

        atomJ += atomF

        return (atomJ, atomF)

    def convection_3D(self, state):
        bil = numpy.zeros([self.nx, self.ny, self.nz, 3, self.dof, self.dof, 3])
        averages = numpy.zeros([self.nx, self.ny, self.nz, self.dof, self.dof])
        weighted_averages = numpy.zeros([self.nx, self.ny, self.nz, self.dof, self.dof])

        convective_term = ConvectiveTerm(self.nx, self.ny, self.nz, self.dim, self.x, self.y, self.z)

        convective_term.backward_average_x(bil[:, :, :, :, 0, :, :], averages[:, :, :, 0, :],
                                           weighted_averages[:, :, :, 0, :], state[:, :, :, 0]) # tMxU
        convective_term.forward_average_x(bil[:, :, :, :, 1, :, :], averages[:, :, :, 1, :],
                                          weighted_averages[:, :, :, 1, :], state[:, :, :, 1]) # tMxV
        convective_term.forward_average_x(bil[:, :, :, :, 2, :, :], averages[:, :, :, 2, :],
                                          weighted_averages[:, :, :, 2, :], state[:, :, :, 2]) # tMxW
        convective_term.forward_average_y(bil[:, :, :, :, 0, :, :], averages[:, :, :, 0, :],
                                          weighted_averages[:, :, :, 0, :], state[:, :, :, 0]) # tMyU
        convective_term.backward_average_y(bil[:, :, :, :, 1, :, :], averages[:, :, :, 1, :],
                                           weighted_averages[:, :, :, 1, :], state[:, :, :, 1]) # tMyV
        convective_term.forward_average_y(bil[:, :, :, :, 2, :, :], averages[:, :, :, 2, :],
                                          weighted_averages[:, :, :, 2, :], state[:, :, :, 2]) # tMyW
        convective_term.forward_average_z(bil[:, :, :, :, 0, :, :], averages[:, :, :, 0, :],
                                          weighted_averages[:, :, :, 0, :], state[:, :, :, 0]) # tMzU
        convective_term.forward_average_z(bil[:, :, :, :, 1, :, :], averages[:, :, :, 1, :],
                                          weighted_averages[:, :, :, 1, :], state[:, :, :, 1]) # tMzV
        convective_term.backward_average_z(bil[:, :, :, :, 2, :, :], averages[:, :, :, 2, :],
                                           weighted_averages[:, :, :, 2, :], state[:, :, :, 2]) # tMzW

        if self.dof > self.dim+1:
            convective_term.forward_average_x(bil[:, :, :, :, self.dim+1, :, :], averages[:, :, :, self.dim+1, :],
                                              weighted_averages[:, :, :, self.dim+1, :], state[:, :, :, self.dim+1]) # tMxT
            convective_term.forward_average_y(bil[:, :, :, :, self.dim+1, :, :], averages[:, :, :, self.dim+1, :],
                                              weighted_averages[:, :, :, self.dim+1, :], state[:, :, :, self.dim+1]) # tMyT
            convective_term.forward_average_z(bil[:, :, :, :, self.dim+1, :, :], averages[:, :, :, self.dim+1, :],
                                              weighted_averages[:, :, :, self.dim+1, :], state[:, :, :, self.dim+1]) # tMzT
            convective_term.value_u(bil[:, :, :, :, :, self.dim+1, :], averages[:, :, :, :, self.dim+1],
                                    weighted_averages[:, :, :, :, self.dim+1], state)
            convective_term.value_v(bil[:, :, :, :, :, self.dim+1, :], averages[:, :, :, :, self.dim+1],
                                    weighted_averages[:, :, :, :, self.dim+1], state)
            convective_term.value_w(bil[:, :, :, :, :, self.dim+1, :], averages[:, :, :, :, self.dim+1],
                                    weighted_averages[:, :, :, :, self.dim+1], state)

        convective_term.u_x(bil) # tMxUMxU
        convective_term.u_y(bil) # tMxVMyU
        convective_term.u_z(bil) # tMxWMzU
        convective_term.v_x(bil) # tMyUMxV
        convective_term.v_y(bil) # tMyVMyV
        convective_term.v_z(bil) # tMyWMzV
        convective_term.w_x(bil) # tMzUMxW
        convective_term.w_y(bil) # tMzVMyW
        convective_term.w_z(bil) # tMzWMzW

        if self.dof > self.dim + 1:
            convective_term.T_x(bil)
            convective_term.T_y(bil)
            convective_term.T_z(bil)

        convective_term.boundary_east(bil)
        convective_term.boundary_west(bil)
        convective_term.boundary_north(bil)
        convective_term.boundary_south(bil)
        convective_term.boundary_top(bil)
        convective_term.boundary_bottom(bil)

        atomJ = numpy.zeros([self.nx, self.ny, self.nz, self.dof, self.dof, 3, 3, 3])
        atomF = numpy.zeros([self.nx, self.ny, self.nz, self.dof, self.dof, 3, 3, 3])

        self.convection_u_u(atomJ, atomF, averages, weighted_averages, bil)
        self.convection_u_v(atomJ, atomF, averages, weighted_averages, bil)
        self.convection_u_w(atomJ, atomF, averages, weighted_averages, bil)
        self.convection_v_u(atomJ, atomF, averages, weighted_averages, bil)
        self.convection_v_v(atomJ, atomF, averages, weighted_averages, bil)
        self.convection_v_w(atomJ, atomF, averages, weighted_averages, bil)
        self.convection_w_u(atomJ, atomF, averages, weighted_averages, bil)
        self.convection_w_v(atomJ, atomF, averages, weighted_averages, bil)
        self.convection_w_w(atomJ, atomF, averages, weighted_averages, bil)

        if self.dof > self.dim + 1:
            Pr = self.get_parameter('Prandtl Number', 1.0)
            atomJ /= Pr
            atomF /= Pr

            self.convection_T_u(atomJ, atomF, averages, weighted_averages, bil)
            self.convection_T_v(atomJ, atomF, averages, weighted_averages, bil)
            self.convection_T_w(atomJ, atomF, averages, weighted_averages, bil)

        atomJ += atomF

        return (atomJ, atomF)


class ConvectiveTerm:

    def __init__(self, nx, ny, nz, dim, x, y, z):
        self.nx = nx
        self.ny = ny
        self.nz = nz

        self.dim = dim

        self.x = x
        self.y = y
        self.z = z

    def backward_average_x(self, bil, averages, weighted_averages, state):
        bil[:, :, :, 0, 0, 0:2] = 1/2
        bil[:, :, :, 1, 0, 0:2] = 1/2

        averages[1:self.nx, :, :, 0] += 1/2 * state[0:self.nx-1, :, :]
        averages[0:self.nx-1, :, :, 0] += 1/2 * state[0:self.nx-1, :, :]

        weighted_averages[:, :, :, 0] = averages[:, :, :, 0]

    def _forward_average_x(self, bil, averages, state, i):
        # distance between u[i] and u[i-1]
        dxmh = self.x[i] - self.x[i-1]
        # distance between u[i+1] and u[i]
        dxph = self.x[i+1] - self.x[i]
        # distance between v[i+1] and v[i]
        dx = (self.x[i+1] - self.x[i-1]) / 2

        bil[i, :, :, 1, 0, 1] += 1/2 * dxmh / dx
        bil[i, :, :, 1, 0, 2] += 1/2 * dxph / dx

        averages[i, :, :, 0] += 1/2 * state[i, :, :] * dxmh / dx
        averages[i, :, :, 0] += 1/2 * state[i+1, :, :] * dxph / dx

    def forward_average_x(self, bil, averages, weighted_averages, state):
        bil[:, :, :, 0, 0, 1:3] = 1/2

        averages[0:self.nx-1, :, :, 0] += 1/2 * state[0:self.nx-1, :, :]
        averages[0:self.nx-1, :, :, 0] += 1/2 * state[1:self.nx, :, :]

        for i in range(self.nx-1):
            self._forward_average_x(bil, weighted_averages, state, i)

    def backward_average_y(self, bil, averages, weighted_averages, state):
        bil[:, :, :, 0, 1, 0:2] = 1/2
        bil[:, :, :, 1, 1, 0:2] = 1/2

        averages[:, 1:self.ny, :, 1] += 1/2 * state[:, 0:self.ny-1, :]
        averages[:, 0:self.ny-1, :, 1] += 1/2 * state[:, 0:self.ny-1, :]

        weighted_averages[:, :, :, 1] = averages[:, :, :, 1]

    def _forward_average_y(self, bil, averages, state, j):
        # distance between v[j] and v[j-1]
        dymh = self.y[j] - self.y[j-1]
        # distance between v[j+1] and v[j]
        dyph = self.y[j+1] - self.y[j]
        # distance between u[j+1] and u[j]
        dy = (self.y[j+1] - self.y[j-1]) / 2

        bil[:, j, :, 1, 1, 1] += 1/2 * dymh / dy
        bil[:, j, :, 1, 1, 2] += 1/2 * dyph / dy

        averages[:, j, :, 1] += 1/2 * state[:, j, :] * dymh / dy
        averages[:, j, :, 1] += 1/2 * state[:, j+1, :] * dyph / dy

    def forward_average_y(self, bil, averages, weighted_averages, state):
        bil[:, :, :, 0, 1, 1:3] = 1/2

        averages[:, 0:self.ny-1, :, 1] += 1/2 * state[:, 0:self.ny-1, :]
        averages[:, 0:self.ny-1, :, 1] += 1/2 * state[:, 1:self.ny, :]

        for j in range(self.ny-1):
            self._forward_average_y(bil, weighted_averages, state, j)

    def backward_average_z(self, bil, averages, weighted_averages, state):
        bil[:, :, :, 0, 2, 0:2] = 1/2
        bil[:, :, :, 1, 2, 0:2] = 1/2

        averages[:, :, 1:self.nz, 2] += 1/2 * state[:, :, 0:self.nz-1]
        averages[:, :, 0:self.nz-1, 2] += 1/2 * state[:, :, 0:self.nz-1]

        weighted_averages[:, :, :, 2] = averages[:, :, :, 2]

    def _forward_average_z(self, bil, averages, state, k):
        # distance between w[k] and w[k-1]
        dzmh = self.z[k] - self.z[k-1]
        # distance between w[k+1] and w[k]
        dzph = self.z[k+1] - self.z[k]
        # distance between u[k+1] and u[k]
        dz = (self.z[k+1] - self.z[k-1]) / 2

        bil[:, :, k, 1, 2, 1] += 1/2 * dzmh / dz
        bil[:, :, k, 1, 2, 2] += 1/2 * dzph / dz

        averages[:, :, k, 2] += 1/2 * state[:, :, k] * dzmh / dz
        averages[:, :, k, 2] += 1/2 * state[:, :, k+1] * dzph / dz

    def forward_average_z(self, bil, averages, weighted_averages, state):
        bil[:, :, :, 0, 2, 1:3] = 1/2

        averages[:, :, 0:self.nz-1, 2] += 1/2 * state[:, :, 0:self.nz-1]
        averages[:, :, 0:self.nz-1, 2] += 1/2 * state[:, :, 1:self.nz]

        for k in range(self.nz-1):
            self._forward_average_z(bil, weighted_averages, state, k)

    def value_u(self, bil, averages, weighted_averages, state):
        bil[:, :, :, 0, 0, 1] = 1
        bil[:, :, :, 1, 0, 1] = 1
        averages[0:self.nx-1, :, :, 0] = state[0:self.nx-1, :, :, 0]
        weighted_averages[:, :, :, 0] = averages[:, :, :, 0]

    def value_v(self, bil, averages, weighted_averages, state):
        bil[:, :, :, 0, 1, 1] = 1
        bil[:, :, :, 1, 1, 1] = 1
        averages[:, 0:self.ny-1, :, 1] = state[:, 0:self.ny-1, :, 1]
        weighted_averages[:, :, :, 1] = averages[:, :, :, 1]

    def value_w(self, bil, averages, weighted_averages, state):
        bil[:, :, :, 0, 2, 1] = 1
        bil[:, :, :, 1, 2, 1] = 1
        averages[:, :, 0:self.nz-1, 2] = state[:, :, 0:self.nz-1, 2]
        weighted_averages[:, :, :, 2] = averages[:, :, :, 2]

    def u_x(self, bil):
        for i in range(self.nx):
            for j in range(self.ny):
                for k in range(self.nz):
                    Discretization._forward_u_x(bil[i, j, k, 2, 0, 0, :], i, j, k, self.x, self.y, self.z)

    def v_y(self, bil):
        for i in range(self.nx):
            for j in range(self.ny):
                for k in range(self.nz):
                    Discretization._forward_u_x(bil[i, j, k, 2, 1, 1, :], j, i, k, self.y, self.x, self.z)

    def w_z(self, bil):
        for i in range(self.nx):
            for j in range(self.ny):
                for k in range(self.nz):
                    Discretization._forward_u_x(bil[i, j, k, 2, 2, 2, :], k, j, i, self.z, self.y, self.x)

    def u_y(self, bil):
        for i in range(self.nx):
            for j in range(self.ny):
                for k in range(self.nz):
                    Discretization._backward_u_y(bil[i, j, k, 2, 1, 0, :], i, j, k, self.x, self.y, self.z)

    def v_x(self, bil):
        for i in range(self.nx):
            for j in range(self.ny):
                for k in range(self.nz):
                    Discretization._backward_u_y(bil[i, j, k, 2, 0, 1, :], j, i, k, self.y, self.x, self.z)

    def w_y(self, bil):
        for i in range(self.nx):
            for j in range(self.ny):
                for k in range(self.nz):
                    Discretization._backward_u_y(bil[i, j, k, 2, 1, 2, :], k, j, i, self.z, self.y, self.x)

    def u_z(self, bil):
        for i in range(self.nx):
            for j in range(self.ny):
                for k in range(self.nz):
                    Discretization._backward_u_z(bil[i, j, k, 2, 2, 0, :], i, j, k, self.x, self.y, self.z)

    def v_z(self, bil):
        for i in range(self.nx):
            for j in range(self.ny):
                for k in range(self.nz):
                    Discretization._backward_u_z(bil[i, j, k, 2, 2, 1, :], j, i, k, self.y, self.x, self.z)

    def w_x(self, bil):
        for i in range(self.nx):
            for j in range(self.ny):
                for k in range(self.nz):
                    Discretization._backward_u_z(bil[i, j, k, 2, 0, 2, :], k, j, i, self.z, self.y, self.x)

    def T_x(self, bil):
        for i in range(self.nx):
            for j in range(self.ny):
                for k in range(self.nz):
                    Discretization._backward_u_x(bil[i, j, k, 2, 0, self.dim+1, :], i, j, k, self.x, self.y, self.z)

    def T_y(self, bil):
        for i in range(self.nx):
            for j in range(self.ny):
                for k in range(self.nz):
                    Discretization._backward_u_x(bil[i, j, k, 2, 1, self.dim+1, :], j, i, k, self.y, self.x, self.z)

    def T_z(self, bil):
        for i in range(self.nx):
            for j in range(self.ny):
                for k in range(self.nz):
                    Discretization._backward_u_x(bil[i, j, k, 2, 2, self.dim+1, :], k, j, i, self.z, self.y, self.x)

    def boundary_east(self, bil):
        tmp = numpy.copy(bil[self.nx-1, :, :, 0, 0, 0, 0])
        tmp2 = numpy.copy(bil[self.nx-1, :, :, 1, 0, 0, 0])
        bil[self.nx-1, :, :, :, :, 0, :] = 0
        bil[self.nx-1, :, :, 0, 0, :, :] = 0
        bil[self.nx-1, :, :, 0, 0, 0, 0] = tmp

        bil[self.nx-1, :, :, 1, 0, :, :] = 0
        bil[self.nx-1, :, :, 1, 0, 0, 0] = tmp2

    def boundary_west(self, bil):
        bil[0, :, :, :, 0, :, 0] = 0

    def boundary_north(self, bil):
        tmp = numpy.copy(bil[:, self.ny-1, :, 0, 1, 1, 0])
        tmp2 = numpy.copy(bil[:, self.ny-1, :, 1, 1, 1, 0])
        bil[:, self.ny-1, :, :, :, 1, :] = 0
        bil[:, self.ny-1, :, 0, 1, :, :] = 0
        bil[:, self.ny-1, :, 0, 1, 1, 0] = tmp

        bil[:, self.ny-1, :, 1, 1, :, :] = 0
        bil[:, self.ny-1, :, 1, 1, 1, 0] = tmp2

    def boundary_south(self, bil):
        bil[:, 0, :, :, 1, :, 0] = 0

    def boundary_top(self, bil):
        tmp = numpy.copy(bil[:, :, self.nz-1, 0, 2, 2, 0])
        tmp2 = numpy.copy(bil[:, :, self.nz-1, 1, 2, 2, 0])
        bil[:, :, self.nz-1, :, :, 2, :] = 0
        bil[:, :, self.nz-1, 0, 2, :, :] = 0
        bil[:, :, self.nz-1, 0, 2, 2, 0] = tmp

        bil[:, :, self.nz-1, 1, 2, :, :] = 0
        bil[:, :, self.nz-1, 1, 2, 2, 0] = tmp2

    def boundary_bottom(self, bil):
        bil[:, :, 0, :, 2, :, 0] = 0
