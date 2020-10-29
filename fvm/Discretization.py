import numpy
import copy

from fvm import utils
from fvm import BoundaryConditions
from fvm import CrsMatrix

class Discretization:

    def __init__(self, parameters, nx, ny, nz, dof):
        self.parameters = copy.copy(parameters)

        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.dof = dof

        self.x = utils.create_uniform_coordinate_vector(self.nx)
        self.y = utils.create_uniform_coordinate_vector(self.ny)
        self.z = utils.create_uniform_coordinate_vector(self.nz)

    def set_parameter(self, name, value):
        self.parameters[name] = value

    def get_parameter(self, name, default=0):
        return self.parameters.get(name, default)

    def linear_part(self):

        Re = self.get_parameter('Reynolds Number')
        Ra = self.get_parameter('Rayleigh Number')
        Pr = self.get_parameter('Prandtl Number')

        if Re == 0:
            Re = 1

        atom = 1 / Re * (self.u_xx(self.x, self.y, self.z) + self.u_yy(self.x, self.y, self.z) + self.u_zz(self.x, self.y, self.z) \
                      +  self.v_xx(self.x, self.y, self.z) + self.v_yy(self.x, self.y, self.z) + self.v_zz(self.x, self.y, self.z) \
                      +  self.w_xx(self.x, self.y, self.z) + self.w_yy(self.x, self.y, self.z) + self.w_zz(self.x, self.y, self.z)) \
            - (self.p_x(self.x, self.y, self.z) + self.p_y(self.x, self.y, self.z) + self.p_z(self.x, self.y, self.z)) \
            + self.div(self.x, self.y, self.z)

        if Ra:
            atom += Ra * self.forward_average_T_z(self.x, self.y, self.z)

        if Pr:
            atom += 1 / Pr * (self.T_xx(self.x, self.y, self.z) + self.T_yy(self.x, self.y, self.z) + self.T_zz(self.x, self.y, self.z))
            atom += 1 / Pr * self.backward_average_w_z(self.x, self.y, self.z)

        return atom

    def nonlinear_part(self, state):
        state_mtx = utils.create_state_mtx(state, self.nx, self.ny, self.nz, self.dof)

        Re = self.get_parameter('Reynolds Number')
        if Re == 0:
            state_mtx[:, :, :, :] = 0

        return self.convection(state_mtx, self.x, self.y, self.z)

    def rhs(self, state, atom):
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
        n = self.nx * self.ny * self.nz * self.dof

        # Put the state in shifted matrix form
        state_mtx = numpy.zeros([self.nx+2, self.ny+2, self.nz+2, self.dof])
        state_mtx[1:self.nx+1, 1:self.ny+1, 1:self.nz+1, :] = utils.create_state_mtx(state, self.nx, self.ny, self.nz, self.dof)

        # Add up all contributions without iterating over the domain
        out_mtx = numpy.zeros([self.nx, self.ny, self.nz, self.dof])
        for k in range(3):
            for j in range(3):
                for i in range(3):
                    for d1 in range(self.dof):
                        for d2 in range(self.dof):
                            out_mtx[:, :, :, d1] -= atom[:, :, :, d1, d2, i, j, k] * state_mtx[i:(i+self.nx), j:(j+self.ny), k:(k+self.nz), d2]

        return utils.create_state_vec(out_mtx, self.nx, self.ny, self.nz, self.dof)

    def jacobian(self, atom):
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
                                jcoA[idx] = row + (config[1]-1) * self.dof + (config[2]-1) * self.nx * self.dof + (config[3]-1) * self.nx * self.ny * self.dof + config[0] - d1
                                coA[idx] = atom[i, j, k, d1, config[0], config[1], config[2], config[3]]
                                idx += 1
                        row += 1
                        begA[row] = idx
        return CrsMatrix(coA, jcoA, begA)

    @staticmethod
    def _problem_type_equals(first, second):
        return first.lower() == second.lower()

    def boundaries(self, atom):
        boundary_conditions = BoundaryConditions(self.nx, self.ny, self.nz, self.dof)
        problem_type = self.get_parameter('Problem Type', 'Lid-driven cavity')

        frc = numpy.zeros(self.nx * self.ny * self.nz * self.dof)

        if Discretization._problem_type_equals(problem_type, 'Rayleigh-Benard'):
            frc += boundary_conditions.heatflux_east(atom, self.x, self.y, self.z, 0)
        elif Discretization._problem_type_equals(problem_type, 'Differentially heated cavity'):
            frc += boundary_conditions.temperature_east(atom, -1/2)
        else:
            boundary_conditions.dirichlet_east(atom)

        if Discretization._problem_type_equals(problem_type, 'Rayleigh-Benard'):
            frc += boundary_conditions.heatflux_west(atom, self.x, self.y, self.z, 0)
        elif Discretization._problem_type_equals(problem_type, 'Differentially heated cavity'):
            frc += boundary_conditions.temperature_west(atom, 1/2)
        else:
            boundary_conditions.dirichlet_west(atom)

        if Discretization._problem_type_equals(problem_type, 'Lid-driven cavity') and self.nz <= 1:
            frc += boundary_conditions.moving_lid_north(atom, 1)
        elif Discretization._problem_type_equals(problem_type, 'Rayleigh-Benard'):
            frc += boundary_conditions.heatflux_north(atom, self.x, self.y, self.z, 0)
        elif Discretization._problem_type_equals(problem_type, 'Differentially heated cavity'):
            frc += boundary_conditions.heatflux_north(atom, self.x, self.y, self.z, 0)
        else:
            boundary_conditions.dirichlet_north(atom)

        if Discretization._problem_type_equals(problem_type, 'Rayleigh-Benard'):
            frc += boundary_conditions.heatflux_south(atom, self.x, self.y, self.z, 0)
        elif Discretization._problem_type_equals(problem_type, 'Differentially heated cavity'):
            frc += boundary_conditions.heatflux_south(atom, self.x, self.y, self.z, 0)
        else:
            boundary_conditions.dirichlet_south(atom)

        if Discretization._problem_type_equals(problem_type, 'Lid-driven cavity') and self.nz > 1:
            frc += boundary_conditions.moving_lid_top(atom, 1)
        elif Discretization._problem_type_equals(problem_type, 'Differentially heated cavity'):
            frc += boundary_conditions.heatflux_top(atom, self.x, self.y, self.z, 0)
        else:
            boundary_conditions.dirichlet_top(atom)

        if Discretization._problem_type_equals(problem_type, 'Differentially heated cavity'):
            frc += boundary_conditions.heatflux_bottom(atom, self.x, self.y, self.z, 0)
        else:
            boundary_conditions.dirichlet_bottom(atom)

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

    def u_xx(self, x, y, z):
        atom = numpy.zeros([self.nx, self.ny, self.nz, self.dof, self.dof, 3, 3, 3])
        for i in range(self.nx):
            for j in range(self.ny):
                for k in range(self.nz):
                    Discretization._u_xx(atom[i, j, k, 0, 0, :, 1, 1], i, j, k, x, y, z)
        return atom

    def v_yy(self, x, y, z):
        atom = numpy.zeros([self.nx, self.ny, self.nz, self.dof, self.dof, 3, 3, 3])
        for i in range(self.nx):
            for j in range(self.ny):
                for k in range(self.nz):
                    Discretization._u_xx(atom[i, j, k, 1, 1, 1, :, 1], j, i, k, y, x, z)
        return atom

    def w_zz(self, x, y, z):
        atom = numpy.zeros([self.nx, self.ny, self.nz, self.dof, self.dof, 3, 3, 3])
        for i in range(self.nx):
            for j in range(self.ny):
                for k in range(self.nz):
                    Discretization._u_xx(atom[i, j, k, 2, 2, 1, 1, :], k, j, i, z, y, x)
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
                    Discretization._u_yy(atom[i, j, k, 0, 0, 1, :, 1], i, j, k, x, y, z)
        return atom

    def v_xx(self, x, y, z):
        atom = numpy.zeros([self.nx, self.ny, self.nz, self.dof, self.dof, 3, 3, 3])
        for i in range(self.nx):
            for j in range(self.ny):
                for k in range(self.nz):
                    Discretization._u_yy(atom[i, j, k, 1, 1, :, 1, 1], j, i, k, y, x, z)
        return atom

    def w_yy(self, x, y, z):
        atom = numpy.zeros([self.nx, self.ny, self.nz, self.dof, self.dof, 3, 3, 3])
        for i in range(self.nx):
            for j in range(self.ny):
                for k in range(self.nz):
                    Discretization._u_yy(atom[i, j, k, 2, 2, 1, :, 1], k, j, i, z, y, x)
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
                    Discretization._u_zz(atom[i, j, k, 0, 0, 1, 1, :], i, j, k, x, y, z)
        return atom

    def v_zz(self, x, y, z):
        atom = numpy.zeros([self.nx, self.ny, self.nz, self.dof, self.dof, 3, 3, 3])
        for i in range(self.nx):
            for j in range(self.ny):
                for k in range(self.nz):
                    Discretization._u_zz(atom[i, j, k, 1, 1, 1, 1, :], j, i, k, y, x, z)
        return atom

    def w_xx(self, x, y, z):
        atom = numpy.zeros([self.nx, self.ny, self.nz, self.dof, self.dof, 3, 3, 3])
        for i in range(self.nx):
            for j in range(self.ny):
                for k in range(self.nz):
                    Discretization._u_zz(atom[i, j, k, 2, 2, :, 1, 1], k, j, i, z, y, x)
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
                    Discretization._T_xx(atom[i, j, k, 4, 4, :, 1, 1], i, j, k, x, y, z)
        return atom

    def T_yy(self, x, y, z):
        atom = numpy.zeros([self.nx, self.ny, self.nz, self.dof, self.dof, 3, 3, 3])
        for i in range(self.nx):
            for j in range(self.ny):
                for k in range(self.nz):
                    Discretization._T_xx(atom[i, j, k, 4, 4, 1, :, 1], j, i, k, y, x, z)
        return atom

    def T_zz(self, x, y, z):
        atom = numpy.zeros([self.nx, self.ny, self.nz, self.dof, self.dof, 3, 3, 3])
        for i in range(self.nx):
            for j in range(self.ny):
                for k in range(self.nz):
                    Discretization._T_xx(atom[i, j, k, 4, 4, 1, 1, :], k, j, i, z, y, x)
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
                    Discretization._forward_u_x(atom[i, j, k, 0, 3, :, 1, 1], i, j, k, x, y, z)
        return atom

    def p_y(self, x, y, z):
        atom = numpy.zeros([self.nx, self.ny, self.nz, self.dof, self.dof, 3, 3, 3])
        for i in range(self.nx):
            for j in range(self.ny):
                for k in range(self.nz):
                    Discretization._forward_u_x(atom[i, j, k, 1, 3, 1, :, 1], j, i, k, y, x, z)
        return atom

    def p_z(self, x, y, z):
        atom = numpy.zeros([self.nx, self.ny, self.nz, self.dof, self.dof, 3, 3, 3])
        for i in range(self.nx):
            for j in range(self.ny):
                for k in range(self.nz):
                    Discretization._forward_u_x(atom[i, j, k, 2, 3, 1, 1, :], k, j, i, z, y, x)
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
                    Discretization._backward_u_x(atom[i, j, k, 3, 0, :, 1, 1], i, j, k, x, y, z)
        return atom

    def v_y(self, x, y, z):
        atom = numpy.zeros([self.nx, self.ny, self.nz, self.dof, self.dof, 3, 3, 3])
        for i in range(self.nx):
            for j in range(self.ny):
                for k in range(self.nz):
                    Discretization._backward_u_x(atom[i, j, k, 3, 1, 1, :, 1], j, i, k, y, x, z)
        return atom

    def w_z(self, x, y, z):
        atom = numpy.zeros([self.nx, self.ny, self.nz, self.dof, self.dof, 3, 3, 3])
        for i in range(self.nx):
            for j in range(self.ny):
                for k in range(self.nz):
                    Discretization._backward_u_x(atom[i, j, k, 3, 2, 1, 1, :], k, j, i, z, y, x)
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

    def div(self, x, y, z):
        return self.u_x(x, y, z) + self.v_y(x, y, z) + self.w_z(x, y, z)

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

    def forward_average_T_z(self, x, y, z):
        atom = numpy.zeros([self.nx, self.ny, self.nz, self.dof, self.dof, 3, 3, 3])
        for i in range(self.nx):
            for j in range(self.ny):
                for k in range(self.nz):
                    Discretization._forward_average_x(atom[i, j, k, 2, 4, 1, 1, :], k, j, i, z, y, x)
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

    def backward_average_w_z(self, x, y, z):
        atom = numpy.zeros([self.nx, self.ny, self.nz, self.dof, self.dof, 3, 3, 3])
        for i in range(self.nx):
            for j in range(self.ny):
                for k in range(self.nz):
                    Discretization._backward_average_x(atom[i, j, k, 4, 2, 1, 1, :], k, j, i, z, y, x)
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
    def _convection_u_v(atomJ, atomF, averages, bil, varU, varV, nx, i):
        for d1 in range(3):
            i2 = i + d1 - 1

            v_x = bil[i, :, :, 1, varU, varV, d1]
            if not numpy.any(v_x):
                continue

            coef1 = averages[i2, :, :, varU, varV] * v_x
            for d2 in range(3):
                coef2 = bil[i2, :, :, 0, varV, varU, d2]
                if numpy.any(coef2):
                    idx = [1, 1, 1]
                    idx[varU] += d1 - 1
                    idx[varU] += d2 - 1
                    atomF[i, :, :, varV, varV, idx[0], idx[1], idx[2]] -= coef1 * coef2

            coef1 = averages[i2, :, :, varV, varU] * v_x
            for d2 in range(3):
                coef2 = bil[i2, :, :, 0, varU, varV, d2]
                if numpy.any(coef2):
                    idx = [1, 1, 1]
                    idx[varU] += d1 - 1
                    idx[varV if varV < 3 else varU] += d2 - 1
                    atomJ[i, :, :, varV, varU, idx[0], idx[1], idx[2]] -= coef1 * coef2

    @staticmethod
    def _convection_v_u(atomJ, atomF, averages, bil, varV, varU, ny, j):
        for d1 in range(3):
            j2 = j + d1 - 1

            u_y = bil[:, j, :, 1, varV, varU, d1]
            if not numpy.any(u_y):
                continue

            coef1 = averages[:, j2, :, varV, varU] * u_y
            for d2 in range(3):
                coef2 = bil[:, j2, :, 0, varU, varV, d2]
                if numpy.any(coef2):
                    idx = [1, 1, 1]
                    idx[varV] += d1 - 1
                    idx[varV] += d2 - 1
                    atomF[:, j, :, varU, varU, idx[0], idx[1], idx[2]] -= coef1 * coef2

            coef1 = averages[:, j2, :, varU, varV] * u_y
            for d2 in range(3):
                coef2 = bil[:, j2, :, 0, varV, varU, d2]
                if numpy.any(coef2):
                    idx = [1, 1, 1]
                    idx[varV] += d1 - 1
                    idx[varU if varU < 3 else varV] += d2 - 1
                    atomJ[:, j, :, varU, varV, idx[0], idx[1], idx[2]] -= coef1 * coef2

    @staticmethod
    def _convection_w_u(atomJ, atomF, averages, bil, varW, varU, nz, k):
        for d1 in range(3):
            k2 = k + d1 - 1

            u_z = bil[:, :, k, 1, varW, varU, d1]
            if not numpy.any(u_z):
                continue

            coef1 = averages[:, :, k2, varW, varU] * u_z
            for d2 in range(3):
                coef2 = bil[:, :, k2, 0, varU, varW, d2]
                if numpy.any(coef2):
                    idx = [1, 1, 1]
                    idx[varW] += d1 - 1
                    idx[varW] += d2 - 1
                    atomF[:, :, k, varU, varU, idx[0], idx[1], idx[2]] -= coef1 * coef2

            coef1 = averages[:, :, k2, varU, varW] * u_z
            for d2 in range(3):
                coef2 = bil[:, :, k2, 0, varW, varU, d2]
                if numpy.any(coef2):
                    idx = [1, 1, 1]
                    idx[varW] += d1 - 1
                    idx[varU if varU < 3 else varW] += d2 - 1
                    atomJ[:, :, k, varU, varW, idx[0], idx[1], idx[2]] -= coef1 * coef2

    def convection_u_u(self, atomJ, atomF, averages, bil):
        for i in range(self.nx):
            Discretization._convection_u_v(atomJ, atomF, averages, bil, 0, 0, self.nx, i)

    def convection_v_u(self, atomJ, atomF, averages, bil):
        for j in range(self.ny):
            Discretization._convection_v_u(atomJ, atomF, averages, bil, 1, 0, self.ny, j)

    def convection_w_u(self, atomJ, atomF, averages, bil):
        for k in range(self.nz):
            Discretization._convection_w_u(atomJ, atomF, averages, bil, 2, 0, self.nz, k)

    def convection_u_v(self, atomJ, atomF, averages, bil):
        for i in range(self.nx):
            Discretization._convection_u_v(atomJ, atomF, averages, bil, 0, 1, self.nx, i)

    def convection_v_v(self, atomJ, atomF, averages, bil):
        for j in range(self.ny):
            Discretization._convection_v_u(atomJ, atomF, averages, bil, 1, 1, self.ny, j)

    def convection_w_v(self, atomJ, atomF, averages, bil):
        for k in range(self.nz):
            Discretization._convection_w_u(atomJ, atomF, averages, bil, 2, 1, self.nz, k)

    def convection_u_w(self, atomJ, atomF, averages, bil):
        for i in range(self.nx):
            Discretization._convection_u_v(atomJ, atomF, averages, bil, 0, 2, self.nx, i)

    def convection_v_w(self, atomJ, atomF, averages, bil):
        for j in range(self.ny):
            Discretization._convection_v_u(atomJ, atomF, averages, bil, 1, 2, self.ny, j)

    def convection_w_w(self, atomJ, atomF, averages, bil):
        for k in range(self.nz):
            Discretization._convection_w_u(atomJ, atomF, averages, bil, 2, 2, self.nz, k)

    def convection_T_u(self, atomJ, atomF, averages, bil):
        for i in range(self.nx):
            Discretization._convection_u_v(atomJ, atomF, averages, bil, 0, 4, self.nx, i)

    def convection_T_v(self, atomJ, atomF, averages, bil):
        for j in range(self.ny):
            Discretization._convection_v_u(atomJ, atomF, averages, bil, 1, 4, self.ny, j)

    def convection_T_w(self, atomJ, atomF, averages, bil):
        for k in range(self.nz):
            Discretization._convection_w_u(atomJ, atomF, averages, bil, 2, 4, self.nz, k)

    def convection(self, state, x, y, z):
        bil = numpy.zeros([self.nx, self.ny, self.nz, 2, self.dof, self.dof, 3])
        averages = numpy.zeros([self.nx, self.ny, self.nz, self.dof, self.dof])

        convective_term = ConvectiveTerm(self.nx, self.ny, self.nz)

        convective_term.backward_average_x(bil[:, :, :, :, 0, :, :], averages[:, :, :, 0, :], state[:, :, :, 0]) # tMxU
        convective_term.forward_average_x(bil[:, :, :, :, 1, :, :], averages[:, :, :, 1, :], state[:, :, :, 1]) # tMxV
        convective_term.forward_average_x(bil[:, :, :, :, 2, :, :], averages[:, :, :, 2, :], state[:, :, :, 2]) # tMxW
        convective_term.forward_average_y(bil[:, :, :, :, 0, :, :], averages[:, :, :, 0, :], state[:, :, :, 0]) # tMyU
        convective_term.backward_average_y(bil[:, :, :, :, 1, :, :], averages[:, :, :, 1, :], state[:, :, :, 1]) # tMyV
        convective_term.forward_average_y(bil[:, :, :, :, 2, :, :], averages[:, :, :, 2, :], state[:, :, :, 2]) # tMyW
        convective_term.forward_average_z(bil[:, :, :, :, 0, :, :], averages[:, :, :, 0, :], state[:, :, :, 0]) # tMzU
        convective_term.forward_average_z(bil[:, :, :, :, 1, :, :], averages[:, :, :, 1, :], state[:, :, :, 1]) # tMzV
        convective_term.backward_average_z(bil[:, :, :, :, 2, :, :], averages[:, :, :, 2, :], state[:, :, :, 2]) # tMzW

        if self.dof > 4:
            convective_term.forward_average_x(bil[:, :, :, :, 4, :, :], averages[:, :, :, 4, :], state[:, :, :, 4]) # tMxT
            convective_term.forward_average_y(bil[:, :, :, :, 4, :, :], averages[:, :, :, 4, :], state[:, :, :, 4]) # tMyT
            convective_term.forward_average_z(bil[:, :, :, :, 4, :, :], averages[:, :, :, 4, :], state[:, :, :, 4]) # tMzT
            convective_term.value_u(bil[:, :, :, :, :, 4, :], averages[:, :, :, :, 4], state)
            convective_term.value_v(bil[:, :, :, :, :, 4, :], averages[:, :, :, :, 4], state)
            convective_term.value_w(bil[:, :, :, :, :, 4, :], averages[:, :, :, :, 4], state)

        convective_term.u_x(bil, x, y, z) # tMxUMxU
        convective_term.u_y(bil, x, y, z) # tMxVMyU
        convective_term.u_z(bil, x, y, z) # tMxWMzU
        convective_term.v_x(bil, x, y, z) # tMyUMxV
        convective_term.v_y(bil, x, y, z) # tMyVMyV
        convective_term.v_z(bil, x, y, z) # tMyWMzV
        convective_term.w_x(bil, x, y, z) # tMzUMxW
        convective_term.w_y(bil, x, y, z) # tMzVMyW
        convective_term.w_z(bil, x, y, z) # tMzWMzW

        if self.dof > 4:
            convective_term.T_x(bil, x, y, z)
            convective_term.T_y(bil, x, y, z)
            convective_term.T_z(bil, x, y, z)

        convective_term.dirichlet_east(bil)
        convective_term.dirichlet_west(bil)
        convective_term.dirichlet_north(bil)
        convective_term.dirichlet_south(bil)
        convective_term.dirichlet_top(bil)
        convective_term.dirichlet_bottom(bil)

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

        if self.dof > 4:
            self.convection_T_u(atomJ, atomF, averages, bil)
            self.convection_T_v(atomJ, atomF, averages, bil)
            self.convection_T_w(atomJ, atomF, averages, bil)

        atomJ += atomF

        return (atomJ, atomF)


class ConvectiveTerm:

    def __init__(self, nx, ny, nz):
        self.nx = nx
        self.ny = ny
        self.nz = nz

    def backward_average_x(self, bil, averages, state):
        bil[:, :, :, 0, 0, 0:2] = 1/2
        averages[1:self.nx, :, :, 0] += 1/2 * state[0:self.nx-1, :, :]
        averages[0:self.nx-1, :, :, 0] += 1/2 * state[0:self.nx-1, :, :]

    def forward_average_x(self, bil, averages, state):
        bil[:, :, :, 0, 0, 1:3] = 1/2
        averages[0:self.nx-1, :, :, 0] += 1/2 * state[0:self.nx-1, :, :]
        averages[0:self.nx-1, :, :, 0] += 1/2 * state[1:self.nx, :, :]

    def backward_average_y(self, bil, averages, state):
        bil[:, :, :, 0, 1, 0:2] = 1/2
        averages[:, 1:self.ny, :, 1] += 1/2 * state[:, 0:self.ny-1, :]
        averages[:, 0:self.ny-1, :, 1] += 1/2 * state[:, 0:self.ny-1, :]

    def forward_average_y(self, bil, averages, state):
        bil[:, :, :, 0, 1, 1:3] = 1/2
        averages[:, 0:self.ny-1, :, 1] += 1/2 * state[:, 0:self.ny-1, :]
        averages[:, 0:self.ny-1, :, 1] += 1/2 * state[:, 1:self.ny, :]

    def backward_average_z(self, bil, averages, state):
        bil[:, :, :, 0, 2, 0:2] = 1/2
        averages[:, :, 1:self.nz, 2] += 1/2 * state[:, :, 0:self.nz-1]
        averages[:, :, 0:self.nz-1, 2] += 1/2 * state[:, :, 0:self.nz-1]

    def forward_average_z(self, bil, averages, state):
        bil[:, :, :, 0, 2, 1:3] = 1/2
        averages[:, :, 0:self.nz-1, 2] += 1/2 * state[:, :, 0:self.nz-1]
        averages[:, :, 0:self.nz-1, 2] += 1/2 * state[:, :, 1:self.nz]

    def value_u(self, bil, averages, state):
        bil[:, :, :, 0, 0, 1] = 1
        averages[0:self.nx-1, :, :, 0] = state[0:self.nx-1, :, :, 0]

    def value_v(self, bil, averages, state):
        bil[:, :, :, 0, 1, 1] = 1
        averages[:, 0:self.ny-1, :, 1] = state[:, 0:self.ny-1, :, 1]

    def value_w(self, bil, averages, state):
        bil[:, :, :, 0, 2, 1] = 1
        averages[:, :, 0:self.nz-1, 2] = state[:, :, 0:self.nz-1, 2]

    def u_x(self, bil, x, y, z):
        for i in range(self.nx):
            for j in range(self.ny):
                for k in range(self.nz):
                    Discretization._forward_u_x(bil[i, j, k, 1, 0, 0, :], i, j, k, x, y, z)

    def v_y(self, bil, x, y, z):
        for i in range(self.nx):
            for j in range(self.ny):
                for k in range(self.nz):
                    Discretization._forward_u_x(bil[i, j, k, 1, 1, 1, :], j, i, k, y, x, z)

    def w_z(self, bil, x, y, z):
        for i in range(self.nx):
            for j in range(self.ny):
                for k in range(self.nz):
                    Discretization._forward_u_x(bil[i, j, k, 1, 2, 2, :], k, j, i, z, y, x)

    def u_y(self, bil, x, y, z):
        for i in range(self.nx):
            for j in range(self.ny):
                for k in range(self.nz):
                    Discretization._backward_u_y(bil[i, j, k, 1, 1, 0, :], i, j, k, x, y, z)

    def v_x(self, bil, x, y, z):
        for i in range(self.nx):
            for j in range(self.ny):
                for k in range(self.nz):
                    Discretization._backward_u_y(bil[i, j, k, 1, 0, 1, :], j, i, k, y, x, z)

    def w_y(self, bil, x, y, z):
        for i in range(self.nx):
            for j in range(self.ny):
                for k in range(self.nz):
                    Discretization._backward_u_y(bil[i, j, k, 1, 1, 2, :], k, j, i, z, y, x)

    def u_z(self, bil, x, y, z):
        for i in range(self.nx):
            for j in range(self.ny):
                for k in range(self.nz):
                    Discretization._backward_u_z(bil[i, j, k, 1, 2, 0, :], i, j, k, x, y, z)

    def v_z(self, bil, x, y, z):
        for i in range(self.nx):
            for j in range(self.ny):
                for k in range(self.nz):
                    Discretization._backward_u_z(bil[i, j, k, 1, 2, 1, :], j, i, k, y, x, z)

    def w_x(self, bil, x, y, z):
        for i in range(self.nx):
            for j in range(self.ny):
                for k in range(self.nz):
                    Discretization._backward_u_z(bil[i, j, k, 1, 0, 2, :], k, j, i, z, y, x)

    def T_x(self, bil, x, y, z):
        for i in range(self.nx):
            for j in range(self.ny):
                for k in range(self.nz):
                    Discretization._backward_u_x(bil[i, j, k, 1, 0, 4, :], i, j, k, x, y, z)

    def T_y(self, bil, x, y, z):
        for i in range(self.nx):
            for j in range(self.ny):
                for k in range(self.nz):
                    Discretization._backward_u_x(bil[i, j, k, 1, 1, 4, :], j, i, k, y, x, z)

    def T_z(self, bil, x, y, z):
        for i in range(self.nx):
            for j in range(self.ny):
                for k in range(self.nz):
                    Discretization._backward_u_x(bil[i, j, k, 1, 2, 4, :], k, j, i, z, y, x)

    def dirichlet_east(self, bil):
        tmp = numpy.copy(bil[self.nx-1, :, :, 0, 0, 0, 0])
        bil[self.nx-1, :, :, :, :, 0, :] = 0
        bil[self.nx-1, :, :, 0, 0, 0, 0] = tmp

    def dirichlet_west(self, bil):
        bil[0, :, :, 0, 0, 0, 0] = 0

    def dirichlet_north(self, bil):
        tmp = numpy.copy(bil[:, self.ny-1, :, 0, 1, 1, 0])
        bil[:, self.ny-1, :, :, :, 1, :] = 0
        bil[:, self.ny-1, :, 0, 1, 1, 0] = tmp

    def dirichlet_south(self, bil):
        bil[:, 0, :, 0, 1, 1, 0] = 0

    def dirichlet_top(self, bil):
        tmp = numpy.copy(bil[:, :, self.nz-1, 0, 2, 2, 0])
        bil[:, :, self.nz-1, :, :, 2, :] = 0
        bil[:, :, self.nz-1, 0, 2, 2, 0] = tmp

    def dirichlet_bottom(self, bil):
        bil[:, :, 0, 0, 2, 2, 0] = 0
