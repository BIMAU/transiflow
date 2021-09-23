import numpy

from fvm import utils
from fvm import BoundaryConditions
from fvm import Discretization

class CylindricalDiscretization(Discretization):
    '''Finite volume discretization of the incompressible Navier-Stokes
    equations on a (possibly non-uniform) Arakawa C-grid in a
    cylindrical coordinate system. For details on the implementation
    and the ordering of the variables, see the Discretization class.

    '''

    def __init__(self, parameters, nr, ntheta, nz, dim, dof, r=None, theta=None, z=None):
        self.parameters = parameters

        if self.parameters.get('Grid Stretching', False) or 'Grid Stretching Factor' in self.parameters.keys():
            r = utils.create_stretched_coordinate_vector(
                self.parameters.get('R-min', 0.0), self.parameters.get('R-max', 1.0), nr,
                self.parameters.get('Grid Stretching Factor', 1.5)) if r is None else r

            # TODO: Maybe force this if dim = 2?
            z = utils.create_stretched_coordinate_vector(
                self.parameters.get('Z-min', 0.0), self.parameters.get('Z-max', 1.0), nz,
                self.parameters.get('Grid Stretching Factor', 1.5)) if z is None else z
        else:
            r = utils.create_uniform_coordinate_vector(
                self.parameters.get('R-min', 0.0), self.parameters.get('R-max', 1.0), nr) if r is None else r

            # TODO: Maybe force this if dim = 2?
            z = utils.create_uniform_coordinate_vector(
                self.parameters.get('Z-min', 0.0), self.parameters.get('Z-max', 1.0), nz) if z is None else z

        theta = utils.create_uniform_coordinate_vector(
            self.parameters.get('Theta-min', 0.0), self.parameters.get('Theta-max', 1.0), ntheta) \
            if theta is None else theta

        Discretization.__init__(self, parameters, nr, ntheta, nz, dim, dof, r, theta, z)

    def _linear_part_2D(self):
        '''Compute the linear part of the equation in case the domain is 2D.
        In case Re = 0 we instead compute the linear part for the Stokes
        problem.'''

        Re = self.get_parameter('Reynolds Number')

        if Re == 0:
            Re = 1

        atom = 1 / Re * (self.iruscale(self.u_rr()) + self.iru2scale(self.u_tt() - self.value_u() - 2 * self.v_t_u())
                         + self.irvscale(self.v_rr()) + self.irv2scale(self.v_tt() - self.value_v() - 2 * self.u_t_v())) \
            - (self.p_r() + self.irvscale(self.p_t())) \
            + self.div()

        return atom

    def boundaries(self, atom):
        '''Compute boundary conditions for the currently defined problem type.'''

        # TODO: Make it possible to interface this from the outside.

        boundary_conditions = BoundaryConditions(self.nx, self.ny, self.nz, self.dim, self.dof, self.x, self.y, self.z)

        frc = numpy.zeros(self.nx * self.ny * self.nz * self.dof)

        if self.problem_type_equals('Taylor-Couette'):
            self.y_periodic = True

            vo = self.get_parameter('Outer Velocity', 2)
            vi = self.get_parameter('Inner Velocity', 1)
            frc += boundary_conditions.moving_lid_east(atom, vo)
            frc += boundary_conditions.moving_lid_west(atom, vi)

            if self.dim == 2 or self.nz <= 1:
                return frc

            boundary_conditions.no_slip_top(atom)
            boundary_conditions.no_slip_bottom(atom)
        else:
            raise Exception('Invalid problem type %s' % self.get_parameter('Problem Type'))

        return frc

    # Below are all of the discretizations of separate parts of
    # equations that we can solve using FVM. This takes into account
    # non-uniform grids. New discretizations such as derivatives have
    # to be implemented in a similar way.

    def iruscale(self, atom):
        '''Scale atom by 1/r at the location of u'''
        for i in range(self.nx):
            atom[i, :, :, :, :, :, :, :] /= self.x[i]
        return atom

    def irvscale(self, atom):
        '''Scale atom by 1/r at the location of v'''
        for i in range(self.nx):
            atom[i, :, :, :, :, :, :, :] /= (self.x[i] + self.x[i-1]) / 2
        return atom

    def iru2scale(self, atom):
        '''Scale atom by 1/r^2 at the location of u'''
        for i in range(self.nx):
            atom[i, :, :, :, :, :, :, :] /= self.x[i] * self.x[i]
        return atom

    def irv2scale(self, atom):
        '''Scale atom by 1/r^2 at the location of v'''
        for i in range(self.nx):
            atom[i, :, :, :, :, :, :, :] /= (self.x[i] + self.x[i-1]) * (self.x[i] + self.x[i-1]) / 4
        return atom

    @staticmethod
    def _u_rr(atom, i, j, k, x, y, z):
        # distance between u[i] and u[i-1]
        dx = x[i] - x[i-1]
        rv = x[i-1] + dx / 2
        # distance between u[i+1] and u[i]
        dxp1 = x[i+1] - x[i]
        rvp1 = x[i] + dxp1 / 2
        # volume size in the y direction
        dy = y[j] - y[j-1]
        # volume size in the z direction
        dz = z[k] - z[k-1]

        # second order finite difference
        atom[0] = rv / dx * dy * dz
        atom[2] = rvp1 / dxp1 * dy * dz
        atom[1] = -atom[0] - atom[2]

    def u_rr(self):
        atom = numpy.zeros([self.nx, self.ny, self.nz, self.dof, self.dof, 3, 3, 3])
        for i in range(self.nx):
            for j in range(self.ny):
                for k in range(self.nz):
                    CylindricalDiscretization._u_rr(atom[i, j, k, 0, 0, :, 1, 1], i, j, k, self.x, self.y, self.z)
        return atom

    def v_tt(self):
        return self.v_yy()

    @staticmethod
    def _v_rr(atom, i, j, k, x, y, z):
        # distance between v[i] and v[i-1]
        dx = (x[i] - x[i-2]) / 2
        # distance between v[i+1] and v[i]
        dxp1 = (x[i+1] - x[i-1]) / 2
        # volume size in the y direction
        dy = (y[j+1] - y[j-1]) / 2
        # volume size in the z direction
        dz = z[k] - z[k-1]

        # second order finite difference
        atom[0] = x[i-1] / dx * dy * dz
        atom[2] = x[i] / dxp1 * dy * dz
        atom[1] = -atom[0] - atom[2]

    def u_tt(self):
        return self.u_yy()

    def v_rr(self):
        atom = numpy.zeros([self.nx, self.ny, self.nz, self.dof, self.dof, 3, 3, 3])
        for i in range(self.nx):
            for j in range(self.ny):
                for k in range(self.nz):
                    CylindricalDiscretization._v_rr(atom[i, j, k, 1, 1, :, 1, 1], i, j, k, self.x, self.y, self.z)
        return atom

    def p_r(self):
        return self.p_x()

    def p_t(self):
        return self.p_y()

    def v_t_u(self):
        atom = numpy.zeros([self.nx, self.ny, self.nz, self.dof, self.dof, 3, 3, 3])
        for i in range(self.nx):
            for j in range(self.ny):
                for k in range(self.nz):
                    Discretization._backward_u_y(atom[i, j, k, 0, 1, 1, :, 1], i, j, k, self.x, self.y, self.z)
                    Discretization._backward_u_y(atom[i, j, k, 0, 1, 2, :, 1], i, j, k, self.x, self.y, self.z)
                    atom[i, j, k, 0, 1, :, :, :] /= 2
        return atom

    def u_t_v(self):
        atom = numpy.zeros([self.nx, self.ny, self.nz, self.dof, self.dof, 3, 3, 3])
        for i in range(self.nx):
            for j in range(self.ny):
                for k in range(self.nz):
                    Discretization._backward_u_x(atom[i, j, k, 1, 0, 1, :, 1], j, i, k, self.y, self.x, self.z)
                    Discretization._backward_u_x(atom[i, j, k, 1, 0, 2, :, 1], j, i, k, self.y, self.x, self.z)
                    atom[i, j, k, 1, 0, :, :, :] /= 2
        return atom

    @staticmethod
    def _value_u(atom, i, j, k, x, y, z):
        # volume size in the x direction
        dx = (x[i+1] - x[i-1]) / 2
        # volume size in the y direction
        dy = y[j] - y[j-1]
        # volume size in the z direction
        dz = z[k] - z[k-1]

        atom[1] = dx * dy * dz

    def value_u(self):
        atom = numpy.zeros([self.nx, self.ny, self.nz, self.dof, self.dof, 3, 3, 3])
        for i in range(self.nx):
            for j in range(self.ny):
                for k in range(self.nz):
                    CylindricalDiscretization._value_u(atom[i, j, k, 0, 0, :, 1, 1], i, j, k, self.x, self.y, self.z)
        return atom

    def value_v(self):
        atom = numpy.zeros([self.nx, self.ny, self.nz, self.dof, self.dof, 3, 3, 3])
        for i in range(self.nx):
            for j in range(self.ny):
                for k in range(self.nz):
                    CylindricalDiscretization._value_u(atom[i, j, k, 1, 1, :, 1, 1], j, i, k, self.y, self.x, self.z)
        return atom

    def value_w(self):
        atom = numpy.zeros([self.nx, self.ny, self.nz, self.dof, self.dof, 3, 3, 3])
        for i in range(self.nx):
            for j in range(self.ny):
                for k in range(self.nz):
                    CylindricalDiscretization._value_u(atom[i, j, k, 0, 0, :, 1, 1], k, j, i, self.z, self.y, self.x)
        return atom

    @staticmethod
    def _backward_u_r(atom, i, j, k, x, y, z):
        # volume size in the y direction
        dy = y[j] - y[j-1]
        # volume size in the z direction
        dz = z[k] - z[k-1]

        # backward difference
        atom[1] = x[i] * dy * dz
        atom[0] = -x[i-1] * dy * dz

    def u_r(self):
        atom = numpy.zeros([self.nx, self.ny, self.nz, self.dof, self.dof, 3, 3, 3])
        for i in range(self.nx):
            for j in range(self.ny):
                for k in range(self.nz):
                    CylindricalDiscretization._backward_u_r(atom[i, j, k, self.dim, 0, :, 1, 1], i, j, k,
                                                            self.x, self.y, self.z)
        return atom

    def div(self):
        if self.dim == 2:
            return self.irvscale(self.u_r() + self.v_y())
        return self.irvscale(self.u_r() + self.v_y() + self.w_z())

    def convection_v_u(self, atomJ_in, atomF_in, averages, weighted_averages, bil):
        atomJ = numpy.zeros([self.nx, self.ny, self.nz, self.dof, self.dof, 3, 3, 3])
        atomF = numpy.zeros([self.nx, self.ny, self.nz, self.dof, self.dof, 3, 3, 3])

        Discretization.convection_v_u(self, atomJ, atomF, averages, weighted_averages, bil)
        self.iruscale(atomJ)
        self.iruscale(atomF)

        atomJ_in += atomJ
        atomF_in += atomF

    def convection_v_v(self, atomJ_in, atomF_in, averages, weighted_averages, bil):
        atomJ = numpy.zeros([self.nx, self.ny, self.nz, self.dof, self.dof, 3, 3, 3])
        atomF = numpy.zeros([self.nx, self.ny, self.nz, self.dof, self.dof, 3, 3, 3])

        Discretization.convection_v_v(self, atomJ, atomF, averages, weighted_averages, bil)
        self.irvscale(atomJ)
        self.irvscale(atomF)

        atomJ_in += atomJ
        atomF_in += atomF

    def _weighted_average_x(atom, i, j, k, x, y, z):
        # volume size in the x direction
        dx = (x[i+1] - x[i-1]) / 2

        # distance between u[i] and u[i-1]
        dxmh = x[i] - x[i-1]
        # distance between u[i+1] and u[i]
        dxph = x[i+1] - x[i]

        atom[1] = 1 / 2 * dxmh / dx
        atom[2] = 1 / 2 * dxph / dx

    def value_v_at_u(self, state):
        averages = numpy.zeros([self.nx, self.ny, self.nz])

        atom = numpy.zeros(3)
        for i in range(self.nx-1):
            CylindricalDiscretization._weighted_average_x(atom, i, 0, 0, self.x, self.y, self.z)

            averages[i, :, :] = 1 / 2 * atom[1] * state[i+1, 0:self.ny, 1:self.nz+1, 1]
            averages[i, :, :] = 1 / 2 * atom[2] * state[i+2, 0:self.ny, 1:self.nz+1, 1]
            averages[i, :, :] = 1 / 2 * atom[1] * state[i+1, 1:self.ny+1, 1:self.nz+1, 1]
            averages[i, :, :] = 1 / 2 * atom[2] * state[i+2, 1:self.ny+1, 1:self.nz+1, 1]

        return averages

    @staticmethod
    def _value_v_at_u(atom, i, j, k, x, y, z):
        # volume size in the x direction
        dx = (x[i+1] - x[i-1]) / 2
        # volume size in the y direction
        dy = y[j] - y[j-1]
        # volume size in the z direction
        dz = z[k] - z[k-1]

        CylindricalDiscretization._weighted_average_x(atom, i, j, k, x, y, z)

        atom[1] *= dx * dy * dz
        atom[2] *= dx * dy * dz

    def value_v_v(self, state):
        atom = numpy.zeros([self.nx, self.ny, self.nz, self.dof, self.dof, 3, 3, 3])

        averages = self.value_v_at_u(state)
        for i in range(self.nx):
            for j in range(self.ny):
                for k in range(self.nz):
                    CylindricalDiscretization._value_v_at_u(atom[i, j, k, 0, 1, :, 1, 1], i, j, k,
                                                            self.x, self.y, self.z)
                    CylindricalDiscretization._value_v_at_u(atom[i, j, k, 0, 1, :, 2, 1], i, j, k,
                                                            self.x, self.y, self.z)
                    atom[i, j, k, 0, 1, :, :, :] *= -averages[i, j, k]

        return atom

    def value_u_at_v(self, state):
        averages = numpy.zeros([self.nx, self.ny, self.nz])

        atom = numpy.zeros(3)
        for j in range(self.ny-1):
            CylindricalDiscretization._weighted_average_x(atom, j, 0, 0, self.y, self.x, self.z)

            averages[:, j, :] = 1 / 2 * atom[1] * state[0:self.nx, j+1, 1:self.nz+1, 0]
            averages[:, j, :] = 1 / 2 * atom[2] * state[0:self.nx, j+2, 1:self.nz+1, 0]
            averages[:, j, :] = 1 / 2 * atom[1] * state[1:self.nx+1, j+1, 1:self.nz+1, 0]
            averages[:, j, :] = 1 / 2 * atom[2] * state[1:self.nx+1, j+2, 1:self.nz+1, 0]

        return averages

    def value_u_v(self, state):
        atom = numpy.zeros([self.nx, self.ny, self.nz, self.dof, self.dof, 3, 3, 3])

        averages = self.value_u_at_v(state)
        for i in range(self.nx):
            for j in range(self.ny):
                for k in range(self.nz):
                    CylindricalDiscretization._value_u(atom[i, j, k, 1, 1, :, 1, 1], j, i, k, self.y, self.x, self.z)
                    atom[i, j, k, 1, 1, :, :, :] *= averages[i, j, k]

        return atom

    def convection_2D(self, state):
        atomJ, atomF = Discretization.convection_2D(self, state)

        # atom = self.iruscale(self.value_v_v(state))
        # atom += self.irvscale(self.value_u_v(state))

        # atomJ += atom
        # atomF += atom

        return (atomJ, atomF)
