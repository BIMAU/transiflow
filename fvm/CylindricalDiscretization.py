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

        if self.problem_type_equals('Taylor-Couette'):
            self.y_periodic = True

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

    def _linear_part_3D(self):
        '''Compute the linear part of the equation in case the domain is 3D.
        In case Re = 0 we instead compute the linear part for the Stokes
        problem.'''

        Re = self.get_parameter('Reynolds Number')

        if Re == 0:
            Re = 1

        atom = 1 / Re * (self.iruscale(self.u_rr()) + self.iru2scale(self.u_tt() - self.value_u() - 2 * self.v_t_u())
                         + self.u_zz()
                         + self.irvscale(self.v_rr()) + self.irv2scale(self.v_tt() - self.value_v() - 2 * self.u_t_v())
                         + self.v_zz()
                         + self.irvscale(self.w_rr()) + self.irv2scale(self.w_tt()) + self.w_zz()) \
            - (self.p_r() + self.irvscale(self.p_t()) + self.p_z()) \
            + self.div()

        return atom

    def nonlinear_part(self, state):
        '''Compute the nonlinear part of the equation. In case Re = 0 this
        does nothing.'''

        state_mtx = utils.create_padded_state_mtx(state, self.nx, self.ny, self.nz, self.dof,
                                                  self.x_periodic, self.y_periodic, self.z_periodic)

        Re = self.get_parameter('Reynolds Number')
        if Re == 0:
            state_mtx[:, :, :, :] = 0

        atomJ = numpy.zeros([self.nx, self.ny, self.nz, self.dof, self.dof, 3, 3, 3])
        atomF = numpy.zeros([self.nx, self.ny, self.nz, self.dof, self.dof, 3, 3, 3])

        self.u_u_r(atomJ, atomF, state_mtx)
        self.u_v_r(atomJ, atomF, state_mtx)
        self.v_u_t(atomJ, atomF, state_mtx)
        self.v_v_t(atomJ, atomF, state_mtx)

        self.v_v(atomJ, atomF, state_mtx)
        self.u_v(atomJ, atomF, state_mtx)

        if self.dim > 2:
            self.u_w_r(atomJ, atomF, state_mtx)
            self.v_w_t(atomJ, atomF, state_mtx)
            self.w_u_z(atomJ, atomF, state_mtx)
            self.w_v_z(atomJ, atomF, state_mtx)
            self.w_w_z(atomJ, atomF, state_mtx)

        atomJ += atomF

        return (atomJ, atomF)

    def boundaries(self, atom):
        '''Compute boundary conditions for the currently defined problem type.'''

        # TODO: Make it possible to interface this from the outside.

        boundary_conditions = BoundaryConditions(self.nx, self.ny, self.nz, self.dim, self.dof, self.x, self.y, self.z)

        frc = numpy.zeros(self.nx * self.ny * self.nz * self.dof)

        if self.problem_type_equals('Taylor-Couette'):
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

    def w_tt(self):
        return self.w_yy()

    @staticmethod
    def _w_rr(atom, i, j, k, x, y, z):
        # distance between v[i] and v[i-1]
        dx = (x[i] - x[i-2]) / 2
        # distance between v[i+1] and v[i]
        dxp1 = (x[i+1] - x[i-1]) / 2
        # volume size in the y direction
        dy = y[j] - y[j-1]
        # volume size in the z direction
        dz = (z[k+1] - z[k-1]) / 2

        # second order finite difference
        atom[0] = x[i-1] / dx * dy * dz
        atom[2] = x[i] / dxp1 * dy * dz
        atom[1] = -atom[0] - atom[2]

    def w_rr(self):
        atom = numpy.zeros([self.nx, self.ny, self.nz, self.dof, self.dof, 3, 3, 3])
        for i in range(self.nx):
            for j in range(self.ny):
                for k in range(self.nz):
                    CylindricalDiscretization._w_rr(atom[i, j, k, 1, 1, :, 1, 1], i, j, k, self.x, self.y, self.z)
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
        return self.irvscale(self.u_r() + self.v_y()) + self.w_z()

    def u_u_r(self, atomJ, atomF, state):
        Discretization.u_u_x(self, atomJ, atomF, state)

    def u_v_r(self, atomJ, atomF, state):
        Discretization.u_v_x(self, atomJ, atomF, state)

    def u_w_r(self, atomJ, atomF, state):
        Discretization.u_w_x(self, atomJ, atomF, state)

    def v_u_t(self, atomJ_in, atomF_in, state):
        atomJ = numpy.zeros([self.nx, self.ny, self.nz, self.dof, self.dof, 3, 3, 3])
        atomF = numpy.zeros([self.nx, self.ny, self.nz, self.dof, self.dof, 3, 3, 3])

        Discretization.v_u_y(self, atomJ, atomF, state)
        self.iruscale(atomJ)
        self.iruscale(atomF)

        atomJ_in += atomJ
        atomF_in += atomF

    def v_v_t(self, atomJ_in, atomF_in, state):
        atomJ = numpy.zeros([self.nx, self.ny, self.nz, self.dof, self.dof, 3, 3, 3])
        atomF = numpy.zeros([self.nx, self.ny, self.nz, self.dof, self.dof, 3, 3, 3])

        Discretization.v_v_y(self, atomJ, atomF, state)
        self.irvscale(atomJ)
        self.irvscale(atomF)

        atomJ_in += atomJ
        atomF_in += atomF

    def v_w_t(self, atomJ_in, atomF_in, state):
        atomJ = numpy.zeros([self.nx, self.ny, self.nz, self.dof, self.dof, 3, 3, 3])
        atomF = numpy.zeros([self.nx, self.ny, self.nz, self.dof, self.dof, 3, 3, 3])

        Discretization.v_w_y(self, atomJ, atomF, state)
        self.irvscale(atomJ)
        self.irvscale(atomF)

        atomJ_in += atomJ
        atomF_in += atomF

    def v_v(self, atomJ_in, atomF_in, state):
        averages_v = self.weighted_average_x(state[:, :, :, 1])
        averages_v = (averages_v[:, 0:self.ny, :] + averages_v[:, 1:self.ny+1, :]) / 2

        atom = numpy.zeros([self.nx, self.ny, self.nz, self.dof, self.dof, 3, 3, 3])

        atom_value = numpy.zeros(1)
        atom_average = numpy.zeros(2)
        for i in range(self.nx):
            for j in range(self.ny):
                for k in range(self.nz):
                    Discretization._mass_x(atom_value, i, j, k, self.x, self.y, self.z)
                    Discretization._weighted_average(atom_average, i, self.x)
                    atom[i, j, k, 0, 1, 1:3, 0, 1] += atom_value * atom_average * averages_v[i, j, k+1] * 1 / 2
                    atom[i, j, k, 0, 1, 1:3, 1, 1] += atom_value * atom_average * averages_v[i, j, k+1] * 1 / 2

        self.iruscale(atom)

        atomJ_in += atom
        atomF_in += atom

    def u_v(self, atomJ_in, atomF_in, state):
        averages_u = self.weighted_average_y(state[:, :, :, 0])
        averages_u = (averages_u[0:self.nx, :, :] + averages_u[1:self.nx+1, :, :]) / 2
        averages_v = state[1:self.nx+1, 1:self.ny+1, 1:self.nz+1, 1]

        atomJ = numpy.zeros([self.nx, self.ny, self.nz, self.dof, self.dof, 3, 3, 3])
        atomF = numpy.zeros([self.nx, self.ny, self.nz, self.dof, self.dof, 3, 3, 3])

        atom_value = numpy.zeros(1)
        atom_average = numpy.zeros(2)
        for i in range(self.nx):
            for j in range(self.ny):
                for k in range(self.nz):
                    Discretization._mass_x(atom_value, j, i, k, self.y, self.x, self.z)
                    atomF[i, j, k, 1, 1, 1, 1, 1] -= atom_value * averages_u[i, j, k+1]

                    Discretization._weighted_average(atom_average, j, self.y)
                    atomJ[i, j, k, 1, 0, 0, 1:3, 1] -= atom_value * atom_average * averages_v[i, j, k] * 1 / 2
                    atomJ[i, j, k, 1, 0, 1, 1:3, 1] -= atom_value * atom_average * averages_v[i, j, k] * 1 / 2

        self.irvscale(atomF)
        self.irvscale(atomJ)

        atomJ_in += atomJ
        atomF_in += atomF
