import numpy

from transiflow import utils
from transiflow.Discretization import Discretization

class CylindricalDiscretization(Discretization):
    '''Finite volume discretization of the incompressible
    Navier-Stokes equations on a (possibly non-uniform) Arakawa C-grid
    in a cylindrical coordinate system. For details on the
    implementation and the ordering of the variables, see the
    :ref:`transiflow.discretization` class.

    '''

    def __init__(self, parameters, nr, ntheta, nz, dim=None, dof=None,
                 r=None, theta=None, z=None, boundary_conditions=None):
        self.parameters = parameters

        ri = self.parameters.get('R-min', 1.0)
        ro = self.parameters.get('R-max', 2.0)
        self.eta = ri / ro

        L = self.parameters.get('Z-max', 1.0) - self.parameters.get('Z-min', 0.0)

        r = self.get_coordinate_vector(1, 1 / self.eta, nr) if r is None else r

        theta = utils.create_uniform_coordinate_vector(self.parameters.get('Theta-min', 0.0),
                                                       self.parameters.get('Theta-max', 2 * numpy.pi),
                                                       ntheta) if theta is None else theta

        z = utils.create_uniform_coordinate_vector(0, L / self.eta, nz) if z is None else z

        Discretization.__init__(self, parameters, nr, ntheta, nz, dim, dof,
                                r, theta, z, boundary_conditions)

        self.y_periodic = True
        if self.parameters.get('Z-periodic', False):
            self.z_periodic = True

    def _linear_part_2D(self):
        '''Compute the linear part of the equation in case the domain is 2D.
        In case Re = 0 we instead compute the linear part for the Stokes
        problem.'''

        Re = self.get_parameter('Reynolds Number')

        if Re == 0:
            Re = 1

        return 1 / Re * (self.iruscale(self.u_rr()) + self.iru2scale(self.u_tt() - self.value_u() - 2 * self.v_t_u())
                         + self.irvscale(self.v_rr()) + self.irv2scale(self.v_tt() - self.value_v() + 2 * self.u_t_v())) \
            - (self.p_r() + self.irvscale(self.p_t())) \
            + self.div()

    def _linear_part_3D(self):
        '''Compute the linear part of the equation in case the domain is 3D.
        In case Re = 0 we instead compute the linear part for the Stokes
        problem.'''

        Ta = self.get_parameter('Taylor Number')
        if Ta == 0:
            Ta = self.get_parameter('Reynolds Number') / (1 / self.eta - 1)

        if Ta == 0:
            Ta = 1

        return 1 / Ta * (self.iruscale(self.u_rr()) + self.iru2scale(self.u_tt() - self.value_u() - 2 * self.v_t_u())
                         + self.u_zz()
                         + self.irvscale(self.v_rr()) + self.irv2scale(self.v_tt() - self.value_v() + 2 * self.u_t_v())
                         + self.v_zz()
                         + self.irvscale(self.w_rr()) + self.irv2scale(self.w_tt()) + self.w_zz()) \
            - (self.p_r() + self.irvscale(self.p_t()) + self.p_z()) \
            + self.div()

    def nonlinear_part(self, state):
        r'''Compute the nonlinear part of the equation. In case $\Re = 0$ this
        does nothing.

        :meta private:

        '''

        state_mtx = utils.create_padded_state_mtx(state, self.nx, self.ny, self.nz, self.dof,
                                                  self.x_periodic, self.y_periodic, self.z_periodic)

        atomJ = numpy.zeros((self.nx, self.ny, self.nz, self.dof, self.dof, 3, 3, 3))
        atomF = numpy.zeros((self.nx, self.ny, self.nz, self.dof, self.dof, 3, 3, 3))

        Ta = self.get_parameter('Taylor Number')
        Re = self.get_parameter('Reynolds Number')
        if Re == 0 and Ta == 0:
            return (atomJ, atomF)

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

    def _taylor_couette(self, boundary_conditions, atom):
        '''Boundary conditions for the Taylor-Couette problem'''
        ri = self.parameters.get('R-min', 1.0)
        ro = self.parameters.get('R-max', 2.0)
        wo = self.get_parameter('Outer Angular Velocity', 0)
        wi = self.get_parameter('Inner Angular Velocity', 1)

        # This is not supported by the non-dimensionalization
        assert wo == 0

        boundary_conditions.moving_lid_east(atom, (wo * ro) / (wi * ri))
        boundary_conditions.moving_lid_west(atom, 1)

        if self.dim <= 2 or self.nz <= 1:
            return boundary_conditions.get_forcing()

        asym = self.get_parameter('Asymmetry Parameter')
        frc2 = numpy.zeros((self.nx, self.ny, self.nz, self.dof))
        frc2[self.nx-1, 0, :, 2] = asym * numpy.cos(self.z[0:self.nz] / self.z[self.nz-1] * numpy.pi)

        frc = boundary_conditions.get_forcing()
        frc += utils.create_state_vec(frc2, self.nx, self.ny, self.nz, self.dof)

        if not self.z_periodic:
            boundary_conditions.no_slip_top(atom)
            boundary_conditions.no_slip_bottom(atom)

        return frc

    def _setup_boundary_conditions(self):
        '''Setup boundary conditions for the currently defined problem type.'''
        if self.boundary_conditions:
            return
        elif self.problem_type_equals('Taylor-Couette'):
            self.boundary_conditions = self._taylor_couette
        else:
            raise Exception('Invalid problem type %s' % self.get_parameter('Problem Type'))

    # Below are all of the discretizations of separate parts of
    # equations that we can solve using FVM. This takes into account
    # non-uniform grids. New discretizations such as derivatives have
    # to be implemented in a similar way.

    def rvscale(self, atom):
        '''Scale atom by r at the location of v

        :meta private:

        '''
        for i in range(self.nx):
            atom[i, :, :, :, :, :, :, :] *= (self.x[i] + self.x[i-1]) / 2
        return atom

    def iruscale(self, atom):
        '''Scale atom by 1/r at the location of u

        :meta private:

        '''
        for i in range(self.nx):
            atom[i, :, :, :, :, :, :, :] /= self.x[i]
        return atom

    def irvscale(self, atom):
        '''Scale atom by 1/r at the location of v

        :meta private:

        '''
        for i in range(self.nx):
            atom[i, :, :, :, :, :, :, :] /= (self.x[i] + self.x[i-1]) / 2
        return atom

    def iru2scale(self, atom):
        '''Scale atom by 1/r^2 at the location of u

        :meta private:

        '''
        for i in range(self.nx):
            atom[i, :, :, :, :, :, :, :] /= self.x[i] * self.x[i]
        return atom

    def irv2scale(self, atom):
        '''Scale atom by 1/r^2 at the location of v

        :meta private:

        '''
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
        ''':meta private:'''
        atom = numpy.zeros((self.nx, self.ny, self.nz, self.dof, self.dof, 3, 3, 3))
        for i, j, k in numpy.ndindex(self.nx, self.ny, self.nz):
            self._u_rr(atom[i, j, k, 0, 0, :, 1, 1], i, j, k, self.x, self.y, self.z)
        return atom

    def v_tt(self):
        ''':meta private:'''
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
        ''':meta private:'''
        return self.u_yy()

    def v_rr(self):
        ''':meta private:'''
        atom = numpy.zeros((self.nx, self.ny, self.nz, self.dof, self.dof, 3, 3, 3))
        for i, j, k in numpy.ndindex(self.nx, self.ny, self.nz):
            self._v_rr(atom[i, j, k, 1, 1, :, 1, 1], i, j, k, self.x, self.y, self.z)
        return atom

    def w_tt(self):
        ''':meta private:'''
        return self.w_yy()

    def w_rr(self):
        ''':meta private:'''
        atom = numpy.zeros((self.nx, self.ny, self.nz, self.dof, self.dof, 3, 3, 3))
        for i, j, k in numpy.ndindex(self.nx, self.ny, self.nz):
            self._v_rr(atom[i, j, k, 2, 2, :, 1, 1], i, k, j, self.x, self.z, self.y)
        return atom

    def p_r(self):
        ''':meta private:'''
        return self.p_x()

    def p_t(self):
        ''':meta private:'''
        return self.p_y()

    def v_t_u(self):
        ''':meta private:'''
        atom = numpy.zeros((self.nx, self.ny, self.nz, self.dof, self.dof, 3, 3, 3))
        for i, j, k in numpy.ndindex(self.nx, self.ny, self.nz):
            Discretization._backward_u_y(atom[i, j, k, 0, 1, 1, :, 1], i, j, k, self.x, self.y, self.z)
            Discretization._backward_u_y(atom[i, j, k, 0, 1, 2, :, 1], i, j, k, self.x, self.y, self.z)
            atom[i, j, k, 0, 1, :, :, :] /= 2
        return atom

    def u_t_v(self):
        ''':meta private:'''
        atom = numpy.zeros((self.nx, self.ny, self.nz, self.dof, self.dof, 3, 3, 3))
        for i, j, k in numpy.ndindex(self.nx, self.ny, self.nz):
            Discretization._forward_u_x(atom[i, j, k, 1, 0, 0, :, 1], j, i, k, self.y, self.x, self.z)
            Discretization._forward_u_x(atom[i, j, k, 1, 0, 1, :, 1], j, i, k, self.y, self.x, self.z)
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
        ''':meta private:'''
        atom = numpy.zeros((self.nx, self.ny, self.nz, self.dof, self.dof, 3, 3, 3))
        for i, j, k in numpy.ndindex(self.nx, self.ny, self.nz):
            self._value_u(atom[i, j, k, 0, 0, :, 1, 1], i, j, k, self.x, self.y, self.z)
        return atom

    def value_v(self):
        ''':meta private:'''
        atom = numpy.zeros((self.nx, self.ny, self.nz, self.dof, self.dof, 3, 3, 3))
        for i, j, k in numpy.ndindex(self.nx, self.ny, self.nz):
            self._value_u(atom[i, j, k, 1, 1, :, 1, 1], j, i, k, self.y, self.x, self.z)
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
        ''':meta private:'''
        atom = numpy.zeros((self.nx, self.ny, self.nz, self.dof, self.dof, 3, 3, 3))
        for i, j, k in numpy.ndindex(self.nx, self.ny, self.nz):
            self._backward_u_r(atom[i, j, k, self.dim, 0, :, 1, 1], i, j, k, self.x, self.y, self.z)
        return atom

    def div(self):
        ''':meta private:'''
        if self.dim == 2:
            return self.u_x() + self.v_y()
        return self.u_x() + self.v_y() + self.rvscale(self.w_z())

    def u_u_r(self, atomJ, atomF, state):
        ''':meta private:'''
        Discretization.u_u_x(self, atomJ, atomF, state)

    def u_v_r(self, atomJ, atomF, state):
        ''':meta private:'''
        Discretization.u_v_x(self, atomJ, atomF, state)

    def u_w_r(self, atomJ, atomF, state):
        ''':meta private:'''
        Discretization.u_w_x(self, atomJ, atomF, state)

    def v_u_t(self, atomJ_in, atomF_in, state):
        ''':meta private:'''
        atomJ = numpy.zeros((self.nx, self.ny, self.nz, self.dof, self.dof, 3, 3, 3))
        atomF = numpy.zeros((self.nx, self.ny, self.nz, self.dof, self.dof, 3, 3, 3))

        Discretization.v_u_y(self, atomJ, atomF, state)
        self.iruscale(atomJ)
        self.iruscale(atomF)

        atomJ_in += atomJ
        atomF_in += atomF

    def v_v_t(self, atomJ_in, atomF_in, state):
        ''':meta private:'''
        atomJ = numpy.zeros((self.nx, self.ny, self.nz, self.dof, self.dof, 3, 3, 3))
        atomF = numpy.zeros((self.nx, self.ny, self.nz, self.dof, self.dof, 3, 3, 3))

        Discretization.v_v_y(self, atomJ, atomF, state)
        self.irvscale(atomJ)
        self.irvscale(atomF)

        atomJ_in += atomJ
        atomF_in += atomF

    def v_w_t(self, atomJ_in, atomF_in, state):
        ''':meta private:'''
        atomJ = numpy.zeros((self.nx, self.ny, self.nz, self.dof, self.dof, 3, 3, 3))
        atomF = numpy.zeros((self.nx, self.ny, self.nz, self.dof, self.dof, 3, 3, 3))

        Discretization.v_w_y(self, atomJ, atomF, state)
        self.irvscale(atomJ)
        self.irvscale(atomF)

        atomJ_in += atomJ
        atomF_in += atomF

    def v_v(self, atomJ_in, atomF_in, state):
        ''':meta private:'''
        averages_v = self.weighted_average_x(state[:, :, :, 1])
        averages_v = (averages_v[:, 0:self.ny, :] + averages_v[:, 1:self.ny+1, :]) / 2

        atom = numpy.zeros((self.nx, self.ny, self.nz, self.dof, self.dof, 3, 3, 3))

        atom_value = numpy.zeros(1)
        atom_average = numpy.zeros(2)
        for i, j, k in numpy.ndindex(self.nx, self.ny, self.nz):
            Discretization._mass_x(atom_value, i, j, k, self.x, self.y, self.z)
            Discretization._weighted_average(atom_average, i, self.x)
            atom[i, j, k, 0, 1, 1:3, 0, 1] += atom_value * atom_average * averages_v[i, j, k+1] * 1 / 2
            atom[i, j, k, 0, 1, 1:3, 1, 1] += atom_value * atom_average * averages_v[i, j, k+1] * 1 / 2

        self.iruscale(atom)

        atomJ_in += atom
        atomF_in += atom

    def u_v(self, atomJ_in, atomF_in, state):
        ''':meta private:'''
        averages_u = self.weighted_average_y(state[:, :, :, 0])
        averages_u = (averages_u[0:self.nx, :, :] + averages_u[1:self.nx+1, :, :]) / 2
        averages_v = state[1:self.nx+1, 1:self.ny+1, 1:self.nz+1, 1]

        atomJ = numpy.zeros((self.nx, self.ny, self.nz, self.dof, self.dof, 3, 3, 3))
        atomF = numpy.zeros((self.nx, self.ny, self.nz, self.dof, self.dof, 3, 3, 3))

        atom_value = numpy.zeros(1)
        atom_average = numpy.zeros(2)
        for i, j, k in numpy.ndindex(self.nx, self.ny, self.nz):
            Discretization._mass_x(atom_value, j, i, k, self.y, self.x, self.z)
            atomF[i, j, k, 1, 1, 1, 1:2, 1] -= atom_value * averages_u[i, j, k+1]

            Discretization._weighted_average(atom_average, j, self.y)
            atomJ[i, j, k, 1, 0, 0, 1:3, 1] -= atom_value * atom_average * averages_v[i, j, k] * 1 / 2
            atomJ[i, j, k, 1, 0, 1, 1:3, 1] -= atom_value * atom_average * averages_v[i, j, k] * 1 / 2

        self.irvscale(atomF)
        self.irvscale(atomJ)

        atomJ_in += atomJ
        atomF_in += atomF
