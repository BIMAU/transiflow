import numpy

from transiflow import utils
from transiflow.Discretization import Discretization

class OceanDiscretization(Discretization):
    '''Finite volume discretization of an ocean model in a spherical
    polar coordinate system with vertical z-coordinate. For details on
    the implementation and the ordering of the variables, see the
    :ref:`transiflow.discretization` class.

    '''

    def __init__(self, parameters, nx, ny, nz, dim=None, dof=None,
                 x=None, y=None, z=None, boundary_conditions=None):
        self.parameters = parameters

        xmin = self.parameters.get('X-min', 0.0) * numpy.pi / 180
        xmax = self.parameters.get('X-max', 360.0) * numpy.pi / 180
        x = utils.create_uniform_coordinate_vector(xmin, xmax, nx) if x is None else x

        ymin = self.parameters.get('Y-min', -85.5) * numpy.pi / 180
        ymax = self.parameters.get('Y-max', 85.5) * numpy.pi / 180
        y = utils.create_uniform_coordinate_vector(ymin, ymax, ny) if y is None else y

        zmin = self.parameters.get('Z-min', -1)
        zmax = self.parameters.get('Z-max', 0)
        qz = self.parameters.get('Grid Stretching Factor', 2.25)
        z = utils.create_stretched_coordinate_vector_from_function(
            zmin, zmax, nz,
            lambda z: numpy.tanh(qz * z) / numpy.tanh(qz))

        Discretization.__init__(self, parameters, nx, ny, nz, dim, dof,
                                x, y, z, boundary_conditions)

        if self.parameters.get('X-periodic', False):
            self.x_periodic = True

    def _linear_part_2D(self):
        raise NotImplementedError

    def _linear_part_3D(self):
        # Dimensional parameters
        A_H = self.parameters.get('Horizontal Friction Coefficient', 2.5e+05)
        A_V = self.parameters.get('Vertical Friction Coefficient', 5.0e-03)
        Omega_0 = self.parameters.get('Earth Rotation Rate', 7.292e-05)
        eta_f = self.parameters.get('Rotation flag', 1)
        r_0 = self.parameters.get('Earth Radius', 6.37e+06)
        depth = self.parameters.get('Depth', 5000)

        # Non-dimensional parameters
        Ek_H = self.parameters.get('Horizontal Ekman Number',
                                   A_H / (2 * Omega_0 * r_0 * r_0)) * 100
        Ek_V = self.parameters.get('Vertical Ekman Number',
                                   A_V / (2 * Omega_0 * depth * depth))

        return Ek_H * (self.u_xx() + self.u_yy()
                       - self.icos2uscale(self.value_u() + 2 * self.sinuscale(self.v_x_at_u()))
                       + self.v_xx() + self.v_yy()
                       - self.icos2vscale(self.value_v() - 2 * self.sinvscale(self.u_x_at_v()))) \
            + Ek_V * (self.u_zz() + self.v_zz()) \
            - (self.icosuscale(self.p_x()) + self.p_y() + self.p_z()) \
            + eta_f * (self.sinuscale(self.v_at_u()) - self.sinvscale(self.u_at_v())) \
            + self.div()

    def nonlinear_part(self, state):
        # state_mtx = utils.create_padded_state_mtx(state, self.nx, self.ny, self.nz, self.dof,
        #                                           self.x_periodic, self.y_periodic, self.z_periodic)

        atomJ = numpy.zeros((self.nx, self.ny, self.nz, self.dof, self.dof, 3, 3, 3))
        atomF = numpy.zeros((self.nx, self.ny, self.nz, self.dof, self.dof, 3, 3, 3))

        return (atomJ, atomF)

    def _ocean(self, boundary_conditions, atom):
        '''Boundary conditions for the 3D ocean circulation'''
        return boundary_conditions.get_forcing()

    def _setup_boundary_conditions(self):
        '''Setup boundary conditions for the currently defined problem type.'''
        if self.boundary_conditions:
            return
        elif self.problem_type_equals('Ocean'):
            self.boundary_conditions = self._ocean
        else:
            raise Exception('Invalid problem type %s' % self.get_parameter('Problem Type'))

    # Below are all of the discretizations of separate parts of
    # equations that we can solve using FVM. This takes into account
    # non-uniform grids. New discretizations such as derivatives have
    # to be implemented in a similar way.

    def icosuscale(self, atom):
        '''Scale atom by $1 / cos(y)$ at the location of u.

        :meta private:

        '''
        y_center = utils.compute_coordinate_vector_centers(self.y)
        ucos = numpy.cos(y_center)

        for j in range(self.ny):
            atom[:, j, :, :, :, :, :, :] /= ucos[j]

        return atom

    def icos2uscale(self, atom):
        '''Scale atom by $1 / cos(y)^2$ at the location of u.

        :meta private:

        '''
        y_center = utils.compute_coordinate_vector_centers(self.y)
        ucos2 = numpy.cos(y_center) ** 2

        for j in range(self.ny):
            atom[:, j, :, :, :, :, :, :] /= ucos2[j]

        return atom

    def sinuscale(self, atom):
        '''Scale atom by $sin(y)$ at the location of u.

        :meta private:

        '''
        y_center = utils.compute_coordinate_vector_centers(self.y)
        usin = numpy.sin(y_center)

        for j in range(self.ny):
            atom[:, j, :, :, :, :, :, :] *= usin[j]

        return atom

    def icos2vscale(self, atom):
        '''Scale atom by $1 / cos(y)^2$ at the location of v.

        :meta private:

        '''
        vcos2 = numpy.cos(self.y) ** 2
        for j in range(self.ny):
            atom[:, j, :, :, :, :, :, :] /= vcos2[j]

        return atom

    def sinvscale(self, atom):
        '''Scale atom by $sin(y)$ at the location of v.

        :meta private:

        '''
        vsin = numpy.sin(self.y)
        for j in range(self.ny):
            atom[:, j, :, :, :, :, :, :] *= vsin[j]

        return atom

    def div(self):
        ''':meta private:'''
        return self.icosuscale(self.u_x() + self.v_y()) + self.w_z()

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

        cos2 = numpy.cos(y[j-1] + dy / 2) ** 2

        # second order finite difference
        atom[0] = 1 / dx / cos2 * dy * dz
        atom[2] = 1 / dxp1 / cos2 * dy * dz
        atom[1] = -atom[0] - atom[2]

    def u_xx(self):
        ''':meta private:'''
        atom = numpy.zeros((self.nx, self.ny, self.nz, self.dof, self.dof, 3, 3, 3))
        for i, j, k in numpy.ndindex(self.nx, self.ny, self.nz):
            self._u_xx(atom[i, j, k, 0, 0, :, 1, 1], i, j, k, self.x, self.y, self.z)
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

        yc = y[j] - (y[j] - y[j-1]) / 2

        # second order finite difference
        atom[0] = 1 / dy * numpy.cos(y[j-1]) / numpy.cos(yc) * dx * dz
        atom[2] = 1 / dyp1 * numpy.cos(y[j]) / numpy.cos(yc) * dx * dz
        atom[1] = -atom[0] - atom[2]

    def u_yy(self):
        ''':meta private:'''
        atom = numpy.zeros((self.nx, self.ny, self.nz, self.dof, self.dof, 3, 3, 3))
        for i, j, k in numpy.ndindex(self.nx, self.ny, self.nz):
            self._u_yy(atom[i, j, k, 0, 0, 1, :, 1], i, j, k, self.x, self.y, self.z)
        return atom

    @staticmethod
    def _v_xx(atom, i, j, k, x, y, z):
        # distance between v[i] and v[i-1]
        dx = (x[i] - x[i-2]) / 2
        # distance between u[i+1] and u[i]
        dxp1 = (x[i+1] - x[i-1]) / 2
        # volume size in the y direction
        dy = (y[j+1] - y[j-1]) / 2
        # volume size in the z direction
        dz = z[k] - z[k-1]

        cos2 = numpy.cos(y[j]) ** 2

        # second order finite difference
        atom[0] = 1 / dx / cos2 * dy * dz
        atom[2] = 1 / dxp1 / cos2 * dy * dz
        atom[1] = -atom[0] - atom[2]

    def v_xx(self):
        ''':meta private:'''
        atom = numpy.zeros((self.nx, self.ny, self.nz, self.dof, self.dof, 3, 3, 3))
        for i, j, k in numpy.ndindex(self.nx, self.ny, self.nz):
            self._v_xx(atom[i, j, k, 1, 1, :, 1, 1], i, j, k, self.x, self.y, self.z)
        return atom

    @staticmethod
    def _v_yy(atom, i, j, k, x, y, z):
        # distance between v[j] and v[j-1]
        dy = y[j] - y[j-1]
        # distance between v[j+1] and v[j]
        dyp1 = y[j+1] - y[j]
        # volume size in the x direction
        dx = x[i] - x[i-1]
        # volume size in the z direction
        dz = z[k] - z[k-1]

        yc = y[j] - dy / 2
        ycp1 = y[j] + dyp1 / 2

        # second order finite difference
        atom[0] = 1 / dy * numpy.cos(yc) / numpy.cos(y[j]) * dx * dz
        atom[2] = 1 / dyp1 * numpy.cos(ycp1) / numpy.cos(y[j]) * dx * dz
        atom[1] = -atom[0] - atom[2]

    def v_yy(self):
        ''':meta private:'''
        atom = numpy.zeros((self.nx, self.ny, self.nz, self.dof, self.dof, 3, 3, 3))
        for i, j, k in numpy.ndindex(self.nx, self.ny, self.nz):
            self._v_yy(atom[i, j, k, 1, 1, 1, :, 1], i, j, k, self.x, self.y, self.z)
        return atom

    def v_y(self):
        ''':meta private:'''
        atom = Discretization.v_y(self)
        for j in range(self.ny):
            atom[:, j, :, self.dim, 1, 1, 0, 1] *= numpy.cos(self.y[j-1])
            atom[:, j, :, self.dim, 1, 1, 1, 1] *= numpy.cos(self.y[j])
        return atom

    def u_x_at_v(self):
        ''':meta private:'''
        atom = numpy.zeros((self.nx, self.ny, self.nz, self.dof, self.dof, 3, 3, 3))
        for i, j, k in numpy.ndindex(self.nx, self.ny, self.nz):
            self._backward_u_x(atom[i, j, k, 1, 0, :, 1, 1], i, j, k, self.x, self.y, self.z)
            self._backward_u_x(atom[i, j, k, 1, 0, :, 2, 1], i, j, k, self.x, self.y, self.z)
        return atom / 2

    def v_x_at_u(self):
        ''':meta private:'''
        atom = numpy.zeros((self.nx, self.ny, self.nz, self.dof, self.dof, 3, 3, 3))
        for i, j, k in numpy.ndindex(self.nx, self.ny, self.nz):
            self._forward_u_x(atom[i, j, k, 0, 1, :, 0, 1], i, j, k, self.x, self.y, self.z)
            self._forward_u_x(atom[i, j, k, 0, 1, :, 1, 1], i, j, k, self.x, self.y, self.z)
        return atom / 2
