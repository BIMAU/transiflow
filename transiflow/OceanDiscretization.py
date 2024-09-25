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

        D = self.parameters.get('Z-max', 0) - self.parameters.get('Z-min', -1)
        z = utils.create_uniform_coordinate_vector(-D, 0, nz) if z is None else z

        Discretization.__init__(self, parameters, nx, ny, nz, dim, dof,
                                x, y, z, boundary_conditions)

        if self.parameters.get('X-periodic', False):
            self.x_periodic = True

    def _linear_part_2D(self):
        raise NotImplementedError

    def _linear_part_3D(self):
        '''Compute the linear part of the equation in case the domain is 3D.
        In case Re = 0 we instead compute the linear part for the Stokes
        problem.'''

        # Dimensional parameters
        A_H = self.parameters.get('Horizontal Friction Coefficient', 2.5e+05)
        Omega_0 = self.parameters.get('Earth Rotation Rate', 7.292e-05)
        r_0 = self.parameters.get('Earth Radius', 6.37e+06)
        depth = self.parameters.get('Depth', 5000)

        # Non-dimensional parameters
        Ek_V = self.parameters.get('Vertical Ekman Number',
                                   A_H / (2 * Omega_0 * depth * depth))
        Ek_H = self.parameters.get('Horizontal Ekman Number',
                                   A_H / (2 * Omega_0 * r_0 * r_0)) * 100

        return -Ek_H * (self.u_xx() + self.u_yy() + self.v_xx() + self.v_yy())

    def nonlinear_part(self, state):
        r'''Compute the nonlinear part of the equation. In case $\Re = 0$ this
        does nothing.

        :meta private:

        '''

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
