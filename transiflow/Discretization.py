import numpy

from transiflow import utils
from transiflow import BoundaryConditions
from transiflow import CrsMatrix

class Discretization:
    r'''Finite volume discretization of the incompressible
    Navier-Stokes equations on a (possibly non-uniform) Arakawa
    C-grid. After discretization, the incompressible Navier-Stokes
    equations can be writen as a system of ordinary differential
    equations with algebraic constraints (DAE) of the form

    .. math:: M(p) \frac{\d u}{\d t} = F(u, p)

    where $u$ is a state vector, $p$ is a set of parameters, $M(p)$ is
    a singular mass matrix, and $F(u, p)$ is the non-linear function
    including the forcing that is applied to the system.

    The variables in the state vector $u$ are ordered according to

    | ``[u, v, w, p, T, S, u, v, w, p, T, S, u, v, w, p, T, S, ...]``

    at positions

    | ``(0, 0, 0), (1, 0, 0), ..., (0, 1, 0), ..., (0, 0, 1), ...``

    Velocities ``u, v, w`` are staggered according to the C-grid
    definition, pressure ``p``, temperature ``T`` and salinity ``S``
    are defined in the centers of the grid cells. Variables are left
    out if they are not relevant to the problem. A 2D lid-driven
    cavity, for instance, only has ``[u, v, p]``.

    Parameters
    ----------
    parameters : dict
        Key-value pairs describing parameters of the model, for
        instance the Renolds number and the problem type. Possible
        values can be found in :ref:`problem definitions`.
    nx : int
        Grid size in the x direction.
    ny : int
        Grid size in the y direction.
    nz : int, optional
        Grid size in the z direction. 1 for 2-dimensional problems.
        This is the default.
    dim : int, optional
        Physical dimension of the problem. In case this is set to 2, w
        is not referenced in the state vector. The default is based on
        the value of nz.
    dof : int, optional
        Degrees of freedom for this problem. This should be set to dim
        plus 1 for each of pressure, temperature and salinity, if they
        are required for your problem. For example a 3D differentially
        heated cavity has dof = 3 + 1 + 1 = 5.
    x : array_like, optional
        Coordinate vector in the x direction.
    y : array_like, optional
        Coordinate vector in the y direction.
    z : array_like, optional
        Coordinate vector in the z direction.
    boundary_conditions : function, optional
        User-supplied function that implements the boundary
        conditions. It is called as ``boundary_conditions(bc, atom)``
        where ``bc`` is an instance of the :class:`.BoundaryConditions`
        class.

    Notes
    -----

    All discretizations are defined on atoms which for every grid cell
    define the contributions of neighbouring grid cells. In 3D this
    means 27 contributions from neighbouring grid cells are defined
    for every grid cell. For instance

    | ``atom[i, j, k, :, :, 1, 1, 1]``

    contains the contribution from the current cell at point
    ``(i, j, k)`` and

    | ``atom[i, j, k, :, :, 1, 0, 1]``

    contains the contribution from the cell south of the current one.
    A discretization of $u_{xx}$ on a uniform grid in 1D could be
    defined by

    | ``atom[i, j, k, :, :, 0, 1, 1] =  1 / dx``
    | ``atom[i, j, k, :, :, 1, 1, 1] = -2 / dx``
    | ``atom[i, j, k, :, :, 2, 1, 1] =  1 / dx``

    where we note that the mass matrix is also scaled by ``dx``.

    The remaining two indices, denoted by ``(:, :)`` above, contain
    the variable that is being used and the location of the equation
    that is being discretized. If we work in 2D, and we compute
    $p_x$ located in the first equation (as in the standard
    formulation of the incompressible Navier-Stokes equations), then
    the contribution is stored in

    | ``atom[:, :, :, 0, 2, :, :, :]``

    where the 0 comes from the first equation, 2 comes from the pressure.

    '''

    def __init__(self, parameters, nx, ny, nz=1, dim=None, dof=None,
                 x=None, y=None, z=None, boundary_conditions=None):
        self.parameters = parameters
        self.old_parameters = None

        self.nx = nx
        self.ny = ny
        self.nz = nz

        self.dim = dim
        if dim is None:
            self.dim = 3 if self.nz > 1 else 2

        self.dof = dof
        if dof is None:
            self.set_dof()

        self.x_periodic = False
        self.y_periodic = False
        self.z_periodic = False

        if self.nz == 1:
            self.z_periodic = True

        self.x = self.get_coordinate_vector(self.parameters.get('X-min', 0.0),
                                            self.parameters.get('X-max', 1.0),
                                            self.nx) if x is None else x
        self.y = self.get_coordinate_vector(self.parameters.get('Y-min', 0.0),
                                            self.parameters.get('Y-max', 1.0),
                                            self.ny) if y is None else y
        self.z = self.get_coordinate_vector(self.parameters.get('Z-min', 0.0),
                                            self.parameters.get('Z-max', 1.0),
                                            self.nz) if z is None else z

        self.atom = None

        self.boundary_conditions = boundary_conditions
        self._setup_boundary_conditions()

    def set_parameter(self, name, value):
        '''Set a parameter in ``self.parameters``. Changing a value in
        ``self.parameters`` will make us recompute the linear part of
        the equation.

        Parameters
        ----------
        name : str
            Name of the parameter
        value : Any
            Value of the parameter. Generally a floating point value

        '''

        if name in self.parameters and self.get_parameter(name) == value:
            return

        self.parameters[name] = value

    def get_parameter(self, name, default=0):
        '''Get a parameter from ``self.parameters``.

        Parameters
        ----------
        name : str
            Name of the parameter
        default : Any, optional
            Default return value if the parameter is not found in
            ``self.parameters``. The default value is 0.

        Returns
        -------
        The value of the parameter

        '''

        if name not in self.parameters:
            return default

        return self.parameters.get(name, default)

    def get_coordinate_vector(self, start, end, nx):
        '''Get a coordinate vector according to the set parameters.

        Parameters
        ----------
        start : float
            Start of the domain
        end : float
            End of the domain
        nx : int
            Amount of steps

        '''

        if self.parameters.get('Grid Stretching', False) or 'Grid Stretching Factor' in self.parameters.keys():
            if self.parameters.get('Grid Stretching Method', 'tanh') == 'sin':
                return utils.create_stretched_coordinate_vector2(
                    start, end, nx, self.parameters.get('Grid Stretching Factor', 0.1))

            return utils.create_stretched_coordinate_vector(
                start, end, nx, self.parameters.get('Grid Stretching Factor', 1.5))

        return utils.create_uniform_coordinate_vector(start, end, nx)

    def linear_part(self):
        '''Compute the linear part of the equation. Return a cached version if
        possible.

        :meta private:

        '''

        parameters = dict(self.parameters)
        if parameters != self.old_parameters:
            if self.dim == 2:
                self.atom = self._linear_part_2D()
            else:
                self.atom = self._linear_part_3D()

            self.old_parameters = parameters

        return self.atom

    def _linear_part_2D(self):
        '''Compute the linear part of the equation in case the domain
        is 2D. In case Re = 0 we instead compute the linear part for
        the Stokes problem.

        '''

        Re = self.get_parameter('Reynolds Number', 1.0)
        Ra = self.get_parameter('Rayleigh Number', 1.0)
        Pr = self.get_parameter('Prandtl Number', 1.0)
        Gr = self.get_parameter('Grashof Number', Ra / Pr)
        Le = self.get_parameter('Lewis Number', 1.0)

        if Re == 0:
            Re = 1

        if Gr == 0:
            Gr = 1 / Pr

        atom = 1 / (Re * numpy.sqrt(Gr)) * (self.u_xx() + self.u_yy()
                                            + self.v_xx() + self.v_yy()) \
            - (self.p_x() + self.p_y()) \
            + self.div()

        beta = self.get_parameter('Rossby Parameter')
        if beta:
            atom -= beta * self.coriolis()

        if self.dof > 3:
            atom += 1 / (Pr * numpy.sqrt(Gr)) * (self.T_xx() + self.T_yy())
            atom += self.forward_average_T_y()

        if self.dof > 4:
            atom += 1 / (Le * Pr * numpy.sqrt(Gr)) * (self.S_xx() + self.S_yy())
            atom -= self.forward_average_S_y()

        if self.problem_type_equals('Rayleigh-Benard Perturbation'):
            Bi = self.get_parameter('Biot Number')
            atom += Bi / (Bi + 1) * self.backward_average_v_y()

        return atom

    def _linear_part_3D(self):
        '''Compute the linear part of the equation in case the domain
        is 3D. In case Re = 0 we instead compute the linear part for
        the Stokes problem.

        '''

        Re = self.get_parameter('Reynolds Number', 1.0)
        Ra = self.get_parameter('Rayleigh Number', 1.0)
        Pr = self.get_parameter('Prandtl Number', 1.0)
        Gr = self.get_parameter('Grashof Number', Ra / Pr)
        Le = self.get_parameter('Lewis Number', 1.0)

        if Re == 0:
            Re = 1

        if Gr == 0:
            Gr = 1 / Pr

        atom = 1 / (Re * numpy.sqrt(Gr)) * (self.u_xx() + self.u_yy() + self.u_zz()
                                            + self.v_xx() + self.v_yy() + self.v_zz()
                                            + self.w_xx() + self.w_yy() + self.w_zz()) \
            - (self.p_x() + self.p_y() + self.p_z()) \
            + self.div()

        if self.dof > 4:
            atom += 1 / (Pr * numpy.sqrt(Gr)) * (self.T_xx() + self.T_yy() + self.T_zz())
            if self.nz > 1:
                atom += self.forward_average_T_z()
            else:
                atom += self.forward_average_T_y()

        if self.dof > 5:
            atom += 1 / (Le * Pr * numpy.sqrt(Gr)) * (self.S_xx() + self.S_yy() + self.S_zz())
            if self.nz > 1:
                atom -= self.forward_average_S_z()
            else:
                atom -= self.forward_average_S_y()

        if self.problem_type_equals('Rayleigh-Benard Perturbation'):
            Bi = self.get_parameter('Biot Number')
            if self.nz > 1:
                atom += Bi / (Bi + 1) * self.backward_average_w_z()
            else:
                atom += Bi / (Bi + 1) * self.backward_average_v_y()

        return atom

    def nonlinear_part(self, state):
        r'''Compute the nonlinear part of the equation. In case $\Re =
        0$ this does nothing.

        :meta private:

        '''

        state_mtx = utils.create_padded_state_mtx(state, self.nx, self.ny, self.nz, self.dof,
                                                  self.x_periodic, self.y_periodic, self.z_periodic)

        atomJ = numpy.zeros((self.nx, self.ny, self.nz, self.dof, self.dof, 3, 3, 3))
        atomF = numpy.zeros((self.nx, self.ny, self.nz, self.dof, self.dof, 3, 3, 3))

        Re = self.get_parameter('Reynolds Number')
        if Re == 0 and not self.dof > self.dim + 1:
            return (atomJ, atomF)

        self.u_u_x(atomJ, atomF, state_mtx)
        self.u_v_x(atomJ, atomF, state_mtx)
        self.v_u_y(atomJ, atomF, state_mtx)
        self.v_v_y(atomJ, atomF, state_mtx)

        if self.dim > 2:
            self.u_w_x(atomJ, atomF, state_mtx)
            self.v_w_y(atomJ, atomF, state_mtx)
            self.w_u_z(atomJ, atomF, state_mtx)
            self.w_v_z(atomJ, atomF, state_mtx)
            self.w_w_z(atomJ, atomF, state_mtx)

        if self.dof > self.dim + 1:
            self.u_T_x(atomJ, atomF, state_mtx)
            self.v_T_y(atomJ, atomF, state_mtx)

            if self.dim > 2:
                self.w_T_z(atomJ, atomF, state_mtx)

        if self.dof > self.dim + 2:
            self.u_S_x(atomJ, atomF, state_mtx)
            self.v_S_y(atomJ, atomF, state_mtx)

            if self.dim > 2:
                self.w_S_z(atomJ, atomF, state_mtx)

        atomJ += atomF

        return (atomJ, atomF)

    def rhs(self, state):
        r'''Compute the right-hand side of the DAE. That is the
        right-hand side $F(u, p)$ in

        .. math:: M(p) \frac{\d u}{\d t} = F(u, p)

        Parameters
        ----------
        state : array_like
            State $u$ at which to evaluate $F(u, p)$

        Returns
        -------
        rhs : array_like
            The value of $F(u, p)$

        '''

        atomJ, atomF = self.nonlinear_part(state)
        atomF += self.linear_part()

        frc = self.boundaries(atomF)

        return self.assemble_rhs(state, atomF) + frc

    def jacobian(self, state):
        r'''Compute the Jacobian matrix $J(u, p)$ of the right-hand
        side of the DAE. That is the Jacobian matrix of $F(u, p)$ in

        .. math:: M(p) \frac{\d u}{\d t} = F(u, p)

        Parameters
        ----------
        state : array_like
            State $u$ at which to evaluate $J(u, p)$

        Returns
        -------
        jac : CrsMatrix
            The matrix $J(u, p)$ in CSR format

        '''

        atomJ, atomF = self.nonlinear_part(state)
        atomJ += self.linear_part()

        self.boundaries(atomJ)

        return self.assemble_jacobian(atomJ)

    def mass_matrix(self):
        r'''Compute the mass matrix of the DAE. That is the mass
        matrix $M(p)$ in

        .. math:: M(p) \frac{\d u}{\d t} = F(u, p)

        Returns
        -------
        mass : CrsMatrix
            The matrix $M(p)$ in CSR format

        '''

        atom = self.mass_x() + self.mass_y()
        if self.dim == 3:
            atom += self.mass_z()
        if self.dof > self.dim + 1:
            atom += self.mass_T()
        if self.dof > self.dim + 2:
            atom += self.mass_S()
        return self.assemble_mass_matrix(atom)

    def assemble_rhs(self, state, atom):
        '''Assemble the right-hand side. Optimized version of

        .. code-block:: Python

            for k in range(nz):
                for j in range(ny):
                    for i in range(nx):
                        for d1 in range(dof):
                            for z in range(3):
                                for y in range(3):
                                    for x in range(3):
                                        for d2 in range(dof):
                                            if abs(atom[i, j, k, d1, d2, x, y, z]) > 1e-14:
                                                offset = row + (x-1) * dof + (y-1) * nx * dof + \
                                                               (z-1) * nx * ny * dof + d2 - d1
                                                out[row] -= atom[i, j, k, d1, d2, x, y, z] * state[offset]
                            row += 1

        :meta private:

        '''

        # Put the state in shifted matrix form
        state_mtx = utils.create_padded_state_mtx(state, self.nx, self.ny, self.nz, self.dof,
                                                  self.x_periodic, self.y_periodic, self.z_periodic)

        # Add up all contributions without iterating over the domain
        out_mtx = numpy.zeros((self.nx, self.ny, self.nz, self.dof))
        for k, j, i, d1, d2 in numpy.ndindex(3, 3, 3, self.dof, self.dof):
            out_mtx[:, :, :, d1] += atom[:, :, :, d1, d2, i, j, k] \
                * state_mtx[i:(i+self.nx), j:(j+self.ny), k:(k+self.nz), d2]

        return utils.create_state_vec(out_mtx, self.nx, self.ny, self.nz, self.dof)

    def assemble_jacobian(self, atom):
        '''Assemble the Jacobian. Optimized version of

        .. code-block:: Python

            for k in range(nz):
                for j in range(ny):
                    for i in range(nx):
                        for d1 in range(dof):
                            for z in range(3):
                                for y in range(3):
                                    for x in range(3):
                                        for d2 in range(dof):
                                            if abs(atom[i, j, k, d1, d2, x, y, z]) > 1e-14:
                                               jcoA[idx] = row + (x-1) * dof + (y-1) * nx * dof + \
                                                                 (z-1) * nx * ny * dof + d2 - d1
                                               coA[idx] = atom[i, j, k, d1, d2, x, y, z]
                                               idx += 1
                            row += 1
                            begA[row] = idx

        :meta private:

        '''

        row = 0
        idx = 0
        n = self.nx * self.ny * self.nz * self.dof
        coA = numpy.zeros(27*n)
        jcoA = numpy.zeros(27*n, dtype=int)
        begA = numpy.zeros(n+1, dtype=int)

        # Check where values are nonzero in the atoms
        configs = []
        for d2, x, y, z in numpy.ndindex(self.dof, 3, 3, 3):
            if numpy.any(atom[:, :, :, :, d2, x, y, z]):
                configs.append((d2, x, y, z))

        # Iterate only over configurations with values in there
        for k, j, i, d1 in numpy.ndindex(self.nz, self.ny, self.nx, self.dof):
            for d2, x, y, z in configs:
                if abs(atom[i, j, k, d1, d2, x, y, z]) > 1e-14:
                    jcoA[idx] = ((i + x - 1) % self.nx) * self.dof \
                        + ((j + y - 1) % self.ny) * self.nx * self.dof + \
                        + ((k + z - 1) % self.nz) * self.nx * self.ny * self.dof + d2
                    coA[idx] = atom[i, j, k, d1, d2, x, y, z]
                    idx += 1
            row += 1
            begA[row] = idx

        return CrsMatrix(coA, jcoA, begA)

    def assemble_mass_matrix(self, atom):
        '''Assemble the mass matrix.

        :meta private:

        '''

        row = 0
        idx = 0
        n = self.nx * self.ny * self.nz * self.dof
        coA = numpy.zeros(n)
        jcoA = numpy.zeros(n, dtype=int)
        begA = numpy.zeros(n+1, dtype=int)

        for k, j, i, d in numpy.ndindex(self.nz, self.ny, self.nx, self.dof):
            if abs(atom[i, j, k, d]) > 1e-14:
                jcoA[idx] = (i + (j + k * self.ny) * self.nx) * self.dof + d
                coA[idx] = atom[i, j, k, d]
                idx += 1
            row += 1
            begA[row] = idx

        return CrsMatrix(coA, jcoA, begA)

    def problem_type_equals(self, second):
        '''Test against the problem type in ``self.parameterlist``.

        :meta private:

        '''
        first = self.get_parameter('Problem Type', 'Lid-driven Cavity')
        return first.lower() == second.lower()

    def set_dof(self):
        '''Set ``self.dof`` based on the problem type.

        :meta private:

        '''

        self.dof = self.dim + 1
        if self.problem_type_equals('Rayleigh-Benard') \
           or self.problem_type_equals('Rayleigh-Benard Perturbation'):
            self.dof = self.dim + 2
        elif self.problem_type_equals('Differentially Heated Cavity'):
            self.dof = self.dim + 2
        elif self.problem_type_equals('AMOC'):
            self.dof = self.dim + 3

    # Boundary conditions

    def _lid_driven_cavity(self, boundary_conditions, atom):
        '''Boundary conditions for the lid-driven cavity'''
        v = self.get_parameter('Lid Velocity', 1)
        boundary_conditions.no_slip_east(atom)
        boundary_conditions.no_slip_west(atom)

        boundary_conditions.no_slip_south(atom)
        if self.dim == 2 or self.nz <= 1:
            boundary_conditions.moving_lid_north(atom, v)
            return boundary_conditions.get_forcing()

        boundary_conditions.no_slip_north(atom)

        boundary_conditions.no_slip_bottom(atom)
        boundary_conditions.moving_lid_top(atom, v)

        return boundary_conditions.get_forcing()

    def _rayleigh_benard(self, boundary_conditions, atom):
        '''Boundary conditions for the Rayleigh-Benard problem'''
        asym = self.get_parameter('Asymmetry Parameter')
        boundary_conditions.heat_flux_east(atom, asym)
        boundary_conditions.heat_flux_west(atom, 0)
        boundary_conditions.no_slip_east(atom)
        boundary_conditions.no_slip_west(atom)

        Bi = self.get_parameter('Biot Number')
        bottom_temperature = 1 if self.problem_type_equals('Rayleigh-Benard') else 0

        if self.dim == 2 or self.nz <= 1:
            boundary_conditions.heat_flux_north(atom, 0, Bi)
            boundary_conditions.temperature_south(atom, bottom_temperature)
            boundary_conditions.free_slip_north(atom)
            boundary_conditions.no_slip_south(atom)
            return boundary_conditions.get_forcing()

        boundary_conditions.heat_flux_north(atom, 0)
        boundary_conditions.heat_flux_south(atom, 0)
        boundary_conditions.no_slip_north(atom)
        boundary_conditions.no_slip_south(atom)

        boundary_conditions.heat_flux_top(atom, 0, Bi)
        boundary_conditions.temperature_bottom(atom, bottom_temperature)
        boundary_conditions.free_slip_top(atom)
        boundary_conditions.no_slip_bottom(atom)

        return boundary_conditions.get_forcing()

    def _differentially_heated_cavity(self, boundary_conditions, atom):
        '''Boundary conditions for the differentially heated cavity'''
        boundary_conditions.temperature_east(atom, -1/2)
        boundary_conditions.temperature_west(atom, 1/2)
        boundary_conditions.no_slip_east(atom)
        boundary_conditions.no_slip_west(atom)

        boundary_conditions.heat_flux_north(atom, 0)
        boundary_conditions.heat_flux_south(atom, 0)
        boundary_conditions.no_slip_north(atom)
        boundary_conditions.no_slip_south(atom)

        if self.dim > 2 and self.nz > 1:
            boundary_conditions.heat_flux_top(atom, 0)
            boundary_conditions.heat_flux_bottom(atom, 0)
            boundary_conditions.no_slip_top(atom)
            boundary_conditions.no_slip_bottom(atom)

        return boundary_conditions.get_forcing()

    def _double_gyre(self, boundary_conditions, atom):
        '''Boundary conditions for the double-gyre QG problem'''
        frc = self.wind_stress()

        boundary_conditions.no_slip_east(atom)
        boundary_conditions.no_slip_west(atom)

        boundary_conditions.free_slip_north(atom)
        boundary_conditions.free_slip_south(atom)

        return frc

    def _amoc(self, boundary_conditions, atom):
        '''Boundary conditions for the 2D AMOC'''
        boundary_conditions.heat_flux_east(atom, 0)
        boundary_conditions.heat_flux_west(atom, 0)
        boundary_conditions.salinity_flux_east(atom, 0)
        boundary_conditions.salinity_flux_west(atom, 0)
        boundary_conditions.free_slip_east(atom)
        boundary_conditions.free_slip_west(atom)

        boundary_conditions.heat_flux_south(atom, 0)
        boundary_conditions.salinity_flux_south(atom, 0)
        boundary_conditions.free_slip_south(atom)

        x = utils.compute_coordinate_vector_centers(self.x)

        theta = self.get_parameter('Temperature Forcing')
        asym = self.get_parameter('Asymmetry Parameter')
        A = self.parameters.get('X-max', 1.0)

        T_S = numpy.zeros((self.nx + 2, self.nz + 2))
        T_S[:, 0] = 1 / 2 * ((1 - asym) * numpy.cos(2 * numpy.pi * (x / A - 1 / 2))
                             + asym * numpy.cos(numpy.pi * x / A) + 1)
        boundary_conditions.temperature_north(atom, theta * T_S)

        sigma = self.get_parameter('Freshwater Flux')
        p = 2

        Q_S = numpy.zeros((self.nx + 2, self.nz + 2))
        Q_S[:, 0] = 3 * numpy.cos(p * numpy.pi * (x / A - 1 / 2)) - 6 / (p * numpy.pi) * numpy.sin(p * numpy.pi / 2)
        boundary_conditions.salinity_flux_north(atom, sigma * Q_S)

        boundary_conditions.free_slip_north(atom)

        # Fix one salinity value
        row = self.dim + 2
        for k, j, i, d, z, y, x in numpy.ndindex(self.nz, self.ny, self.nx, self.dof, 3, 3, 3):
            if ((i + x - 1) % self.nx) * self.dof \
               + ((j + y - 1) % self.ny) * self.nx * self.dof + \
               + ((k + z - 1) % self.nz) * self.nx * self.ny * self.dof + self.dim + 2 == row:
                atom[i, j, k, d, self.dim+2, x, y, z] = 0

        atom[0, 0, 0, self.dim+2, self.dim+2, 1, 1, 1] = -1

        frc = boundary_conditions.get_forcing()
        frc[row] = 0
        return frc

    def boundaries(self, atom):
        '''Apply the boundary conditions specified in
        ``self.boundary_conditions``.

        :meta private:

        '''
        boundary_conditions = BoundaryConditions(
            self.nx, self.ny, self.nz, self.dim, self.dof, self.x, self.y, self.z)

        return self.boundary_conditions(boundary_conditions, atom)

    def _setup_boundary_conditions(self):

        '''Setup boundary conditions for the currently defined problem type.'''
        if self.boundary_conditions:
            return
        elif self.problem_type_equals('Lid-driven Cavity'):
            self.boundary_conditions = self._lid_driven_cavity
        elif (self.problem_type_equals('Rayleigh-Benard')
              or self.problem_type_equals('Rayleigh-Benard Perturbation')):
            self.boundary_conditions = self._rayleigh_benard
        elif self.problem_type_equals('Differentially Heated Cavity'):
            self.boundary_conditions = self._differentially_heated_cavity
        elif self.problem_type_equals('Double Gyre'):
            self.boundary_conditions = self._double_gyre
        elif self.problem_type_equals('AMOC'):
            self.boundary_conditions = self._amoc
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

        # second order finite difference
        atom[0] = 1 / dx * dy * dz
        atom[2] = 1 / dxp1 * dy * dz
        atom[1] = -atom[0] - atom[2]

    def u_xx(self):
        ''':meta private:'''
        atom = numpy.zeros((self.nx, self.ny, self.nz, self.dof, self.dof, 3, 3, 3))
        for i, j, k in numpy.ndindex(self.nx, self.ny, self.nz):
            self._u_xx(atom[i, j, k, 0, 0, :, 1, 1], i, j, k, self.x, self.y, self.z)
        return atom

    def v_yy(self):
        ''':meta private:'''
        atom = numpy.zeros((self.nx, self.ny, self.nz, self.dof, self.dof, 3, 3, 3))
        for i, j, k in numpy.ndindex(self.nx, self.ny, self.nz):
            self._u_xx(atom[i, j, k, 1, 1, 1, :, 1], j, i, k, self.y, self.x, self.z)
        return atom

    def w_zz(self):
        ''':meta private:'''
        atom = numpy.zeros((self.nx, self.ny, self.nz, self.dof, self.dof, 3, 3, 3))
        for i, j, k in numpy.ndindex(self.nx, self.ny, self.nz):
            self._u_xx(atom[i, j, k, 2, 2, 1, 1, :], k, j, i, self.z, self.y, self.x)
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
        ''':meta private:'''
        atom = numpy.zeros((self.nx, self.ny, self.nz, self.dof, self.dof, 3, 3, 3))
        for i, j, k in numpy.ndindex(self.nx, self.ny, self.nz):
            self._u_yy(atom[i, j, k, 0, 0, 1, :, 1], i, j, k, self.x, self.y, self.z)
        return atom

    def v_xx(self):
        ''':meta private:'''
        atom = numpy.zeros((self.nx, self.ny, self.nz, self.dof, self.dof, 3, 3, 3))
        for i, j, k in numpy.ndindex(self.nx, self.ny, self.nz):
            self._u_yy(atom[i, j, k, 1, 1, :, 1, 1], j, i, k, self.y, self.x, self.z)
        return atom

    def w_yy(self):
        ''':meta private:'''
        atom = numpy.zeros((self.nx, self.ny, self.nz, self.dof, self.dof, 3, 3, 3))
        for i, j, k in numpy.ndindex(self.nx, self.ny, self.nz):
            self._u_yy(atom[i, j, k, 2, 2, 1, :, 1], k, j, i, self.z, self.y, self.x)
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
        ''':meta private:'''
        atom = numpy.zeros((self.nx, self.ny, self.nz, self.dof, self.dof, 3, 3, 3))
        for i, j, k in numpy.ndindex(self.nx, self.ny, self.nz):
            self._u_zz(atom[i, j, k, 0, 0, 1, 1, :], i, j, k, self.x, self.y, self.z)
        return atom

    def v_zz(self):
        ''':meta private:'''
        atom = numpy.zeros((self.nx, self.ny, self.nz, self.dof, self.dof, 3, 3, 3))
        for i, j, k in numpy.ndindex(self.nx, self.ny, self.nz):
            self._u_zz(atom[i, j, k, 1, 1, 1, 1, :], j, i, k, self.y, self.x, self.z)
        return atom

    def w_xx(self):
        ''':meta private:'''
        atom = numpy.zeros((self.nx, self.ny, self.nz, self.dof, self.dof, 3, 3, 3))
        for i, j, k in numpy.ndindex(self.nx, self.ny, self.nz):
            self._u_zz(atom[i, j, k, 2, 2, :, 1, 1], k, j, i, self.z, self.y, self.x)
        return atom

    @staticmethod
    def _C_xx(atom, i, j, k, x, y, z):
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

    def C_xx(self, var):
        ''':meta private:'''
        atom = numpy.zeros((self.nx, self.ny, self.nz, self.dof, self.dof, 3, 3, 3))
        for i, j, k in numpy.ndindex(self.nx, self.ny, self.nz):
            self._C_xx(atom[i, j, k, var, var, :, 1, 1], i, j, k, self.x, self.y, self.z)
        return atom

    def C_yy(self, var):
        ''':meta private:'''
        atom = numpy.zeros((self.nx, self.ny, self.nz, self.dof, self.dof, 3, 3, 3))
        for i, j, k in numpy.ndindex(self.nx, self.ny, self.nz):
            self._C_xx(atom[i, j, k, var, var, 1, :, 1], j, i, k, self.y, self.x, self.z)
        return atom

    def C_zz(self, var):
        ''':meta private:'''
        atom = numpy.zeros((self.nx, self.ny, self.nz, self.dof, self.dof, 3, 3, 3))
        for i, j, k in numpy.ndindex(self.nx, self.ny, self.nz):
            self._C_xx(atom[i, j, k, var, var, 1, 1, :], k, j, i, self.z, self.y, self.x)
        return atom

    def T_xx(self):
        ''':meta private:'''
        return self.C_xx(self.dim + 1)

    def T_yy(self):
        ''':meta private:'''
        return self.C_yy(self.dim + 1)

    def T_zz(self):
        ''':meta private:'''
        return self.C_zz(self.dim + 1)

    def S_xx(self):
        ''':meta private:'''
        return self.C_xx(self.dim + 2)

    def S_yy(self):
        ''':meta private:'''
        return self.C_yy(self.dim + 2)

    def S_zz(self):
        ''':meta private:'''
        return self.C_zz(self.dim + 2)

    @staticmethod
    def _forward_u_x(atom, i, j, k, x, y, z):
        # volume size in the y direction
        dy = y[j] - y[j-1]
        # volume size in the z direction
        dz = z[k] - z[k-1]

        # forward difference
        atom[2] = dy * dz
        atom[1] = -atom[2]

    def C_x(self, var):
        ''':meta private:'''
        atom = numpy.zeros((self.nx, self.ny, self.nz, self.dof, self.dof, 3, 3, 3))
        for i, j, k in numpy.ndindex(self.nx, self.ny, self.nz):
            self._forward_u_x(atom[i, j, k, 0, var, :, 1, 1], i, j, k, self.x, self.y, self.z)
        return atom

    def C_y(self, var):
        ''':meta private:'''
        atom = numpy.zeros((self.nx, self.ny, self.nz, self.dof, self.dof, 3, 3, 3))
        for i, j, k in numpy.ndindex(self.nx, self.ny, self.nz):
            self._forward_u_x(atom[i, j, k, 1, var, 1, :, 1], j, i, k, self.y, self.x, self.z)
        return atom

    def C_z(self, var):
        ''':meta private:'''
        atom = numpy.zeros((self.nx, self.ny, self.nz, self.dof, self.dof, 3, 3, 3))
        for i, j, k in numpy.ndindex(self.nx, self.ny, self.nz):
            self._forward_u_x(atom[i, j, k, 2, var, 1, 1, :], k, j, i, self.z, self.y, self.x)
        return atom

    def p_x(self):
        ''':meta private:'''
        return self.C_x(self.dim)

    def p_y(self):
        ''':meta private:'''
        return self.C_y(self.dim)

    def p_z(self):
        ''':meta private:'''
        return self.C_z(self.dim)

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
        ''':meta private:'''
        atom = numpy.zeros((self.nx, self.ny, self.nz, self.dof, self.dof, 3, 3, 3))
        for i, j, k in numpy.ndindex(self.nx, self.ny, self.nz):
            self._backward_u_x(atom[i, j, k, self.dim, 0, :, 1, 1], i, j, k, self.x, self.y, self.z)
        return atom

    def v_y(self):
        ''':meta private:'''
        atom = numpy.zeros((self.nx, self.ny, self.nz, self.dof, self.dof, 3, 3, 3))
        for i, j, k in numpy.ndindex(self.nx, self.ny, self.nz):
            self._backward_u_x(atom[i, j, k, self.dim, 1, 1, :, 1], j, i, k, self.y, self.x, self.z)
        return atom

    def w_z(self):
        ''':meta private:'''
        atom = numpy.zeros((self.nx, self.ny, self.nz, self.dof, self.dof, 3, 3, 3))
        for i, j, k in numpy.ndindex(self.nx, self.ny, self.nz):
            self._backward_u_x(atom[i, j, k, self.dim, 2, 1, 1, :], k, j, i, self.z, self.y, self.x)
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
        ''':meta private:'''
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

    def forward_average_C_y(self, var):
        ''':meta private:'''
        atom = numpy.zeros((self.nx, self.ny, self.nz, self.dof, self.dof, 3, 3, 3))
        for i, j, k in numpy.ndindex(self.nx, self.ny, self.nz):
            self._forward_average_x(atom[i, j, k, 1, var, 1, :, 1], j, i, k, self.y, self.x, self.z)
        return atom

    def forward_average_C_z(self, var):
        ''':meta private:'''
        atom = numpy.zeros((self.nx, self.ny, self.nz, self.dof, self.dof, 3, 3, 3))
        for i, j, k in numpy.ndindex(self.nx, self.ny, self.nz):
            self._forward_average_x(atom[i, j, k, 2, var, 1, 1, :], k, j, i, self.z, self.y, self.x)
        return atom

    def forward_average_T_y(self):
        ''':meta private:'''
        return self.forward_average_C_y(self.dim + 1)

    def forward_average_T_z(self):
        ''':meta private:'''
        return self.forward_average_C_z(self.dim + 1)

    def forward_average_S_y(self):
        ''':meta private:'''
        return self.forward_average_C_y(self.dim + 2)

    def forward_average_S_z(self):
        ''':meta private:'''
        return self.forward_average_C_z(self.dim + 2)

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
        ''':meta private:'''
        atom = numpy.zeros((self.nx, self.ny, self.nz, self.dof, self.dof, 3, 3, 3))
        for i, j, k in numpy.ndindex(self.nx, self.ny, self.nz):
            self._backward_average_x(atom[i, j, k, self.dim+1, 1, 1, :, 1], j, i, k, self.y, self.x, self.z)
        return atom

    def backward_average_w_z(self):
        ''':meta private:'''
        atom = numpy.zeros((self.nx, self.ny, self.nz, self.dof, self.dof, 3, 3, 3))
        for i, j, k in numpy.ndindex(self.nx, self.ny, self.nz):
            self._backward_average_x(atom[i, j, k, self.dim+1, 2, 1, 1, :], k, j, i, self.z, self.y, self.x)
        return atom

    def coriolis(self):
        ''':meta private:'''
        atom = numpy.zeros((self.nx, self.ny, self.nz, self.dof, self.dof, 3, 3, 3))
        for i, j, k in numpy.ndindex(self.nx, self.ny, self.nz):
            # Value of yu at the position of v
            self._forward_average_x(atom[i, j, k, 1, 0, 0, :, 1], j, i, k, self.y, self.x, self.z)
            self._forward_average_x(atom[i, j, k, 1, 0, 1, :, 1], j, i, k, self.y, self.x, self.z)
            atom[i, j, k, 1, 0, :, :, :] *= self.y[j] / 2

            # Value of -yv at the position of u
            self._forward_average_x(atom[i, j, k, 0, 1, :, 0, 1], i, j, k, self.x, self.y, self.z)
            self._forward_average_x(atom[i, j, k, 0, 1, :, 1, 1], i, j, k, self.x, self.y, self.z)
            atom[i, j, k, 0, 1, :, :, :] *= -(self.y[j] + self.y[j-1]) / 4
        return atom

    def wind_stress(self):
        ''':meta private:'''
        alpha = self.get_parameter('Wind Stress Parameter')
        asym = self.get_parameter('Asymmetry Parameter')

        frc = numpy.zeros((self.nx, self.ny, self.nz, self.dof))
        for i, j, k in numpy.ndindex(self.nx - 1, self.ny, self.nz):
            # volume size in the x direction
            dx = (self.x[i+1] - self.x[i-1]) / 2
            # volume size in the y direction
            dy = self.y[j] - self.y[j-1]
            # volume size in the z direction
            dz = self.z[k] - self.z[k-1]

            y = (self.y[j] + self.y[j-1]) / 2
            frc[i, j, k, 0] = - (1 - asym) * numpy.cos(2 * numpy.pi * y) - asym * numpy.cos(numpy.pi * y)
            frc[i, j, k, 0] *= alpha / (2 * numpy.pi) * dx * dy * dz
        return utils.create_state_vec(frc, self.nx, self.ny, self.nz, self.dof)

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
        ''':meta private:'''
        atom = numpy.zeros((self.nx, self.ny, self.nz, self.dof))
        for i, j, k in numpy.ndindex(self.nx, self.ny, self.nz):
            self._mass_x(atom[i, j, k, 0:1], i, j, k, self.x, self.y, self.z)
        return atom

    def mass_y(self):
        ''':meta private:'''
        atom = numpy.zeros((self.nx, self.ny, self.nz, self.dof))
        for i, j, k in numpy.ndindex(self.nx, self.ny, self.nz):
            self._mass_x(atom[i, j, k, 1:2], j, i, k, self.y, self.x, self.z)
        return atom

    def mass_z(self):
        ''':meta private:'''
        atom = numpy.zeros((self.nx, self.ny, self.nz, self.dof))
        for i, j, k in numpy.ndindex(self.nx, self.ny, self.nz):
            self._mass_x(atom[i, j, k, 2:3], k, j, i, self.z, self.y, self.x)
        return atom

    @staticmethod
    def _mass_C(atom, i, j, k, x, y, z):
        # volume size in the x direction
        dx = x[i] - x[i-1]
        # volume size in the y direction
        dy = y[j] - y[j-1]
        # volume size in the z direction
        dz = z[k] - z[k-1]

        atom[0] = dx * dy * dz

    def mass_C(self, var):
        ''':meta private:'''
        atom = numpy.zeros((self.nx, self.ny, self.nz, self.dof))
        for i, j, k in numpy.ndindex(self.nx, self.ny, self.nz):
            self._mass_C(atom[i, j, k, var:var+1], i, j, k, self.x, self.y, self.z)
        return atom

    def mass_T(self):
        ''':meta private:'''
        return self.mass_C(self.dim + 1)

    def mass_S(self):
        ''':meta private:'''
        return self.mass_C(self.dim + 2)

    def average_x(self, state):
        ''':meta private:'''
        averages = numpy.zeros((self.nx+1, self.ny, self.nz))

        cropped_state = state[:, 1:self.ny+1, 1:self.nz+1]

        averages[:, :, :] += 1/2 * cropped_state[0:self.nx+1, :, :]
        averages[:, :, :] += 1/2 * cropped_state[1:self.nx+2, :, :]

        return averages

    def average_y(self, state):
        ''':meta private:'''
        averages = numpy.zeros((self.nx, self.ny+1, self.nz))

        cropped_state = state[1:self.nx+1, :, 1:self.nz+1]

        averages[:, :, :] += 1/2 * cropped_state[:, 0:self.ny+1, :]
        averages[:, :, :] += 1/2 * cropped_state[:, 1:self.ny+2, :]

        return averages

    def average_z(self, state):
        ''':meta private:'''
        averages = numpy.zeros((self.nx, self.ny, self.nz+1))

        cropped_state = state[1:self.nx+1, 1:self.ny+1, :]

        averages[:, :, :] += 1/2 * cropped_state[:, :, 0:self.nz+1]
        averages[:, :, :] += 1/2 * cropped_state[:, :, 1:self.nz+2]

        return averages

    @staticmethod
    def _weighted_average(atom, i, x):
        # volume size in the x direction
        dx = (x[i+1] - x[i-1]) / 2

        # volume sizes associated with the v velocities
        dxmh = x[i] - x[i-1]
        dxph = x[i+1] - x[i]

        atom[0] = 1 / 2 * dxmh / dx
        atom[1] = 1 / 2 * dxph / dx

    def weighted_average_x(self, state):
        ''':meta private:'''
        averages = numpy.zeros((self.nx, self.ny+1, self.nz+1))

        cropped_state = state[:, 0:self.ny+1, 0:self.nz+1]

        atom = numpy.zeros(2)
        for i in range(self.nx):
            self._weighted_average(atom, i, self.x)

            averages[i, :, :] += atom[0] * cropped_state[i+1, :, :]
            averages[i, :, :] += atom[1] * cropped_state[i+2, :, :]

        return averages

    def weighted_average_y(self, state):
        ''':meta private:'''
        averages = numpy.zeros((self.nx+1, self.ny, self.nz+1))

        cropped_state = state[0:self.nx+1, :, 0:self.nz+1]

        atom = numpy.zeros(2)
        for j in range(self.ny):
            self._weighted_average(atom, j, self.y)

            averages[:, j, :] += atom[0] * cropped_state[:, j+1, :]
            averages[:, j, :] += atom[1] * cropped_state[:, j+2, :]

        return averages

    def weighted_average_z(self, state):
        ''':meta private:'''
        averages = numpy.zeros((self.nx+1, self.ny+1, self.nz))

        cropped_state = state[0:self.nx+1, 0:self.ny+1, :]

        atom = numpy.zeros(2)
        for k in range(self.nz):
            self._weighted_average(atom, k, self.z)

            averages[:, :, k] += atom[0] * cropped_state[:, :, k+1]
            averages[:, :, k] += atom[1] * cropped_state[:, :, k+2]

        return averages

    def u_u_x(self, atomJ, atomF, state):
        ''':meta private:'''
        averages = self.average_x(state[:, :, :, 0])

        atom = numpy.zeros(3)
        for i, j, k in numpy.ndindex(self.nx, self.ny, self.nz):
            self._forward_u_x(atom, i, j, k, self.x, self.y, self.z)
            atomF[i, j, k, 0, 0, 0:2, 1, 1] -= atom[1] * averages[i, j, k] * 1 / 2
            atomF[i, j, k, 0, 0, 1:3, 1, 1] -= atom[2] * averages[i+1, j, k] * 1 / 2

            atomJ[i, j, k, 0, 0, 0:2, 1, 1] -= atom[1] * averages[i, j, k] * 1 / 2
            atomJ[i, j, k, 0, 0, 1:3, 1, 1] -= atom[2] * averages[i+1, j, k] * 1 / 2

    def u_v_x(self, atomJ, atomF, state):
        ''':meta private:'''
        averages_u = self.weighted_average_y(state[:, :, :, 0])
        averages_v = self.average_x(state[:, :, :, 1])

        atom = numpy.zeros(3)
        atom_average = numpy.zeros(2)
        for i, j, k in numpy.ndindex(self.nx, self.ny, self.nz):
            self._backward_u_y(atom, j, i, k, self.y, self.x, self.z)
            atomF[i, j, k, 1, 1, 0:2, 1, 1] -= atom[0] * averages_u[i, j, k+1] * 1 / 2
            atomF[i, j, k, 1, 1, 1:3, 1, 1] -= atom[1] * averages_u[i+1, j, k+1] * 1 / 2

            self._weighted_average(atom_average, j, self.y)
            atomJ[i, j, k, 1, 0, 0, 1:3, 1] -= atom[0] * averages_v[i, j, k] * atom_average
            atomJ[i, j, k, 1, 0, 1, 1:3, 1] -= atom[1] * averages_v[i+1, j, k] * atom_average

    def u_w_x(self, atomJ, atomF, state):
        ''':meta private:'''
        averages_u = self.weighted_average_z(state[:, :, :, 0])
        averages_w = self.average_x(state[:, :, :, 2])

        atom = numpy.zeros(3)
        atom_average = numpy.zeros(2)
        for i, j, k in numpy.ndindex(self.nx, self.ny, self.nz):
            self._backward_u_z(atom, k, j, i, self.z, self.y, self.x)
            atomF[i, j, k, 2, 2, 0:2, 1, 1] -= atom[0] * averages_u[i, j+1, k] * 1 / 2
            atomF[i, j, k, 2, 2, 1:3, 1, 1] -= atom[1] * averages_u[i+1, j+1, k] * 1 / 2

            self._weighted_average(atom_average, k, self.z)
            atomJ[i, j, k, 2, 0, 0, 1, 1:3] -= atom[0] * averages_w[i, j, k] * atom_average
            atomJ[i, j, k, 2, 0, 1, 1, 1:3] -= atom[1] * averages_w[i+1, j, k] * atom_average

    def u_C_x(self, atomJ, atomF, state, var):
        ''':meta private:'''
        averages_u = state[0:self.nx+1, 1:self.ny+1, 1:self.nz+1, 0]
        averages_C = self.average_x(state[:, :, :, var])

        atom = numpy.zeros(3)
        for i, j, k in numpy.ndindex(self.nx, self.ny, self.nz):
            self._backward_u_x(atom, i, j, k, self.x, self.y, self.z)
            atomF[i, j, k, var, var, 0:2, 1, 1] -= atom[0] * averages_u[i, j, k] * 1/2
            atomF[i, j, k, var, var, 1:3, 1, 1] -= atom[1] * averages_u[i+1, j, k] * 1/2

            atomJ[i, j, k, var, 0, 0, 1, 1] -= atom[0] * averages_C[i, j, k]
            atomJ[i, j, k, var, 0, 1, 1, 1] -= atom[1] * averages_C[i+1, j, k]

    def u_T_x(self, atomJ, atomF, state):
        ''':meta private:'''
        return self.u_C_x(atomJ, atomF, state, self.dim + 1)

    def u_S_x(self, atomJ, atomF, state):
        ''':meta private:'''
        return self.u_C_x(atomJ, atomF, state, self.dim + 2)

    def v_u_y(self, atomJ, atomF, state):
        ''':meta private:'''
        averages_u = self.average_y(state[:, :, :, 0])
        averages_v = self.weighted_average_x(state[:, :, :, 1])

        atom = numpy.zeros(3)
        atom_average = numpy.zeros(2)
        for i, j, k in numpy.ndindex(self.nx, self.ny, self.nz):
            self._backward_u_y(atom, i, j, k, self.x, self.y, self.z)
            atomF[i, j, k, 0, 0, 1, 0:2, 1] -= atom[0] * averages_v[i, j, k+1] * 1 / 2
            atomF[i, j, k, 0, 0, 1, 1:3, 1] -= atom[1] * averages_v[i, j+1, k+1] * 1 / 2

            self._weighted_average(atom_average, i, self.x)
            atomJ[i, j, k, 0, 1, 1:3, 0, 1] -= atom[0] * averages_u[i, j, k] * atom_average
            atomJ[i, j, k, 0, 1, 1:3, 1, 1] -= atom[1] * averages_u[i, j+1, k] * atom_average

    def v_v_y(self, atomJ, atomF, state):
        ''':meta private:'''
        averages = self.average_y(state[:, :, :, 1])

        atom = numpy.zeros(3)
        for i, j, k in numpy.ndindex(self.nx, self.ny, self.nz):
            self._forward_u_x(atom, j, i, k, self.y, self.x, self.z)
            atomF[i, j, k, 1, 1, 1, 0:2, 1] -= atom[1] * averages[i, j, k] * 1 / 2
            atomF[i, j, k, 1, 1, 1, 1:3, 1] -= atom[2] * averages[i, j+1, k] * 1 / 2

            atomJ[i, j, k, 1, 1, 1, 0:2, 1] -= atom[1] * averages[i, j, k] * 1 / 2
            atomJ[i, j, k, 1, 1, 1, 1:3, 1] -= atom[2] * averages[i, j+1, k] * 1 / 2

    def v_w_y(self, atomJ, atomF, state):
        ''':meta private:'''
        averages_v = self.weighted_average_z(state[:, :, :, 1])
        averages_w = self.average_y(state[:, :, :, 2])

        atom = numpy.zeros(3)
        atom_average = numpy.zeros(2)
        for i, j, k in numpy.ndindex(self.nx, self.ny, self.nz):
            self._backward_u_y(atom, k, j, i, self.z, self.y, self.x)
            atomF[i, j, k, 2, 2, 1, 0:2, 1] -= atom[0] * averages_v[i+1, j, k] * 1 / 2
            atomF[i, j, k, 2, 2, 1, 1:3, 1] -= atom[1] * averages_v[i+1, j+1, k] * 1 / 2

            self._weighted_average(atom_average, k, self.z)
            atomJ[i, j, k, 2, 1, 1, 0, 1:3] -= atom[0] * averages_w[i, j, k] * atom_average
            atomJ[i, j, k, 2, 1, 1, 1, 1:3] -= atom[1] * averages_w[i, j+1, k] * atom_average

    def v_C_y(self, atomJ, atomF, state, var):
        ''':meta private:'''
        averages_v = state[1:self.nx+1, 0:self.ny+1, 1:self.nz+1, 1]
        averages_C = self.average_y(state[:, :, :, var])

        atom = numpy.zeros(3)
        for i, j, k in numpy.ndindex(self.nx, self.ny, self.nz):
            self._backward_u_x(atom, j, i, k, self.y, self.x, self.z)
            atomF[i, j, k, var, var, 1, 0:2, 1] -= atom[0] * averages_v[i, j, k] * 1/2
            atomF[i, j, k, var, var, 1, 1:3, 1] -= atom[1] * averages_v[i, j+1, k] * 1/2

            atomJ[i, j, k, var, 1, 1, 0, 1] -= atom[0] * averages_C[i, j, k]
            atomJ[i, j, k, var, 1, 1, 1, 1] -= atom[1] * averages_C[i, j+1, k]

    def v_T_y(self, atomJ, atomF, state):
        ''':meta private:'''
        return self.v_C_y(atomJ, atomF, state, self.dim + 1)

    def v_S_y(self, atomJ, atomF, state):
        ''':meta private:'''
        return self.v_C_y(atomJ, atomF, state, self.dim + 2)

    def w_u_z(self, atomJ, atomF, state):
        ''':meta private:'''
        averages_u = self.average_z(state[:, :, :, 0])
        averages_w = self.weighted_average_x(state[:, :, :, 2])

        atom = numpy.zeros(3)
        atom_average = numpy.zeros(2)
        for i, j, k in numpy.ndindex(self.nx, self.ny, self.nz):
            self._backward_u_z(atom, i, j, k, self.x, self.y, self.z)
            atomF[i, j, k, 0, 0, 1, 1, 0:2] -= atom[0] * averages_w[i, j+1, k] * 1 / 2
            atomF[i, j, k, 0, 0, 1, 1, 1:3] -= atom[1] * averages_w[i, j+1, k+1] * 1 / 2

            self._weighted_average(atom_average, i, self.x)
            atomJ[i, j, k, 0, 2, 1:3, 1, 0] -= atom[0] * averages_u[i, j, k] * atom_average
            atomJ[i, j, k, 0, 2, 1:3, 1, 1] -= atom[1] * averages_u[i, j, k+1] * atom_average

    def w_v_z(self, atomJ, atomF, state):
        ''':meta private:'''
        averages_v = self.average_z(state[:, :, :, 1])
        averages_w = self.weighted_average_y(state[:, :, :, 2])

        atom = numpy.zeros(3)
        atom_average = numpy.zeros(2)
        for i, j, k in numpy.ndindex(self.nx, self.ny, self.nz):
            self._backward_u_z(atom, j, i, k, self.y, self.x, self.z)
            atomF[i, j, k, 1, 1, 1, 1, 0:2] -= atom[0] * averages_w[i+1, j, k] * 1 / 2
            atomF[i, j, k, 1, 1, 1, 1, 1:3] -= atom[1] * averages_w[i+1, j, k+1] * 1 / 2

            self._weighted_average(atom_average, j, self.y)
            atomJ[i, j, k, 1, 2, 1, 1:3, 0] -= atom[0] * averages_v[i, j, k] * atom_average
            atomJ[i, j, k, 1, 2, 1, 1:3, 1] -= atom[1] * averages_v[i, j, k+1] * atom_average

    def w_w_z(self, atomJ, atomF, state):
        ''':meta private:'''
        averages = self.average_z(state[:, :, :, 2])

        atom = numpy.zeros(3)
        for i, j, k in numpy.ndindex(self.nx, self.ny, self.nz):
            self._forward_u_x(atom, k, j, i, self.z, self.y, self.x)
            atomF[i, j, k, 2, 2, 1, 1, 0:2] -= atom[1] * averages[i, j, k] * 1 / 2
            atomF[i, j, k, 2, 2, 1, 1, 1:3] -= atom[2] * averages[i, j, k+1] * 1 / 2

            atomJ[i, j, k, 2, 2, 1, 1, 0:2] -= atom[1] * averages[i, j, k] * 1 / 2
            atomJ[i, j, k, 2, 2, 1, 1, 1:3] -= atom[2] * averages[i, j, k+1] * 1 / 2

    def w_C_z(self, atomJ, atomF, state, var):
        ''':meta private:'''
        averages_w = state[1:self.nx+1, 1:self.ny+1, 0:self.nz+1, 2]
        averages_C = self.average_z(state[:, :, :, var])

        atom = numpy.zeros(3)
        for i, j, k in numpy.ndindex(self.nx, self.ny, self.nz):
            self._backward_u_x(atom, k, j, i, self.z, self.y, self.x)
            atomF[i, j, k, var, var, 1, 1, 0:2] -= atom[0] * averages_w[i, j, k] * 1/2
            atomF[i, j, k, var, var, 1, 1, 1:3] -= atom[1] * averages_w[i, j, k+1] * 1/2

            atomJ[i, j, k, var, 2, 1, 1, 0] -= atom[0] * averages_C[i, j, k]
            atomJ[i, j, k, var, 2, 1, 1, 1] -= atom[1] * averages_C[i, j, k+1]

    def w_T_z(self, atomJ, atomF, state):
        ''':meta private:'''
        return self.w_C_z(atomJ, atomF, state, self.dim + 1)

    def w_S_z(self, atomJ, atomF, state):
        ''':meta private:'''
        return self.w_C_z(atomJ, atomF, state, self.dim + 2)
