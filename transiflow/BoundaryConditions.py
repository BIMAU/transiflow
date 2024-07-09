import numpy

from .utils import create_state_vec

class BoundaryConditions:
    '''Boundary conditions for the Discretization class. Every method
    expects and atom, modifies it accordingly and also modifies the
    forcing that is stored internally. This class is only used from
    within the Discretization class.

    Parameters
    ----------
    nx : int
        Grid size in the x direction.
    ny : int
        Grid size in the y direction.
    nz : int
        Grid size in the z direction. This is set to 1 for
        2-dimensional problems.
    dim : int
        Physical dimension of the problem. In case this is set to 2, w
        is not referenced in the state vector.
    dof : int
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

    '''

    def __init__(self, nx, ny, nz, dim, dof, x, y, z):

        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.dim = dim
        self.dof = dof

        self.x = x
        self.y = y
        self.z = z

        self.frc = numpy.zeros((self.nx, self.ny, self.nz, self.dof))

    def get_forcing(self):
        '''Return the forcing that is required to be added the the RHS.'''
        return create_state_vec(self.frc, self.nx, self.ny, self.nz, self.dof)

    def no_slip_east(self, atom):
        '''Apply a no slip boundary condition at the east boundary. At this
        boundary $u_i = 0$, $(v_i + v_{i+1}) / 2 = 0$ and similar for
        $w$. So we replace $v_{i+1}$ by $-v_i$, $w_{i+1}$ by $-w_i$,
        set $u_i$ to zero and set all references to cells outside the
        domain to zero as well.

        '''
        atom[self.nx-1, :, :, :, :, 1, :, :] -= atom[self.nx-1, :, :, :, :, 2, :, :]
        atom[self.nx-1, :, :, :, 0, 1, :, :] = 0
        atom[self.nx-1, :, :, 0, :, :, :, :] = 0
        atom[self.nx-1, :, :, :, :, 2, :, :] = 0
        atom[self.nx-1, :, :, 0, 0, 1, 1, 1] = -1
        # TODO: Do we need this?
        atom[self.nx-2, :, :, 0, 0, 2, :, :] = 0

        self.frc[self.nx-1, :, :, 0] = 0

    def no_slip_west(self, atom):
        '''Apply a no slip boundary condition at the west boundary. At this
        boundary $u_{i-1} = 0$, $(v_{i-1} + v_i) / 2 = 0$ and similar
        for $w$. So we simply replace $v_{i-1}$ by $-v_i$, $w_{i-1}$
        by $-w_i$ and set all references to cells outside the domain
        to zero.

        '''
        atom[0, :, :, :, 0, 0, :, :] = 0
        atom[0, :, :, :, :, 1, :, :] -= atom[0, :, :, :, :, 0, :, :]
        atom[0, :, :, :, :, 0, :, :] = 0

    def no_slip_north(self, atom):
        '''Apply a no slip boundary condition at the north boundary. At this
        boundary $v_j = 0$, $(u_j + u_{j+1}) / 2 = 0$ and similar for
        $w$. So we replace $u_{j+1}$ by $-u_j$, $w_{j+1}$ by $-w_j$,
        set $v_j$ to zero and set all references to cells outside the
        domain to zero as well.

        '''
        atom[:, self.ny-1, :, :, :, :, 1, :] -= atom[:, self.ny-1, :, :, :, :, 2, :]
        atom[:, self.ny-1, :, :, 1, :, 1, :] = 0
        atom[:, self.ny-1, :, 1, :, :, :, :] = 0
        atom[:, self.ny-1, :, :, :, :, 2, :] = 0
        atom[:, self.ny-1, :, 1, 1, 1, 1, 1] = -1
        # TODO: Do we need this?
        atom[:, self.ny-2, :, 1, 1, :, 2, :] = 0

        self.frc[:, self.ny-1, :, 1] = 0

    def no_slip_south(self, atom):
        '''Apply a no slip boundary condition at the south boundary. At this
        boundary $v_{j-1} = 0$, $(u_{j-1} + u_j) / 2 = 0$ and similar
        for $w$. So we simply replace $u_{j-1}$ by $-u_j$, $w_{j-1}$
        by $-w_j$ and set all references to cells outside the domain
        to zero.

        '''
        atom[:, 0, :, :, 1, :, 0, :] = 0
        atom[:, 0, :, :, :, :, 1, :] -= atom[:, 0, :, :, :, :, 0, :]
        atom[:, 0, :, :, :, :, 0, :] = 0

    def no_slip_top(self, atom):
        '''Apply a no slip boundary condition at the top boundary. At this
        boundary $w_k = 0$, $(u_k + u_{k+1}) / 2 = 0$ and similar for
        $v$. So we replace $u_{k+1}$ by $-u_k$, $v_{k+1}$ by $-v_k$,
        set $w_k$ to zero and set all references to cells outside the
        domain to zero as well.

        '''
        atom[:, :, self.nz-1, :, :, :, :, 1] -= atom[:, :, self.nz-1, :, :, :, :, 2]
        atom[:, :, self.nz-1, :, 2, :, :, 1] = 0
        atom[:, :, self.nz-1, 2, :, :, :, :] = 0
        atom[:, :, self.nz-1, :, :, :, :, 2] = 0
        atom[:, :, self.nz-1, 2, 2, 1, 1, 1] = -1
        # TODO: Do we need this?
        atom[:, :, self.nz-2, 2, 2, :, :, 2] = 0

        self.frc[:, :, self.nz-1, 1] = 0

    def no_slip_bottom(self, atom):
        '''Apply a no slip boundary condition at the bottom boundary. At this
        boundary $w_{k-1} = 0$, $(u_{k-1} + u_k) / 2 = 0$ and similar
        for $v$. So we simply replace $u_{k-1}$ by $-u_k$, $v_{k-1}$
        by $-v_k$, and set all references to cells outside the domain
        to zero.

        '''
        atom[:, :, 0, :, 2, :, :, 0] = 0
        atom[:, :, 0, :, :, :, :, 1] -= atom[:, :, 0, :, :, :, :, 0]
        atom[:, :, 0, :, :, :, :, 0] = 0

    def free_slip_east(self, atom):
        '''Apply a free slip boundary condition at the east boundary. At this
        boundary $u_i = 0$, $(v_{i+1} - v_i) / h = 0$ and similar for
        $w$. So we replace $v_{i+1}$ by $v_i$, $w_{i+1}$ by $w_i$, set
        $u_i$ to zero and set all references to cells outside the
        domain to zero as well.

        '''
        atom[self.nx-1, :, :, :, :, 1, :, :] += atom[self.nx-1, :, :, :, :, 2, :, :]
        atom[self.nx-1, :, :, :, 0, 1, :, :] = 0
        atom[self.nx-1, :, :, 0, :, :, :, :] = 0
        atom[self.nx-1, :, :, :, :, 2, :, :] = 0
        atom[self.nx-1, :, :, 0, 0, 1, 1, 1] = -1
        # TODO: Do we need this?
        atom[self.nx-2, :, :, 0, 0, 2, :, :] = 0

        self.frc[self.nx-1, :, :, 0] = 0

    def free_slip_west(self, atom):
        '''Apply a free slip boundary condition at the west boundary. At this
        boundary $u_{i-1} = 0$, $(v_i - v_{i-1}) / h = 0$ and similar
        for $w$. So we simply replace $v_{i-1}$ by $v_i$, $w_{i-1}$ by
        $w_i$ and set all references to cells outside the domain to
        zero as well.

        '''
        atom[0, :, :, :, 0, 0, :, :] = 0
        atom[0, :, :, :, :, 1, :, :] += atom[0, :, :, :, :, 0, :, :]
        atom[0, :, :, :, :, 0, :, :] = 0

    def free_slip_north(self, atom):
        '''Apply a free slip boundary condition at the north boundary. At this
        boundary $v_j = 0$, $(u_{j+1} - u_j) / h = 0$ and similar for
        $w$. So we replace $u_{j+1}$ by $u_j$, $w_{j+1}$ by $w_j$, set
        $v_j$ to zero and set all references to cells outside the
        domain to zero as well.

        '''
        atom[:, self.ny-1, :, :, :, :, 1, :] += atom[:, self.ny-1, :, :, :, :, 2, :]
        atom[:, self.ny-1, :, :, 1, :, 1, :] = 0
        atom[:, self.ny-1, :, 1, :, :, :, :] = 0
        atom[:, self.ny-1, :, :, :, :, 2, :] = 0
        atom[:, self.ny-1, :, 1, 1, 1, 1, 1] = -1
        # TODO: Do we need this?
        atom[:, self.ny-2, :, 1, 1, :, 2, :] = 0

        self.frc[:, self.ny-1, :, 1] = 0

    def free_slip_south(self, atom):
        '''Apply a free slip boundary condition at the south boundary. At this
        boundary $v_{j-1} = 0$, $(u_j - u_{j-1}) / h = 0$ and similar
        for $w$. So we simply replace $u_{j-1}$ by $u_j$, $w_{j-1}$ by
        $w_j$ and set all references to cells outside the domain to
        zero as well.

        '''
        atom[:, 0, :, :, 1, :, 0, :] = 0
        atom[:, 0, :, :, :, :, 1, :] += atom[:, 0, :, :, :, :, 0, :]
        atom[:, 0, :, :, :, :, 0, :] = 0

    def free_slip_top(self, atom):
        '''Apply a free slip boundary condition at the top boundary. At this
        boundary $w_k = 0$, $(u_{k+1} - u_k) / h = 0$ and similar for
        $v$. So we replace $u_{k+1}$ by $u_k$, $v_{k+1}$ by $v_k$, set
        $w_k$ to zero and set all references to cells outside the
        domain to zero as well.

        '''
        atom[:, :, self.nz-1, :, :, :, :, 1] += atom[:, :, self.nz-1, :, :, :, :, 2]
        atom[:, :, self.nz-1, :, 2, :, :, 1] = 0
        atom[:, :, self.nz-1, 2, :, :, :, :] = 0
        atom[:, :, self.nz-1, :, :, :, :, 2] = 0
        atom[:, :, self.nz-1, 2, 2, 1, 1, 1] = -1
        # TODO: Do we need this?
        atom[:, :, self.nz-2, 2, 2, :, :, 2] = 0

        self.frc[:, :, self.nz-1, 1] = 0

    def free_slip_bottom(self, atom):
        '''Apply a free slip boundary condition at the south boundary. At this
        boundary $w_{k-1} = 0$, $(u_k - u_{k-1}) / h = 0$ and similar
        for $v$. So we simply replace $u_{k-1}$ by $u_k$, $v_{k-1}$ by
        $v_k$ and set all references to cells outside the domain to
        zero as well.

        '''
        atom[:, :, 0, :, 2, :, :, 0] = 0
        atom[:, :, 0, :, :, :, :, 1] += atom[:, :, 0, :, :, :, :, 0]
        atom[:, :, 0, :, :, :, :, 0] = 0

    def moving_lid_east(self, atom, velocity):
        '''Apply a moving lid at the east boundary with velocity V. At this
        boundary $u_i = 0$, $(v_i + v_{i+1}) / 2 = V$ and no slip for
        $w$. So we replace $v_{i+1}$ by $2V - v_i$ and then apply the no
        slip condition.

        '''
        self.frc += self._constant_forcing_east(atom, 1, 2 * velocity, -1)
        self.no_slip_east(atom)

    def moving_lid_west(self, atom, velocity):
        '''Apply a moving lid at the west boundary with velocity V. At this
        boundary $u_i = 0$, $(v_i + v_{i-1}) / 2 = V$ and no slip for
        $w$. So we replace $v_{i-1}$ by $2V - v_i$ and then apply the no
        slip condition.

        '''
        self.frc += self._constant_forcing_west(atom, 1, 2 * velocity, -1)
        self.no_slip_west(atom)

    def moving_lid_north(self, atom, velocity):
        '''Apply a moving lid at the north boundary with velocity U. At this
        boundary $v_j = 0$, $(u_j + u_{j+1}) / 2 = U$ and no slip for
        $w$. So we replace $u_{j+1}$ by $2U - u_j$ and then apply the no
        slip condition.

        '''
        self.frc += self._constant_forcing_north(atom, 0, 2 * velocity, -1)
        self.no_slip_north(atom)

    def moving_lid_south(self, atom, velocity):
        '''Apply a moving lid at the south boundary with velocity U. At this
        boundary $v_j = 0$, $(u_j + u_{j-1}) / 2 = U$ and no slip for
        $w$. So we replace $u_{j-1}$ by $2U - u_j$ and then apply the no
        slip condition.

        '''
        self.frc += self._constant_forcing_south(atom, 0, 2 * velocity, -1)
        self.no_slip_south(atom)

    def moving_lid_top(self, atom, velocity):
        '''Apply a moving lid at the top boundary with velocity U. At this
        boundary $w_k = 0$, $(u_k + u_{k+1}) / 2 = U$ and no slip for
        $v$. So we replace $u_{k+1}$ by $2U - u_k$ and then apply the no
        slip condition.

        '''
        self.frc += self._constant_forcing_top(atom, 0, 2 * velocity, -1)
        self.no_slip_top(atom)

    def moving_lid_bottom(self, atom, velocity):
        '''Apply a moving lid at the bottom boundary with velocity U. At this
        boundary $w_k = 0$, $(u_k + u_{k-1}) / 2 = U$ and no slip for
        $v$. So we replace $u_{k-1}$ by $2U - u_k$ and then apply the no
        slip condition.

        '''
        self.frc += self._constant_forcing_bottom(atom, 0, 2 * velocity, -1)
        self.no_slip_bottom(atom)

# TODO: These methods are untested with nonzero heat flux and stretched grids

    def temperature_east(self, atom, temperature):
        '''Apply a fixed temperature $T_b$ at the east boundary. We have $(T_i
        + T_{i+1}) / 2 = T_b$ so we replace $T_{i+1}$ by $2T_b - T_i$.

        '''
        self.frc += self._constant_forcing_east(atom, self.dim+1, 2 * temperature, -1)

    def temperature_west(self, atom, temperature):
        '''Apply a fixed temperature $T_b$ at the west boundary. We have $(T_i
        + T_{i-1}) / 2 = T_b$ so we replace $T_{i-1}$ by $2T_b - T_i$.

        '''
        self.frc += self._constant_forcing_west(atom, self.dim+1, 2 * temperature, -1)

    def temperature_north(self, atom, temperature):
        '''Apply a fixed temperature $T_b$ at the north boundary. We have $(T_j
        + T_{j+1}) / 2 = T_b$ so we replace $T_{j+1}$ by $2T_b - T_j$.

        '''
        self.frc += self._constant_forcing_north(atom, self.dim+1, 2 * temperature, -1)

    def temperature_south(self, atom, temperature):
        '''Apply a fixed temperature $T_b$ at the south boundary. We have $(T_j
        + T_{j-1}) / 2 = T_b$ so we replace $T_{j-1}$ by $2T_b - T_j$.

        '''
        self.frc += self._constant_forcing_south(atom, self.dim+1, 2 * temperature, -1)

    def temperature_top(self, atom, temperature):
        '''Apply a fixed temperature $T_b$ at the top boundary. We have $(T_k
        + T_{k+1}) / 2 = T_b$ so we replace $T_{k+1}$ by $2T_b - T_k$.

        '''
        self.frc += self._constant_forcing_top(atom, self.dim+1, 2 * temperature, -1)

    def temperature_bottom(self, atom, temperature):
        '''Apply a fixed temperature $T_b$ at the bottom boundary. We have $(T_k
        + T_{k-1}) / 2 = T_b$ so we replace $T_{k-1}$ by $2T_b - T_k$.

        '''
        self.frc += self._constant_forcing_bottom(atom, self.dim+1, 2 * temperature, -1)

    def heat_flux_east(self, atom, heat_flux, biot=0.0):
        r'''Apply a heat flux $Q$ at the east boundary. We have $(T_{i+1} -
        T_i) / h - \\mathrm{Bi} (T_{i+1} + T_i) / 2 = Q$ so we get
        $T_{i+1} = T_i (1 - h \\mathrm{Bi} / 2) / (1 + h \\mathrm{Bi} /
        2) + h Q / (1 + h \\mathrm{Bi} / 2)$. By default, the Biot
        number is assumed to be zero.

        '''
        h = (self.x[self.nx] - self.x[self.nx-2]) / 2

        forcing_constant = h * heat_flux / (1 + h * biot / 2)
        atom_constant = (1 - h * biot / 2) / (1 + h * biot / 2)
        self.frc += self._constant_forcing_east(atom, self.dim+1, forcing_constant, atom_constant)

    def heat_flux_west(self, atom, heat_flux, biot=0.0):
        r'''Apply a heat flux $Q$ at the west boundary. We have $(T_i -
        T_{i-1}) / h - \\mathrm{Bi} (T_{i-1} + T_i) / 2 = Q$ so we get
        $T_{i-1} = T_i (1 + h \\mathrm{Bi} / 2) / (1 - h \\mathrm{Bi}
        / 2) + h Q / (1 - h \\mathrm{Bi} / 2)$. By default, the Biot
        number is assumed to be zero.

        '''
        h = (self.x[0] - self.x[-2]) / 2

        forcing_constant = -h * heat_flux / (1 - h * biot / 2)
        atom_constant = (1 + h * biot / 2) / (1 - h * biot / 2)
        self.frc += self._constant_forcing_west(atom, self.dim+1, forcing_constant, atom_constant)

    def heat_flux_north(self, atom, heat_flux, biot=0.0):
        r'''Apply a heat flux $Q$ at the north boundary. We have $(T_{j+1} -
        T_j) / h - \\mathrm{Bi} (T_{j+1} + T_j) / 2 = Q$ so we get
        $T_{j+1} = T_j (1 - h \\mathrm{Bi} / 2) / (1 + h \\mathrm{Bi}
        / 2) + h Q / (1 + h \\mathrm{Bi} / 2)$. By default, the Biot
        number is assumed to be zero.

        '''
        h = (self.y[self.ny] - self.y[self.ny-2]) / 2

        forcing_constant = h * heat_flux / (1 + h * biot / 2)
        atom_constant = (1 - h * biot / 2) / (1 + h * biot / 2)
        self.frc += self._constant_forcing_north(atom, self.dim+1, forcing_constant, atom_constant)

    def heat_flux_south(self, atom, heat_flux, biot=0.0):
        r'''Apply a heat flux $Q$ at the south boundary. We have $(T_j -
        T_{j-1}) / h - \\mathrm{Bi} (T_{j-1} + T_i) / 2 = Q$ so we get
        $T_{j-1} = T_j (1 + h \\mathrm{Bi} / 2) / (1 - h \\mathrm{Bi}
        / 2) + h Q / (1 - h \\mathrm{Bi} / 2)$. By default, the Biot
        number is assumed to be zero.

        '''
        h = (self.y[0] - self.y[-2]) / 2

        forcing_constant = -h * heat_flux / (1 - h * biot / 2)
        atom_constant = (1 + h * biot / 2) / (1 - h * biot / 2)
        self.frc += self._constant_forcing_south(atom, self.dim+1, forcing_constant, atom_constant)

    def heat_flux_top(self, atom, heat_flux, biot=0.0):
        r'''Apply a heat flux $Q$ at the top boundary. We have $(T_{k+1} -
        T_k) / h - \\mathrm{Bi} (T_{k+1} + T_k) / 2 = Q$ so we get
        $T_{k+1} = T_k (1 - h \\mathrm{Bi} / 2) / (1 + h \\mathrm{Bi}
        / 2) + h Q / (1 + h \\mathrm{Bi} / 2)$. By default, the Biot
        number is assumed to be zero.

        '''
        h = (self.z[self.nz] - self.z[self.nz-2]) / 2

        forcing_constant = h * heat_flux / (1 + h * biot / 2)
        atom_constant = (1 - h * biot / 2) / (1 + h * biot / 2)
        self.frc += self._constant_forcing_top(atom, self.dim+1, forcing_constant, atom_constant)

    def heat_flux_bottom(self, atom, heat_flux, biot=0.0):
        r'''Apply a heat flux $Q$ at the bottom boundary. We have $(T_k -
        T_{k-1}) / h - \\mathrm{Bi} (T_{k-1} + T_i) / 2 = Q$ so we get
        $T_{k-1} = T_k (1 + h \\mathrm{Bi} / 2) / (1 - h \\mathrm{Bi}
        / 2) + h Q / (1 - h \\mathrm{Bi} / 2)$. By default, the Biot
        number is assumed to be zero.

        '''
        h = (self.z[0] - self.z[-2]) / 2

        forcing_constant = -h * heat_flux / (1 - h * biot / 2)
        atom_constant = (1 + h * biot / 2) / (1 - h * biot / 2)
        self.frc += self._constant_forcing_bottom(atom, self.dim+1, forcing_constant, atom_constant)

    def salinity_flux_east(self, atom, salinity_flux):
        '''Apply a salinity flux $Q$ at the east boundary. We have $(S_{i+1} -
        S_i) / h = Q$ so we replace $S_{i+1}$ by $S_i + hQ$.

        '''
        h = (self.x[self.nx] - self.x[self.nx-2]) / 2
        self.frc += self._constant_forcing_east(atom, self.dim+2, h * salinity_flux, 1)

    def salinity_flux_west(self, atom, salinity_flux):
        '''Apply a salinity flux $Q$ at the west boundary. We have $(S_i -
        S_{i-1}) / h = Q$ so we replace $S_{i-1}$ by $S_i - hQ$.

        '''
        h = (self.x[0] - self.x[-2]) / 2
        self.frc += self._constant_forcing_west(atom, self.dim+2, -h * salinity_flux, 1)

    def salinity_flux_north(self, atom, salinity_flux):
        '''Apply a salinity flux $Q$ at the north boundary. We have $(S_{j+1} -
        S_j) / h = Q$ so we replace $S_{j+1}$ by $S_j + hQ$.

        '''
        h = (self.y[self.ny] - self.y[self.ny-2]) / 2
        self.frc += self._constant_forcing_north(atom, self.dim+2, h * salinity_flux, 1)

    def salinity_flux_south(self, atom, salinity_flux):
        '''Apply a salinity flux $Q$ at the south boundary. We have $(S_j -
        S_{j-1}) / h = Q$ so we replace $S_{j-1}$ by $S_j - hQ$.

        '''
        h = (self.y[0] - self.y[-2]) / 2
        self.frc += self._constant_forcing_south(atom, self.dim+2, -h * salinity_flux, 1)

    def salinity_flux_top(self, atom, salinity_flux):
        '''Apply a salinity flux $Q$ at the top boundary. We have $(S_{k+1} -
        S_k) / h = Q$ so we replace $S_{k+1}$ by $S_k + hQ$.

        '''
        h = (self.z[self.nz] - self.z[self.nz-2]) / 2
        self.frc += self._constant_forcing_top(atom, self.dim+2, h * salinity_flux, 1)

    def salinity_flux_bottom(self, atom, salinity_flux):
        '''Apply a salinity flux $Q$ at the bottom boundary. We have $(S_k -
        S_{k-1}) / h = Q$ so we replace $S_{k-1}$ by $S_k - hQ$.

        '''
        h = (self.z[0] - self.z[-2]) / 2
        self.frc += self._constant_forcing_bottom(atom, self.dim+2, -h * salinity_flux, 1)

    def _constant_forcing(self, atom, nx, ny, var, value):
        value = numpy.ones((nx + 2, ny + 2)) * value

        frc = numpy.zeros((nx, ny, self.dof))
        for j, i, y, x in numpy.ndindex(ny, nx, 3, 3):
            frc[i, j, var] += atom[i, j, var, x, y] * value[i + x - 1, j + y - 1]

        return frc

    def _constant_forcing_east(self, atom, var, forcing_constant, atom_constant):
        frc = numpy.zeros((self.nx, self.ny, self.nz, self.dof))
        frc[self.nx-1, :, :, :] = self._constant_forcing(
            atom[self.nx-1, :, :, :, var, 2, :, :], self.ny, self.nz,
            var, forcing_constant)

        atom[self.nx-1, :, :, :, var, 1, :, :] += atom_constant * atom[self.nx-1, :, :, :, var, 2, :, :]
        atom[self.nx-1, :, :, :, var, 2, :, :] = 0

        return frc

    def _constant_forcing_west(self, atom, var, forcing_constant, atom_constant):
        frc = numpy.zeros((self.nx, self.ny, self.nz, self.dof))
        frc[0, :, :, :] = self._constant_forcing(
            atom[0, :, :, :, var, 0, :, :], self.ny, self.nz,
            var, forcing_constant)

        atom[0, :, :, :, var, 1, :, :] += atom_constant * atom[0, :, :, :, var, 0, :, :]
        atom[0, :, :, :, var, 0, :, :] = 0

        return frc

    def _constant_forcing_north(self, atom, var, forcing_constant, atom_constant):
        frc = numpy.zeros((self.nx, self.ny, self.nz, self.dof))
        frc[:, self.ny-1, :, :] = self._constant_forcing(
            atom[:, self.ny-1, :, :, var, :, 2, :], self.nx, self.nz,
            var, forcing_constant)

        atom[:, self.ny-1, :, :, var, :, 1, :] += atom_constant * atom[:, self.ny-1, :, :, var, :, 2, :]
        atom[:, self.ny-1, :, :, var, :, 2, :] = 0

        return frc

    def _constant_forcing_south(self, atom, var, forcing_constant, atom_constant):
        frc = numpy.zeros((self.nx, self.ny, self.nz, self.dof))
        frc[:, 0, :, :] = self._constant_forcing(
            atom[:, 0, :, :, var, :, 0, :], self.nx, self.nz,
            var, forcing_constant)

        atom[:, 0, :, :, var, :, 1, :] += atom_constant * atom[:, 0, :, :, var, :, 0, :]
        atom[:, 0, :, :, var, :, 0, :] = 0

        return frc

    def _constant_forcing_top(self, atom, var, forcing_constant, atom_constant):
        frc = numpy.zeros((self.nx, self.ny, self.nz, self.dof))
        frc[:, :, self.nz-1, :] = self._constant_forcing(
            atom[:, :, self.nz-1, :, var, :, :, 2], self.nx, self.ny,
            var, forcing_constant)

        atom[:, :, self.nz-1, :, var, :, :, 1] += atom_constant * atom[:, :, self.nz-1, :, var, :, :, 2]
        atom[:, :, self.nz-1, :, var, :, :, 2] = 0

        return frc

    def _constant_forcing_bottom(self, atom, var, forcing_constant, atom_constant):
        frc = numpy.zeros((self.nx, self.ny, self.nz, self.dof))
        frc[:, :, 0, :] = self._constant_forcing(
            atom[:, :, 0, :, var, :, :, 0], self.nx, self.ny,
            var, forcing_constant)

        atom[:, :, 0, :, var, :, :, 1] += atom_constant * atom[:, :, 0, :, var, :, :, 0]
        atom[:, :, 0, :, var, :, :, 0] = 0

        return frc
