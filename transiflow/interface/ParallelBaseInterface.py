import numpy

from transiflow.interface import BaseInterface


def ind2sub(nx, ny, nz, idx, dof=1):
    rem = idx
    var = rem % dof
    rem = rem // dof
    i = rem % nx
    rem = rem // nx
    j = rem % ny
    rem = rem // ny
    k = rem % nz
    return (i, j, k, var)


def sub2ind(nx, ny, nz, dof, i, j, k, var):
    return ((k * ny + j) * nx + i) * dof + var


def get_local_coordinate_vector(x, nx_offset, nx_local):
    x = numpy.roll(x, 2)
    x = x[nx_offset:nx_offset + nx_local + 3]
    return numpy.roll(x, -2)


class ParallelBaseInterface(BaseInterface):
    '''This class defines an interface to the Epetra backend for the
    discretization. We use this so we can write higher level methods
    such as pseudo-arclength continuation without knowing anything
    about the underlying methods such as the solvers that are present
    in the backend we are interfacing with.

    The Epetra backend partitions the domain into Cartesian subdomains,
    while solving linear systems on skew Cartesian subdomains to deal
    with the C-grid discretization. The subdomains will be distributed
    over multiple processors if MPI is used to run the application.'''
    def __init__(self, comm, parameters, nx, ny, nz, dim=None, dof=None):
        super().__init__(parameters, nx, ny, nz, dim, dof)

        self.nx_global = nx
        self.ny_global = ny
        self.nz_global = nz

        self.comm = comm

        self.partition_domain()

        self.discretization.x = get_local_coordinate_vector(self.discretization.x,
                                                            self.nx_offset, self.nx_local)
        self.discretization.y = get_local_coordinate_vector(self.discretization.y,
                                                            self.ny_offset, self.ny_local)
        self.discretization.z = get_local_coordinate_vector(self.discretization.z,
                                                            self.nz_offset, self.nz_local)

        self.discretization.nx = self.nx_local
        self.discretization.ny = self.ny_local
        self.discretization.nz = self.nz_local

        self.nx = self.nx_local
        self.ny = self.ny_local
        self.nz = self.nz_local

    def get_comm_size(self):
        try:
            return self.comm.size
        except AttributeError:
            return self.comm.NumProc()

    def get_comm_rank(self):
        try:
            return self.comm.rank
        except AttributeError:
            return self.comm.MyPID()

    def partition_domain(self):
        '''Partition the domain into Cartesian subdomains for computing the
        discretization.'''

        rmin = 1e100

        self.npx = 1
        self.npy = 1
        self.npz = 1

        nparts = self.get_comm_size()
        pid = self.get_comm_rank()

        found = False

        # check all possibilities of splitting the map
        for t1 in range(1, nparts + 1):
            for t2 in range(1, nparts // t1 + 1):
                t3 = nparts // (t1 * t2)
                if t1 * t2 * t3 == nparts:
                    nx_loc = self.nx_global // t1
                    ny_loc = self.ny_global // t2
                    nz_loc = self.nz_global // t3

                    if (nx_loc * t1 != self.nx_global or ny_loc * t2 != self.ny_global
                            or nz_loc * t3 != self.nz_global):
                        continue

                    r1 = abs(self.nx_global / t1 - self.ny_global / t2)
                    r2 = abs(self.nx_global / t1 - self.nz_global / t3)
                    r3 = abs(self.ny_global / t2 - self.nz_global / t3)
                    r = r1 + r2 + r3

                    if r < rmin:
                        rmin = r
                        self.npx = t1
                        self.npy = t2
                        self.npz = t3
                        found = True

        if not found:
            raise Exception("Could not split %dx%dx%d domain in %d parts." %
                            (self.nx_global, self.ny_global, self.nz_global, nparts))

        self.pidx, self.pidy, self.pidz, _ = ind2sub(self.npx, self.npy, self.npz, pid)

        # Compute the local domain size and offset.
        self.nx_local = self.nx_global // self.npx
        self.ny_local = self.ny_global // self.npy
        self.nz_local = self.nz_global // self.npz

        self.nx_offset = self.nx_local * self.pidx
        self.ny_offset = self.ny_local * self.pidy
        self.nz_offset = self.nz_local * self.pidz

        # Add ghost nodes to factor out boundary conditions in the interior.
        if self.pidx > 0:
            self.nx_local += 2
            self.nx_offset -= 2
        if self.pidx < self.npx - 1:
            self.nx_local += 2
        if self.pidy > 0:
            self.ny_local += 2
            self.ny_offset -= 2
        if self.pidy < self.npy - 1:
            self.ny_local += 2
        if self.pidz > 0:
            self.nz_local += 2
            self.nz_offset -= 2
        if self.pidz < self.npz - 1:
            self.nz_local += 2

    def is_ghost(self, i, j=None, k=None):
        '''If a node is a ghost node that is used only for computing the
        discretization and is located outside of an interior boundary.'''

        if j is None:
            i, j, k, _ = ind2sub(self.nx_local, self.ny_local, self.nz_local, i, self.dof)

        ghost = False
        if self.pidx > 0 and i < 2:
            ghost = True
        elif self.pidx < self.npx - 1 and i >= self.nx_local - 2:
            ghost = True
        elif self.pidy > 0 and j < 2:
            ghost = True
        elif self.pidy < self.npy - 1 and j >= self.ny_local - 2:
            ghost = True
        elif self.pidz > 0 and k < 2:
            ghost = True
        elif self.pidz < self.npz - 1 and k >= self.nz_local - 2:
            ghost = True

        return ghost

    def create_map(self, overlapping=False):
        '''Create a map on which the local discretization domain is defined.
        The overlapping part is only used for computing the discretization.'''

        local_elements = [0] * self.nx_local * self.ny_local * self.nz_local * self.dof

        pos = 0
        for k in range(self.nz_local):
            for j in range(self.ny_local):
                for i in range(self.nx_local):
                    if not overlapping and self.is_ghost(i, j, k):
                        continue
                    for var in range(self.dof):
                        local_elements[pos] = sub2ind(
                            self.nx_global,
                            self.ny_global,
                            self.nz_global,
                            self.dof,
                            i + self.nx_offset,
                            j + self.ny_offset,
                            k + self.nz_offset,
                            var,
                        )
                        pos += 1

        return local_elements[0:pos]

    def rhs(self, state):
        '''Right-hand side in M * du / dt = F(u).'''

        raise NotImplementedError()

    def jacobian(self, state):
        '''Jacobian J of F in M * du / dt = F(u).'''

        raise NotImplementedError()

    def mass_matrix(self):
        '''Mass matrix M in M * du / dt = F(u).'''

        raise NotImplementedError()

    def solve(self, jac, rhs, rhs2=None, V=None, W=None, C=None, solver=None):
        '''Solve J y = x for y with the possibility of solving a bordered system.'''

        raise NotImplementedError()

    def eigs(self, state, return_eigenvectors=False, enable_recycling=False):
        '''Compute the generalized eigenvalues of beta * J(x) * v = alpha * M * v.'''

        raise NotImplementedError()
