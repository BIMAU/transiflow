from PyTrilinos import Epetra
from PyTrilinos import Amesos

import numpy

import fvm


class Vector(Epetra.Vector):
    '''Distributed Epetra_Vector with some extra methods added to it for convenience.'''

    def __neg__(self):
        v = Vector(self)
        v.Scale(-1.0)
        return v

    def __truediv__(self, scal):
        v = Vector(self)
        v.Scale(1.0 / scal)
        return v

    def dot(self, other):
        return self.Dot(other)[0]

    def gather(self):
        local_elements = []
        if self.Comm().MyPID() == 0:
            local_elements = range(self.Map().NumGlobalElements())

        local_map = Epetra.Map(-1, local_elements, 0, self.Comm())
        importer = Epetra.Import(local_map, self.Map())
        out = Epetra.Vector(local_map)
        out.Import(self, importer, Epetra.Insert)
        return out

    @staticmethod
    def from_array(m, x):
        local_elements = []
        if m.Comm().MyPID() == 0:
            local_elements = range(m.NumGlobalElements())

        local_map = Epetra.Map(-1, local_elements, 0, m.Comm())
        importer = Epetra.Import(m, local_map)
        x_local = Vector(Epetra.Copy, local_map, x)
        out = Vector(m)
        out.Import(x_local, importer, Epetra.Insert)
        return out

    size = property(Epetra.Vector.GlobalLength)


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
    x = x[nx_offset:nx_offset+nx_local+3]
    return numpy.roll(x, -2)


class EpetraInterface(fvm.Interface):
    '''This class defines an interface to the Epetra backend for the
    discretization. We use this so we can write higher level methods
    such as pseudo-arclength continuation without knowing anything
    about the underlying methods such as the solvers that are present
    in the backend we are interfacing with.

    The Epetra backend partitions the domain into Cartesian subdomains,
    while solving linear systems on skew Cartesian subdomains to deal
    with the C-grid discretization. The subdomains will be distributed
    over multiple processors if MPI is used to run the application.'''

    def __init__(self, comm, parameters, nx, ny, nz, dim, dof):
        fvm.Interface.__init__(self, parameters, nx, ny, nz, dim, dof)

        self.nx_global = nx
        self.ny_global = ny
        self.nz_global = nz
        self.dof = dof

        self.comm = comm

        self.parameters = parameters

        self.partition_domain()
        self.map = self.create_map()

        self.assembly_map = self.create_map(True)
        self.assembly_importer = Epetra.Import(self.assembly_map, self.map)

        self.solve_map = self.map
        self.solve_importer = Epetra.Import(self.solve_map, self.map)

        self.discretization.x = get_local_coordinate_vector(self.discretization.x, self.nx_offset, self.nx_local)
        self.discretization.y = get_local_coordinate_vector(self.discretization.y, self.ny_offset, self.ny_local)
        self.discretization.z = get_local_coordinate_vector(self.discretization.z, self.nz_offset, self.nz_local)

        self.discretization.nx = self.nx_local
        self.discretization.ny = self.ny_local
        self.discretization.nz = self.nz_local

        self.nx = self.nx_local
        self.ny = self.ny_local
        self.nz = self.nz_local

        self.left_scaling = None
        self.inv_left_scaling = None
        self.right_scaling = None
        self.inv_right_scaling = None

        self.jac = None
        self.mass = None

    def partition_domain(self):
        '''Partition the domain into Cartesian subdomains for computing the
        discretization.'''

        rmin = 1e100

        self.npx = 1
        self.npy = 1
        self.npz = 1

        nparts = self.comm.NumProc()
        pid = self.comm.MyPID()

        found = False

        # check all possibilities of splitting the map
        for t1 in range(1, nparts + 1):
            for t2 in range(1, nparts // t1 + 1):
                t3 = nparts // (t1 * t2)
                if t1 * t2 * t3 == nparts:
                    nx_loc = self.nx_global // t1
                    ny_loc = self.ny_global // t2
                    nz_loc = self.nz_global // t3

                    if nx_loc * t1 != self.nx_global or ny_loc * t2 != self.ny_global or nz_loc * t3 != self.nz_global:
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
            raise Exception('Could not split %dx%dx%d domain in %d parts.' % (self.nx_global, self.ny_global,
                                                                              self.nz_global, nparts))

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
                        local_elements[pos] = sub2ind(self.nx_global, self.ny_global, self.nz_global, self.dof,
                                                      i + self.nx_offset, j + self.ny_offset, k + self.nz_offset, var)
                        pos += 1

        return Epetra.Map(-1, local_elements[0:pos], 0, self.comm)

    def rhs(self, state):
        '''Right-hand side in M * du / dt = F(u) defined on the
        non-overlapping discretization domain map.'''

        state_ass = Vector(self.assembly_map)
        state_ass.Import(state, self.assembly_importer, Epetra.Insert)

        rhs = fvm.Interface.rhs(self, state_ass)
        rhs_ass = Vector(Epetra.Copy, self.assembly_map, rhs)
        rhs = Vector(self.map)
        rhs.Export(rhs_ass, self.assembly_importer, Epetra.Zero)
        return rhs

    def jacobian(self, state):
        '''Jacobian J of F in M * du / dt = F(u) defined on the
        domain map used by Epetra.'''

        state_ass = Vector(self.assembly_map)
        state_ass.Import(state, self.assembly_importer, Epetra.Insert)

        local_jac = fvm.Interface.jacobian(self, state_ass)

        if self.jac is None:
            self.jac = Epetra.FECrsMatrix(Epetra.Copy, self.solve_map, 27)
        else:
            self.jac.PutScalar(0.0)

        for i in range(len(local_jac.begA)-1):
            if self.is_ghost(i):
                continue
            row = self.assembly_map.GID64(i)
            for j in range(local_jac.begA[i], local_jac.begA[i+1]):
                # __setitem__ automatically calls ReplaceGlobalValues if the matrix is filled,
                # InsertGlobalValues otherwise
                self.jac[row, self.assembly_map.GID64(local_jac.jcoA[j])] = local_jac.coA[j]
        self.jac.GlobalAssemble(True, Epetra.Insert)

        return self.jac

    def mass_matrix(self):
        '''Mass matrix M in M * du / dt = F(u) defined on the
        domain map used by Epetra.'''

        local_mass = fvm.Interface.mass_matrix(self)

        if self.mass is None:
            self.mass = Epetra.FECrsMatrix(Epetra.Copy, self.solve_map, 1)
        else:
            self.mass.PutScalar(0.0)

        for i in range(len(local_mass.begA)-1):
            if self.is_ghost(i):
                continue
            row = self.assembly_map.GID64(i)
            for j in range(local_mass.begA[i], local_mass.begA[i+1]):
                # __setitem__ automatically calls ReplaceGlobalValues if the matrix is filled,
                # InsertGlobalValues otherwise
                self.mass[row, self.assembly_map.GID64(local_mass.jcoA[j])] = local_mass.coA[j]
        self.mass.GlobalAssemble(True, Epetra.Insert)

        return self.mass

    def compute_scaling(self):
        '''Compute scaling for the linear problem'''
        self.left_scaling = Vector(self.solve_map)
        self.left_scaling.PutScalar(1.0)

        self.inv_left_scaling = Vector(self.solve_map)
        self.inv_left_scaling.PutScalar(1.0)

        self.right_scaling = Vector(self.solve_map)
        self.right_scaling.PutScalar(1.0)

        self.inv_right_scaling = Vector(self.solve_map)
        self.inv_right_scaling.PutScalar(1.0)

        dim = self.discretization.dim
        dof = self.discretization.dof

        for lrid in range(self.jac.NumMyRows()):
            grid = self.jac.GRID(lrid)
            var = grid % dof
            values, indices = self.jac.ExtractMyRowCopy(lrid)
            for j in range(len(indices)):
                lcid = indices[j]
                value = values[j]
                gcid = self.jac.GCID(lcid)
                if value < 1e-8:
                    continue

                if var != dim and gcid % dof == dim:
                    # If the row is a velocity and the column a pressure
                    self.left_scaling[lrid] = 1 / value
                    self.inv_left_scaling[lrid] = value
                    break
                elif var == dim and gcid % dof != dim and self.jac.MyGRID(gcid):
                    # If the row is a pressure and the column a velocity
                    lid = self.jac.LRID(gcid)
                    self.right_scaling[lid] = 1 / value
                    self.inv_right_scaling[lid] = value

    def scale_matrix(self, mat):
        assert not hasattr(mat, 'scaled') or not mat.scaled
        mat.scaled = True

        mat.LeftScale(self.left_scaling)
        mat.RightScale(self.right_scaling)

    def scale_jacobian(self):
        self.scale_matrix(self.jac)

    def scale_rhs(self, rhs):
        rhs.Multiply(1.0, self.left_scaling, rhs, 0.0)

    def scale_lhs(self, lhs):
        lhs.Multiply(1.0, self.inv_right_scaling, lhs, 0.0)

    def unscale_matrix(self, mat):
        assert mat.scaled
        mat.scaled = False

        mat.LeftScale(self.inv_left_scaling)
        mat.RightScale(self.inv_right_scaling)

    def unscale_jacobian(self):
        self.unscale_matrix(self.jac)

    def unscale_rhs(self, rhs):
        rhs.Multiply(1.0, self.inv_left_scaling, rhs, 0.0)

    def unscale_lhs(self, lhs):
        lhs.Multiply(1.0, self.right_scaling, lhs, 0.0)

    def solve(self, jac, rhs, rhs2=None, V=None, W=None, C=None, solver=None):
        '''Solve J y = x for y with the possibility of solving a bordered system.'''
        if rhs2:
            raise NotImplementedError()

        x = Vector(rhs)

        problem = Epetra.LinearProblem(jac, x, rhs)
        factory = Amesos.Factory()
        solver = factory.Create('Klu', problem)
        solver.SymbolicFactorization()
        solver.NumericFactorization()
        solver.Solve()

        return x

    def eigs(self, state, return_eigenvectors=False, enable_recycling=False):
        '''Compute the generalized eigenvalues of beta * J(x) * v = alpha * M * v.'''
        raise NotImplementedError()
