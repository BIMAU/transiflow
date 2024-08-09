from PyTrilinos import Epetra
from PyTrilinos import Amesos

from transiflow.interface import ParallelBaseInterface


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
        out = Vector(local_map)
        out.Import(self, importer, Epetra.Insert)
        return out

    def all_gather(self):
        local_elements = range(self.Map().NumGlobalElements())
        local_map = Epetra.Map(-1, local_elements, 0, self.Comm())
        importer = Epetra.Import(local_map, self.Map())
        out = Vector(local_map)
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


class Interface(ParallelBaseInterface):
    '''This class defines an interface to the Epetra backend for the
    discretization. We use this so we can write higher level methods
    such as pseudo-arclength continuation without knowing anything
    about the underlying methods such as the solvers that are present
    in the backend we are interfacing with.

    The Epetra backend partitions the domain into Cartesian subdomains,
    while solving linear systems on skew Cartesian subdomains to deal
    with the C-grid discretization. The subdomains will be distributed
    over multiple processors if MPI is used to run the application.'''
    def __init__(self, parameters, nx, ny, nz=1, dim=None, dof=None, comm=None):
        if comm is None:
            comm = Epetra.PyComm()

        ParallelBaseInterface.__init__(self, comm, parameters, nx, ny, nz, dim, dof)

        self.map = self.create_map()

        self.assembly_map = self.create_map(True)
        self.assembly_importer = Epetra.Import(self.assembly_map, self.map)

        self.solve_map = self.map
        self.solve_importer = Epetra.Import(self.solve_map, self.map)

        self.left_scaling = None
        self.inv_left_scaling = None
        self.right_scaling = None
        self.inv_right_scaling = None

        self.jac = None
        self.mass = None

    def vector(self, *args):
        if len(args) == 0:
            return Vector(self.map)

        return Vector(args)

    def vector_from_array(self, array):
        return Vector.from_array(self.map, array)

    def array_from_vector(self, vector):
        return Vector.all_gather(vector)

    def create_map(self, overlapping=False):
        '''Create a map on which the local discretization domain is defined.
        The overlapping part is only used for computing the discretization.'''

        local_elements = ParallelBaseInterface.create_map(self, overlapping)
        return Epetra.Map(-1, local_elements, 0, self.comm)

    def rhs(self, state):
        '''Right-hand side in M * du / dt = F(u) defined on the
        non-overlapping discretization domain map.'''

        state_ass = Vector(self.assembly_map)
        state_ass.Import(state, self.assembly_importer, Epetra.Insert)

        rhs = self.discretization.rhs(state_ass)
        rhs_ass = Vector(Epetra.Copy, self.assembly_map, rhs)
        rhs = Vector(self.map)
        rhs.Export(rhs_ass, self.assembly_importer, Epetra.Zero)
        return rhs

    def jacobian(self, state):
        '''Jacobian J of F in M * du / dt = F(u) defined on the
        domain map used by Epetra.'''

        state_ass = Vector(self.assembly_map)
        state_ass.Import(state, self.assembly_importer, Epetra.Insert)

        local_jac = self.discretization.jacobian(state_ass)

        if self.jac is None:
            self.jac = Epetra.FECrsMatrix(Epetra.Copy, self.solve_map, 27)
        else:
            self.jac.PutScalar(0.0)

        for i in range(len(local_jac.begA) - 1):
            if self.is_ghost(i):
                continue
            row = self.assembly_map.GID64(i)
            for j in range(local_jac.begA[i], local_jac.begA[i + 1]):
                # __setitem__ automatically calls ReplaceGlobalValues if the matrix is filled,
                # InsertGlobalValues otherwise
                self.jac[row, self.assembly_map.GID64(local_jac.jcoA[j])] = local_jac.coA[j]
        self.jac.GlobalAssemble(True, Epetra.Insert)

        return self.jac

    def mass_matrix(self):
        '''Mass matrix M in M * du / dt = F(u) defined on the
        domain map used by Epetra.'''

        local_mass = self.discretization.mass_matrix()

        if self.mass is None:
            self.mass = Epetra.FECrsMatrix(Epetra.Copy, self.solve_map, 1)
        else:
            self.mass.PutScalar(0.0)

        for i in range(len(local_mass.begA) - 1):
            if self.is_ghost(i):
                continue
            row = self.assembly_map.GID64(i)
            for j in range(local_mass.begA[i], local_mass.begA[i + 1]):
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
