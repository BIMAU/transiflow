from PyTrilinos import Epetra
from PyTrilinos import Amesos
from PyTrilinos import Teuchos

import numpy
import sys
import os

import fvm

import HYMLS

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

def set_default_parameter(parameterlist, name, value):
    if name not in parameterlist:
        parameterlist[name] = value

def convert_parameters(parameters, teuchos_parameters=None):
    if teuchos_parameters is None:
        teuchos_parameters = Teuchos.ParameterList()

    if isinstance(parameters, Teuchos.ParameterList):
        return parameters

    for i, j in parameters.items():
        if isinstance(j, dict):
            sublist = teuchos_parameters.sublist(i)
            convert_parameters(j, sublist)
        else:
            try:
                teuchos_parameters.set(i, j)
            except Exception:
                pass

    return teuchos_parameters

def get_local_coordinate_vector(x, nx_offset, nx_local):
    x = numpy.roll(x, 2)
    x = x[nx_offset:nx_offset+nx_local+3]
    return numpy.roll(x, -2)

class Interface(fvm.Interface):
    '''This class defines an interface to the HYMLS backend for the
    discretization. We use this so we can write higher level methods
    such as pseudo-arclength continuation without knowing anything
    about the underlying methods such as the solvers that are present
    in the backend we are interfacing with.

    The HYMLS backend partitions the domain into Cartesian subdomains,
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

        # Disable HYMLS output from MPI ranks != 0
        HYMLS.Tools.InitializeIO(self.comm)

        # Do the same for Python output
        self._original_stdout = sys.stdout
        if self.comm.MyPID() != 0:
            print('PID %d: Disable output to stdout' % self.comm.MyPID())
            sys.stdout = open(os.devnull, 'w')

        self.parameters = parameters
        self.teuchos_parameters = self.get_teuchos_parameters()

        self.partition_domain()
        self.map = self.create_map()

        self.assembly_map = self.create_map(True)
        self.assembly_importer = Epetra.Import(self.assembly_map, self.map)

        partitioner = HYMLS.SkewCartesianPartitioner(self.teuchos_parameters, self.comm)
        partitioner.Partition()

        self.solve_map = partitioner.Map()
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
        self.initialize()

    def __del__(self):
        if self.comm.MyPID() != 0:
            sys.stdout.close()
            sys.stdout = self._original_stdout
            print('PID %d: Re-enable output to stdout' % self.comm.MyPID())

    def get_teuchos_parameters(self):
        teuchos_parameters = convert_parameters(self.parameters)

        problem_parameters = teuchos_parameters.sublist('Problem')
        set_default_parameter(problem_parameters, 'nx', self.nx_global)
        set_default_parameter(problem_parameters, 'ny', self.ny_global)
        set_default_parameter(problem_parameters, 'nz', self.nz_global)

        set_default_parameter(problem_parameters, 'Dimension', self.dim)
        set_default_parameter(problem_parameters, 'Degrees of Freedom', self.dof)
        set_default_parameter(problem_parameters, 'Equations', 'Stokes-C')

        set_default_parameter(problem_parameters, 'x-periodic', self.discretization.x_periodic)
        set_default_parameter(problem_parameters, 'y-periodic', self.discretization.y_periodic)
        set_default_parameter(problem_parameters, 'z-periodic', self.discretization.z_periodic)

        solver_parameters = teuchos_parameters.sublist('Solver')
        solver_parameters.set('Initial Vector', 'Zero')
        solver_parameters.set('Left or Right Preconditioning', 'Right')

        iterative_solver_parameters = solver_parameters.sublist('Iterative Solver')
        iterative_solver_parameters.set('Output Stream', 0)
        set_default_parameter(iterative_solver_parameters, 'Maximum Iterations', 1000)
        set_default_parameter(iterative_solver_parameters, 'Maximum Restarts', 20)
        set_default_parameter(iterative_solver_parameters, 'Num Blocks', 100)
        set_default_parameter(iterative_solver_parameters, 'Flexible Gmres', False)
        set_default_parameter(iterative_solver_parameters, 'Convergence Tolerance', 1e-8)
        set_default_parameter(iterative_solver_parameters, 'Output Frequency', 1)
        set_default_parameter(iterative_solver_parameters, 'Show Maximum Residual Norm Only', False)
        set_default_parameter(iterative_solver_parameters, 'Implicit Residual Scaling', 'Norm of RHS')
        set_default_parameter(iterative_solver_parameters, 'Explicit Residual Scaling', 'Norm of RHS')

        prec_parameters = teuchos_parameters.sublist('Preconditioner')
        prec_parameters.set('Partitioner', 'Skew Cartesian')
        set_default_parameter(prec_parameters, 'Separator Length', min(8, self.nx_global))
        set_default_parameter(prec_parameters, 'Coarsening Factor', 2)
        set_default_parameter(prec_parameters, 'Number of Levels', 1)

        coarse_solver_parameters = prec_parameters.sublist('Coarse Solver')
        set_default_parameter(coarse_solver_parameters, "amesos: solver type", "Amesos_Superludist")

        return teuchos_parameters

    def unset_parameter(self, name, original_parameters):
        '''Set a parameter in self.parameters back to its original value. '''

        if name in original_parameters:
            self.set_parameter(name, original_parameters[name])
            return

        if name in self.parameters:
            if hasattr(self.parameters, 'remove'):
                self.parameters.remove(name)
            else:
                del self.parameters[name]

    def initialize(self):
        '''Initialize the Jacobian and the preconditioner, but make sure the
        nonlinear part is also nonzero so we can replace all values
        later, rather than insert them.'''

        # Backup the original parameters and put model parameters to 1
        parameter_names = ['Reynolds Number', 'Rayleigh Number',
                           'Prandtl Number', 'Rossby Parameter']

        original_parameters = {}
        for i in parameter_names:
            if i in self.parameters:
                original_parameters[i] = self.parameters[i]

            self.set_parameter(i, numpy.random.random())

        # Generate a Jacobian with a random state
        x = Vector(self.map)
        x.Random()
        self.jacobian(x)

        self.compute_scaling()
        self.scale_jacobian()

        self.preconditioner = HYMLS.Preconditioner(self.jac, self.teuchos_parameters)
        self.preconditioner.Initialize()

        self.solver = HYMLS.BorderedSolver(self.jac, self.preconditioner, self.teuchos_parameters)

        self.unscale_jacobian()

        # Put back the original parameters
        for i in parameter_names:
            self.unset_parameter(i, original_parameters)

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
        domain map used by HYMLS.'''

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
        domain map used by HYMLS.'''

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

    def direct_solve(self, jac, rhs):
        '''Currently unused direct solver that was used for testing.'''

        A = Epetra.CrsMatrix(Epetra.Copy, self.map, 27)
        for i in range(len(jac.begA)-1):
            if i == self.dim:
                A[i, i] = -1
                continue
            for j in range(jac.begA[i], jac.begA[i+1]):
                if jac.jcoA[j] != self.dim:
                    A[i, jac.jcoA[j]] = jac.coA[j]
        A.FillComplete()

        rhs[self.dim] = 0
        x = Vector(rhs)

        problem = Epetra.LinearProblem(A, x, rhs)
        factory = Amesos.Factory()
        solver = factory.Create('Klu', problem)
        solver.SymbolicFactorization()
        solver.NumericFactorization()
        solver.Solve()

        return x

    def solve(self, jac, rhs, rhs2=None, V=None, W=None, C=None, solver=None):
        '''Solve J y = x for y with the possibility of solving a bordered system.'''

        if solver is None:
            solver = self.solver

        rhs_sol = Vector(self.solve_map)
        rhs_sol.Import(rhs, self.solve_importer, Epetra.Insert)

        x_sol = Vector(rhs_sol)

        self.compute_scaling()

        if rhs2 is not None:
            rhs2_sol = Epetra.SerialDenseMatrix(1, 1)
            rhs2_sol[0, 0] = rhs2

            x2_sol = Epetra.SerialDenseMatrix(1, 1)

            V_sol = Vector(self.solve_map)
            V_sol.Import(V, self.solve_importer, Epetra.Insert)

            self.scale_rhs(V_sol)

            W_sol = Vector(self.solve_map)
            W_sol.Import(W, self.solve_importer, Epetra.Insert)

            self.unscale_lhs(W_sol)

            C_sol = Epetra.SerialDenseMatrix(1, 1)
            C_sol[0, 0] = C

            solver.SetBorder(V_sol, W_sol, C_sol)

        self.scale_jacobian()
        self.scale_rhs(rhs_sol)

        self.preconditioner.Compute()
        if rhs2 is not None:
            solver.ApplyInverse(rhs_sol, rhs2_sol, x_sol, x2_sol)

            x2 = x2_sol[0, 0]
        else:
            solver.ApplyInverse(rhs_sol, x_sol)

        solver.UnsetBorder()

        self.unscale_jacobian()
        self.unscale_lhs(x_sol)

        x = Vector(rhs)
        x.Export(x_sol, self.solve_importer, Epetra.Insert)

        if rhs2 is not None:
            return x, x2

        return x

    def eigs(self, state, return_eigenvectors=False, enable_recycling=False):
        '''Compute the generalized eigenvalues of beta * J(x) * v = alpha * M * v.'''

        parameters = self.parameters.get('Eigenvalue Solver', {})
        arithmetic = parameters.get('Arithmetic', 'complex')

        if arithmetic == 'complex':
            from jadapy import ComplexEpetraInterface as EpetraInterface
            from fvm.JadaHYMLSInterface import ComplexJadaHYMLSInterface as JadaHYMLSInterface
        else:
            from jadapy import EpetraInterface
            from fvm.JadaHYMLSInterface import BorderedJadaHYMLSInterface as JadaHYMLSInterface

        jac_op = EpetraInterface.CrsMatrix(self.jacobian(state))
        mass_op = EpetraInterface.CrsMatrix(self.mass_matrix())

        self.compute_scaling()
        self.scale_matrix(jac_op)
        self.scale_matrix(mass_op)
        self.scale_jacobian()

        if arithmetic == 'complex':
            jada_interface = JadaHYMLSInterface(self)
            prec = jada_interface.prec
        else:
            jada_interface = JadaHYMLSInterface(self, preconditioned_solve=True)
            prec = None

        ret = self._eigs(jada_interface, jac_op, mass_op, prec,
                         state, return_eigenvectors, enable_recycling)

        if return_eigenvectors:
            self.unscale_lhs(ret[1])

        self.unscale_jacobian()

        return ret
