from PyTrilinos import Epetra
from PyTrilinos import Amesos

import fvm

import HYMLS

class Vector(Epetra.Vector):
    def __neg__(self):
        v = Vector(self)
        v.Scale(-1.0)
        return v

    def __truediv__(self, scal):
        v = Vector(self)
        v.Scale(1.0 / scal)
        return v

    def dot(self, other):
        return self.Dot(other)

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
    return ((k *ny + j) * nx + i) * dof + var

class Interface(fvm.Interface):
    def __init__(self, comm, parameters, nx, ny, nz, dof):
        fvm.Interface.__init__(self, parameters, nx, ny, nz, dof)

        self.nx_global = nx
        self.ny_global = ny
        self.nz_global = nz

        self.comm = comm

        HYMLS.Tools.InitializeIO(self.comm);

        self.parameters = parameters
        problem_parameters = self.parameters.sublist('Problem')
        problem_parameters.set('nx', self.nx_global)
        problem_parameters.set('ny', self.ny_global)
        problem_parameters.set('nz', self.nz_global)
        problem_parameters.set('Equations', 'Stokes-C')

        solver_parameters = self.parameters.sublist('Solver')
        solver_parameters.set('Initial Vector', 'Zero')

        prec_parameters = self.parameters.sublist('Preconditioner')
        prec_parameters.set('Partitioner', 'Skew Cartesian')

        self.partition_domain()
        self.map = self.create_map()

        self.assembly_map = self.create_map(True)
        self.assembly_importer = Epetra.Import(self.assembly_map, self.map)

        partitioner = HYMLS.SkewCartesianPartitioner(self.parameters, self.comm)
        partitioner.Partition()

        self.solve_map = partitioner.Map()
        self.solve_importer = Epetra.Import(self.solve_map, self.map)

    def partition_domain(self):
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
                    nx_loc = self.nx // t1
                    ny_loc = self.ny // t2
                    nz_loc = self.nz // t3

                    if nx_loc * t1 != self.nx or ny_loc * t2 != self.ny or nz_loc * t3 != self.nz:
                        continue

                    r1 = abs(self.nx / t1 - self.ny / t2)
                    r2 = abs(self.nx / t1 - self.nz / t3)

                    r3 = abs(self.ny / t2 - self.nz / t3)
                    r = r1 + r2 + r3

                    if r < rmin:
                        rmin = r
                        self.npx = t1
                        self.npy = t2
                        self.npz = t3
                        found = True

        if not found:
            raise Exception('Could not split %dx%dx%d domain in %d parts.' % (self.nx, self.ny, self.nz, nparts))

        self.pidx, self.pidy, self.pidz, _ = ind2sub(self.npx, self.npy, self.npz, pid)

        # Compute the local domain size and offset.
        self.nx_local = self.nx // self.npx
        self.ny_local = self.ny // self.npy
        self.nz_local = self.nz // self.npz

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

        self.nx = self.nx_local
        self.ny = self.ny_local
        self.nz = self.nz_local

    def is_ghost(self, i, j=None, k=None):
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

    def jacobian(self, state):
        state_ass = Vector(self.assembly_map)
        state_ass.Import(state, self.assembly_importer, Epetra.Insert)

        jac = fvm.Interface.jacobian(self, state_ass)

        A = Epetra.FECrsMatrix(Epetra.Copy, self.solve_map, 27)
        for i in range(len(jac.begA)-1):
            if self.is_ghost(i):
                continue
            row = self.assembly_map.GID64(i)
            for j in range(jac.begA[i], jac.begA[i+1]):
                A.InsertGlobalValue(row, self.assembly_map.GID64(jac.jcoA[j]), jac.coA[j])
        A.GlobalAssemble(True, Epetra.Insert)

        return A

    def rhs(self, state):
        state_ass = Vector(self.assembly_map)
        state_ass.Import(state, self.assembly_importer, Epetra.Insert)

        rhs = fvm.Interface.rhs(self, state_ass)
        rhs_ass = Vector(Epetra.Copy, self.assembly_map, rhs)
        rhs = Vector(self.map)
        rhs.Export(rhs_ass, self.assembly_importer, Epetra.Zero)
        return rhs

    def dirtect_solve(self, jac, rhs):
        A = Epetra.CrsMatrix(Epetra.Copy, self.map, 27)
        for i in range(len(jac.begA)-1):
            if i == 3:
                A[i, i] = -1
                continue
            for j in range(jac.begA[i], jac.begA[i+1]):
                if jac.jcoA[j] != 3:
                    A[i, jac.jcoA[j]] = jac.coA[j]
        A.FillComplete()

        rhs[3] = 0
        x = Vector(rhs)

        problem = Epetra.LinearProblem(A, x, rhs)
        factory = Amesos.Factory()
        solver = factory.Create('Klu', problem)
        solver.SymbolicFactorization()
        solver.NumericFactorization()
        solver.Solve()

        return x

    def solve(self, jac, rhs):
        rhs_sol = Vector(self.solve_map)
        rhs_sol.Import(rhs, self.solve_importer, Epetra.Insert)
        x_sol = Vector(rhs_sol)

        preconditioner = HYMLS.Preconditioner(jac, self.parameters)
        preconditioner.Initialize()
        preconditioner.Compute()

        solver = HYMLS.Solver(jac, preconditioner, self.parameters)
        solver.ApplyInverse(rhs_sol, x_sol)

        x = Vector(rhs)
        x.Export(x_sol, self.solve_importer, Epetra.Insert)

        return x
