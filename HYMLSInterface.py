import continuation

from PyTrilinos import Epetra
from PyTrilinos import Amesos

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

class Interface(continuation.Interface):
    def __init__(self, comm, params, nx, ny, nz):
        continuation.Interface.__init__(self, nx, ny, nz)

        self.nx_global = nx
        self.ny_global = ny
        self.nz_global = nz

        self.dof = 4
        self.comm = comm

        self.partition_domain()
        self.create_map()

        self.params = params
        problem_params = self.params.sublist('Problem')
        problem_params.set('nx', self.nx_global)
        problem_params.set('ny', self.ny_global)
        problem_params.set('nz', self.nz_global)
        problem_params.set('Equations', 'Stokes-C')

        solver_params = self.params.sublist('Solver')
        solver_params.set('Initial Vector', 'Zero')

        prec_params = self.params.sublist('Preconditioner')
        prec_params.set('Partitioner', 'Skew Cartesian')

    def partition_domain(self):
        rmin = 1e100

        sx = 1
        sy = 1
        sz = 1

        nparts = self.comm.NumProc()

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
                        npx = t1
                        npy = t2
                        npz = t3
                        found = True

        if not found:
            raise Exception('Could not split %dx%dx%d domain in %d parts.' % (self.nx, self.ny, self.nz, nparts))

        pidx, pidy, pidz, _ = ind2sub(npx, npy, npz, nparts, 1)

        # Compute the local domain size and offset.
        self.nx_local = self.nx // npx
        self.ny_local = self.ny // npy
        self.nz_local = self.nz // npz

        self.nx_offset = self.nx_local * pidx
        self.ny_offset = self.ny_local * pidy
        self.nz_offset = self.nz_local * pidz

        # Add ghost nodes to factor out boundary conditions in the interior.
        if pidx > 0:
            self.nx_local += 2
            self.nx_offset -= 2
        if pidx < npx - 1:
            self.nx_local += 2
        if pidy > 0:
            self.ny_local += 2
            self.ny_offset -= 2
        if pidy < npy - 1:
            self.ny_local += 2
        if pidz > 0:
            self.nz_local += 2
            self.nz_offset -= 2
        if pidz < npz - 1:
            self.nz_local += 2

        self.nx = self.nx_local
        self.ny = self.ny_local
        self.nz = self.nz_local

    def create_map(self):
        local_elements = [0] * self.nx_local * self.ny_local * self.nz_local * self.dof

        pos = 0
        for k in range(self.nz_offset, self.nz_offset + self.nz_local):
            for j in range(self.ny_offset, self.ny_offset + self.ny_local):
                for i in range(self.nx_offset, self.nx_offset + self.nx_local):
                    for var in range(self.dof):
                        local_elements[pos] = sub2ind(self.nx_global, self.ny_global, self.nz_global, self.dof, i, j, k, var)
                        pos += 1

        self.map = Epetra.Map(-1, local_elements, 0, self.comm)

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

        b = Vector(Epetra.Copy, self.map, rhs)
        b[3] = 0
        x = Vector(b)

        problem = Epetra.LinearProblem(A, x, b)
        factory = Amesos.Factory()
        solver = factory.Create('Klu', problem)
        solver.SymbolicFactorization()
        solver.NumericFactorization()
        solver.Solve()

        return x

    def solve(self, jac, rhs):
        A = Epetra.CrsMatrix(Epetra.Copy, self.map, 27)
        for i in range(len(jac.begA)-1):
            for j in range(jac.begA[i], jac.begA[i+1]):
                A[i, jac.jcoA[j]] = jac.coA[j]
        A.FillComplete()

        b = Vector(Epetra.Copy, self.map, rhs)
        x = Vector(b)

        preconditioner = HYMLS.Preconditioner(A, self.params)
        preconditioner.Initialize()
        preconditioner.Compute()

        solver = HYMLS.Solver(A, preconditioner, self.params)
        solver.ApplyInverse(b, x)

        return x
