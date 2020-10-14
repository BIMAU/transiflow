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

class Interface(continuation.Interface):
    def __init__(self, comm, params, nx, ny, nz):
        continuation.Interface.__init__(self, nx, ny, nz)

        dof = 4
        n = nx * ny * nz * dof
        self.comm = comm
        self.map = Epetra.Map(n, 0, comm)
        self.params = params
        problem_params = self.params.sublist('Problem')
        problem_params.set('nx', self.nx)
        problem_params.set('ny', self.ny)
        problem_params.set('nz', self.nz)
        problem_params.set('Equations', 'Stokes-C')

        solver_params = self.params.sublist('Solver')
        solver_params.set('Initial Vector', 'Zero')

        prec_params = self.params.sublist('Preconditioner')
        prec_params.set('Partitioner', 'Skew Cartesian')

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
