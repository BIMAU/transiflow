import continuation

from PyTrilinos import Epetra
from PyTrilinos import Amesos
from PyTrilinos import Teuchos

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
    def __init__(self, m, nx, ny, nz):
        continuation.Interface.__init__(self, nx, ny, nz)
        self.map = m

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

        params = Teuchos.ParameterList()
        problem_params = params.sublist('Problem')
        problem_params.set('nx', self.nx)
        problem_params.set('ny', self.ny)
        problem_params.set('nz', self.nz)
        problem_params.set('Equations', 'Stokes-C')

        solver_params = params.sublist('Solver')
        solver_params.set('Initial Vector', 'Zero')

        prec_params = params.sublist('Preconditioner')
        prec_params.set('Partitioner', 'Skew Cartesian')
        prec_params.set('Separator Length', 4)
        prec_params.set('Number of Levels', 0)

        preconditioner = HYMLS.Preconditioner(A, params)
        preconditioner.Initialize()
        preconditioner.Compute()

        solver = HYMLS.Solver(A, preconditioner, params)
        solver.ApplyInverse(b, x)

        return x
