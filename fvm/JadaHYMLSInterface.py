import copy

import HYMLS

from jadapy import EpetraInterface
from jadapy import ComplexEpetraInterface

class JadaHYMLSInterface(EpetraInterface.EpetraInterface):

    def __init__(self, map, interface, *args, **kwargs):
        super().__init__(map)
        self.interface = interface
        self.parameters = copy.copy(interface.parameters)

        self.preconditioned_solve = kwargs.get('preconditioned_solve', False)

    def solve(self, op, rhs, tol, maxit):
        solver_parameters = self.parameters.sublist('Solver')

        iterative_solver_parameters = solver_parameters.sublist('Iterative Solver')
        iterative_solver_parameters.set('Convergence Tolerance', tol)
        iterative_solver_parameters.set('Maximum Iterations', maxit)

        if rhs.shape[1] == 2:
            solver_parameters.set('Complex', True)
        else:
            solver_parameters.set('Complex', False)

        out = EpetraInterface.Vector(rhs)

        epetra_op = EpetraInterface.Operator(op)
        if self.preconditioned_solve:
            solver = HYMLS.Solver(epetra_op, self.interface.preconditioner, self.parameters)
        else:
            solver = HYMLS.Solver(epetra_op, epetra_op, self.parameters)
        solver.ApplyInverse(rhs, out)

        return out

    def prec(self, x, *args):
        out = EpetraInterface.Vector(x)
        self.interface.preconditioner.ApplyInverse(x, out)
        return out

class ComplexJadaHYMLSInterface(ComplexEpetraInterface.ComplexEpetraInterface):

    def __init__(self, map, interface, *args, **kwargs):
        super().__init__(map)
        self.interface = interface
        self.parameters = copy.copy(interface.parameters)

        self.preconditioned_solve = kwargs.get('preconditioned_solve', False)

    def solve(self, op, rhs, tol, maxit):
        solver_parameters = self.parameters.sublist('Solver')

        iterative_solver_parameters = solver_parameters.sublist('Iterative Solver')
        iterative_solver_parameters.set('Convergence Tolerance', tol)
        iterative_solver_parameters.set('Maximum Iterations', maxit)

        solver_parameters.set('Complex', True)

        x = EpetraInterface.Vector(rhs.real.Map(), 2)
        y = EpetraInterface.Vector(rhs.real.Map(), 2)
        x[:, 0] = rhs.real
        x[:, 1] = rhs.imag

        epetra_op = ComplexEpetraInterface.Operator(op)
        if self.preconditioned_solve:
            solver = HYMLS.Solver(epetra_op, self.interface.preconditioner, self.parameters)
        else:
            solver = HYMLS.Solver(epetra_op, epetra_op, self.parameters)
        solver.ApplyInverse(x, y)

        out = ComplexEpetraInterface.ComplexVector(y[:, 0], y[:, 1])
        return out

    def prec(self, x, *args):
        y = EpetraInterface.Vector(x.real.Map(), 2)
        z = EpetraInterface.Vector(x.real.Map(), 2)
        y[:, 0] = x.real
        y[:, 1] = x.imag

        self.interface.preconditioner.ApplyInverse(y, z)

        out = ComplexEpetraInterface.ComplexVector(z[:, 0], z[:, 1])
        return out
