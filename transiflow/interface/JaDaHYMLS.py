import HYMLS

from PyTrilinos import Epetra

from jadapy import EpetraInterface
from jadapy import ComplexEpetraInterface

class PrecOp(EpetraInterface.Operator):
    def __init__(self, op, prec):
        super().__init__(op)
        self.prec = prec

    def ApplyInverse(self, x, y):
        self.prec.ApplyInverse(x, y)

        # Create a view here because this is an Epetra.MultiVector.
        z = EpetraInterface.Vector(Epetra.View, y, 0, y.NumVectors())
        z = self.op.proj(z)
        return y.Update(1.0, z, 0.0)

class Interface(EpetraInterface.EpetraInterface):

    def __init__(self, interface, *args, **kwargs):
        super().__init__(interface.solve_map)
        self.interface = interface
        self.parameters = interface.get_teuchos_parameters()
        self.parameters.setParameters(self.parameters.sublist('Eigenvalue Solver'))

        self.preconditioned_solve = kwargs.get('preconditioned_solve', False)

        self.interface.preconditioner.Compute()

    def solve(self, op, rhs, tol, maxit):
        solver_parameters = self.parameters.sublist('Solver')

        iterative_solver_parameters = solver_parameters.sublist('Iterative Solver')
        iterative_solver_parameters.set('Convergence Tolerance', tol if maxit > 1 else 1e-3)
        # iterative_solver_parameters.set('Maximum Iterations', maxit)

        if rhs.shape[1] == 2:
            solver_parameters.set('Complex', True)
        else:
            solver_parameters.set('Complex', False)

        out = EpetraInterface.Vector(rhs)

        epetra_op = EpetraInterface.Operator(op)
        if self.preconditioned_solve:
            epetra_precop = PrecOp(op, self.interface.preconditioner)
            solver = HYMLS.Solver(epetra_op, epetra_precop, self.parameters)
        else:
            solver = HYMLS.Solver(epetra_op, epetra_op, self.parameters)
        solver.ApplyInverse(rhs, out)

        return out

    def prec(self, x, *args):
        out = EpetraInterface.Vector(x)
        self.interface.preconditioner.ApplyInverse(x, out)
        return out

class ComplexPrecOp(EpetraInterface.Operator):
    def __init__(self, op, prec):
        super().__init__(op)
        self.prec = prec

    def ApplyInverse(self, x, y):
        self.prec.ApplyInverse(x, y)

        assert x.NumVectors() == 2
        # Create a view here because this is an Epetra.MultiVector.
        y = EpetraInterface.Vector(Epetra.View, y, 0, y.NumVectors())
        y = ComplexEpetraInterface.ComplexVector(y[:, 0], y[:, 1])

        z = self.op.proj(y)
        y *= 0.0
        y += z
        return 0

class ComplexInterface(ComplexEpetraInterface.ComplexEpetraInterface):

    def __init__(self, interface, *args, **kwargs):
        super().__init__(interface.solve_map)
        self.interface = interface
        self.parameters = interface.get_teuchos_parameters()
        self.parameters.setParameters(self.parameters.sublist('Eigenvalue Solver'))

        self.preconditioned_solve = kwargs.get('preconditioned_solve', False)

        self.interface.preconditioner.Compute()

    def solve(self, op, rhs, tol, maxit):
        solver_parameters = self.parameters.sublist('Solver')

        iterative_solver_parameters = solver_parameters.sublist('Iterative Solver')
        iterative_solver_parameters.set('Convergence Tolerance', tol if maxit > 1 else 1e-3)
        # iterative_solver_parameters.set('Maximum Iterations', maxit)

        solver_parameters.set('Complex', True)

        x = EpetraInterface.Vector(rhs.real.Map(), 2)
        y = EpetraInterface.Vector(rhs.real.Map(), 2)
        x[:, 0] = rhs.real
        x[:, 1] = rhs.imag

        epetra_op = ComplexEpetraInterface.Operator(op)
        if self.preconditioned_solve:
            epetra_precop = ComplexPrecOp(op, self.interface.preconditioner)
            solver = HYMLS.Solver(epetra_op, epetra_precop, self.parameters)
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

class ShiftedOperator(object):
    def __init__(self, op):
        self.A = op.A
        self.B = op.B
        self.prec = op.prec
        self.Q = op.Q
        self.Z = op.Z
        self.Y = op.Y
        self.H = op.H
        self.alpha = op.alpha
        self.beta = op.beta

        self.dtype = self.Q.dtype
        self.shape = self.A.shape

    def matvec(self, x):
        return (self.A @ x) * self.beta - (self.B @ x) * self.alpha

class BorderedInterface(EpetraInterface.EpetraInterface):

    def __init__(self, interface, *args, **kwargs):
        super().__init__(interface.solve_map)
        self.interface = interface
        self.parameters = interface.get_teuchos_parameters()
        self.parameters.setParameters(self.parameters.sublist('Eigenvalue Solver'))

        self.preconditioned_solve = kwargs.get('preconditioned_solve', True)

        self.interface.preconditioner.Compute()

    def solve(self, op, rhs, tol, maxit):
        solver_parameters = self.parameters.sublist('Solver')

        iterative_solver_parameters = solver_parameters.sublist('Iterative Solver')
        iterative_solver_parameters.set('Convergence Tolerance', tol if maxit > 1 else 1e-3)
        # iterative_solver_parameters.set('Maximum Iterations', maxit)

        if rhs.shape[1] == 2:
            solver_parameters.set('Complex', True)
        else:
            solver_parameters.set('Complex', False)
        solver_parameters.set('Use Bordering', True)

        out = EpetraInterface.Vector(rhs)

        epetra_op = EpetraInterface.Operator(ShiftedOperator(op))
        if self.preconditioned_solve:
            solver = HYMLS.Solver(epetra_op, self.interface.preconditioner, self.parameters)
            solver.SetBorder(op.Z, op.Q)
            self.interface.preconditioner.Compute()
            solver.ApplyInverse(rhs, out)
            solver.UnsetBorder()
        else:
            raise Exception('Not implemented')

        return out

    def prec(self, x, *args):
        out = EpetraInterface.Vector(x)
        self.interface.preconditioner.ApplyInverse(x, out)
        return out

class ComplexBorderedInterface(ComplexEpetraInterface.ComplexEpetraInterface):

    def __init__(self, interface, *args, **kwargs):
        super().__init__(interface.solve_map)
        self.interface = interface
        self.parameters = interface.get_teuchos_parameters()
        self.parameters.setParameters(self.parameters.sublist('Eigenvalue Solver'))

        self.preconditioned_solve = kwargs.get('preconditioned_solve', True)

        self.interface.preconditioner.Compute()

    def solve(self, op, rhs, tol, maxit):
        solver_parameters = self.parameters.sublist('Solver')

        iterative_solver_parameters = solver_parameters.sublist('Iterative Solver')
        iterative_solver_parameters.set('Convergence Tolerance', tol if maxit > 1 else 1e-3)
        # iterative_solver_parameters.set('Maximum Iterations', maxit)

        solver_parameters.set('Complex', True)
        solver_parameters.set('Use Bordering', True)

        x = EpetraInterface.Vector(rhs.real.Map(), 2)
        y = EpetraInterface.Vector(rhs.real.Map(), 2)
        x[:, 0] = rhs.real
        x[:, 1] = rhs.imag

        m = op.Q.real.NumVectors()

        Q = EpetraInterface.Vector(rhs.real.Map(), m * 2)
        Q[:, 0:m] = op.Q.real
        Q[:, m:2*m] = op.Q.imag

        Z = EpetraInterface.Vector(rhs.real.Map(), m * 2)
        Z[:, 0:m] = op.Z.real
        Z[:, m:2*m] = op.Z.imag

        epetra_op = ComplexEpetraInterface.Operator(ShiftedOperator(op))
        if self.preconditioned_solve:
            solver = HYMLS.Solver(epetra_op, self.interface.preconditioner, self.parameters)
            solver.SetBorder(Z, Q)
            self.interface.preconditioner.Compute()
            solver.ApplyInverse(x, y)
            solver.UnsetBorder()
        else:
            raise Exception('Not implemented')

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
