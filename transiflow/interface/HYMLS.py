from PyTrilinos import Epetra
from PyTrilinos import Teuchos

import numpy
import sys
import os

from transiflow.interface.Epetra import Vector
from transiflow.interface.Epetra import Interface as EpetraInterface

import HYMLS

def set_default_parameter(parameterlist, name, value):
    if name not in parameterlist:
        parameterlist[name] = value

    return parameterlist[name]

def convert_parameters(parameters, teuchos_parameters=None):
    if isinstance(parameters, Teuchos.ParameterList):
        return Teuchos.ParameterList(parameters)

    if teuchos_parameters is None:
        teuchos_parameters = Teuchos.ParameterList()

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

class Interface(EpetraInterface):
    '''This class defines an interface to the HYMLS backend for the
    discretization. We use this so we can write higher level methods
    such as pseudo-arclength continuation without knowing anything
    about the underlying methods such as the solvers that are present
    in the backend we are interfacing with.

    The HYMLS backend partitions the domain into Cartesian subdomains,
    while solving linear systems on skew Cartesian subdomains to deal
    with the C-grid discretization. The subdomains will be distributed
    over multiple processors if MPI is used to run the application.'''

    def __init__(self, parameters, nx, ny, nz=1, dim=None, dof=None, comm=None):
        EpetraInterface.__init__(self, parameters, nx, ny, nz, dim, dof, comm)

        # Disable HYMLS output from MPI ranks != 0
        HYMLS.Tools.InitializeIO(self.comm)

        # Do the same for Python output
        self._original_stdout = sys.stdout
        if self.comm.MyPID() != 0:
            self.debug_print('PID %d: Disable output to stdout' % self.comm.MyPID())
            sys.stdout = open(os.devnull, 'w')

        self.teuchos_parameters = self.get_teuchos_parameters()

        partitioner = HYMLS.SkewCartesianPartitioner(self.teuchos_parameters, self.comm)
        partitioner.Partition()

        self.solve_map = partitioner.Map()
        self.solve_importer = Epetra.Import(self.solve_map, self.map)

        self.initialize()

    def __del__(self):
        HYMLS.Tools.PrintTiming()
        HYMLS.Tools.PrintMemUsage()

        if self.comm.MyPID() != 0:
            sys.stdout.close()
            sys.stdout = self._original_stdout
            self.debug_print('PID %d: Re-enable output to stdout' % self.comm.MyPID())

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
        maxit = set_default_parameter(iterative_solver_parameters, 'Maximum Iterations', 1000)
        maxsize = set_default_parameter(iterative_solver_parameters, 'Num Blocks', 100)
        set_default_parameter(iterative_solver_parameters, 'Maximum Restarts', maxit // maxsize)
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
            from transiflow.interface.JaDaHYMLS import ComplexInterface as JaDaHYMLSInterface
        else:
            from jadapy import EpetraInterface
            from transiflow.interface.JaDaHYMLS import BorderedInterface as JaDaHYMLSInterface

        jac_op = EpetraInterface.CrsMatrix(self.jacobian(state))
        mass_op = EpetraInterface.CrsMatrix(self.mass_matrix())

        self.compute_scaling()
        self.scale_matrix(jac_op)
        self.scale_matrix(mass_op)
        self.scale_jacobian()

        if arithmetic == 'complex':
            jada_interface = JaDaHYMLSInterface(self)
            prec = jada_interface.prec
        else:
            jada_interface = JaDaHYMLSInterface(self, preconditioned_solve=True)
            prec = None

        ret = self._eigs(jada_interface, jac_op, mass_op, prec,
                         state, return_eigenvectors, enable_recycling)

        if return_eigenvectors:
            self.unscale_lhs(ret[1])

        self.unscale_jacobian()

        return ret
