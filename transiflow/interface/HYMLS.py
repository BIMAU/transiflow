from PyTrilinos import Epetra
from PyTrilinos import Teuchos

import numpy
import sys
import os

from transiflow.interface.Epetra import Vector
from transiflow.interface.Epetra import Interface as EpetraInterface

import HYMLS

def _set_default_parameter(parameterlist, name, value):
    if name not in parameterlist:
        parameterlist[name] = value

    return parameterlist[name]

def _convert_parameters(parameters, teuchos_parameters=None):
    if isinstance(parameters, Teuchos.ParameterList):
        return Teuchos.ParameterList(parameters)

    if teuchos_parameters is None:
        teuchos_parameters = Teuchos.ParameterList()

    for i, j in parameters.items():
        if isinstance(j, dict):
            sublist = teuchos_parameters.sublist(i)
            _convert_parameters(j, sublist)
        else:
            try:
                teuchos_parameters.set(i, j)
            except Exception:
                pass

    return teuchos_parameters

class Interface(EpetraInterface):
    '''This class defines an interface to the HYMLS backend for the
    discretization. This backend can be used for parallel simulations.
    It uses the HYMLS preconditioner for solving linear systems.

    The HYMLS backend partitions the domain into Cartesian subdomains,
    while solving linear systems on skew Cartesian subdomains to deal
    with the C-grid discretization. The subdomains will be distributed
    over multiple processors if MPI is used to run the application.

    See :mod:`.Discretization` for the descriptions of the constructor
    arguments.

    Parameters
    ----------
    parameters : dict
        Key-value pairs that can be used to modify parameters in the
        discretization as well as the preconditioner, iterative solver
        and eigenvalue solver.

    '''

    def __init__(self, parameters, nx, ny, nz=1, dim=None, dof=None,
                 boundary_conditions=None, comm=None):
        EpetraInterface.__init__(self, parameters, nx, ny, nz, dim, dof,
                                 boundary_conditions, comm)

        # Disable HYMLS output from MPI ranks != 0
        HYMLS.Tools.InitializeIO(self.comm)

        # Do the same for Python output
        self._original_stdout = sys.stdout
        if self.comm.MyPID() != 0:
            self._debug_print('PID %d: Disable output to stdout' % self.comm.MyPID())
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
            self._debug_print('PID %d: Re-enable output to stdout' % self.comm.MyPID())

    def get_teuchos_parameters(self):
        '''Get the parameter list in Teuchos.ParameterList format.
        This is a copy of the internal list to solve issues with the
        ``Output Stream`` parameter and the fact that the ``get()``
        method actually sets values in the parameter list.

        :meta private:

        '''
        teuchos_parameters = _convert_parameters(self.parameters)

        problem_parameters = teuchos_parameters.sublist('Problem')
        _set_default_parameter(problem_parameters, 'nx', self.nx_global)
        _set_default_parameter(problem_parameters, 'ny', self.ny_global)
        _set_default_parameter(problem_parameters, 'nz', self.nz_global)

        _set_default_parameter(problem_parameters, 'Dimension', self.dim)
        _set_default_parameter(problem_parameters, 'Degrees of Freedom', self.dof)
        _set_default_parameter(problem_parameters, 'Equations', 'Stokes-C')

        _set_default_parameter(problem_parameters, 'x-periodic', self.discretization.x_periodic)
        _set_default_parameter(problem_parameters, 'y-periodic', self.discretization.y_periodic)
        _set_default_parameter(problem_parameters, 'z-periodic', self.discretization.z_periodic)

        solver_parameters = teuchos_parameters.sublist('Solver')
        solver_parameters.set('Initial Vector', 'Zero')
        solver_parameters.set('Left or Right Preconditioning', 'Right')

        iterative_solver_parameters = solver_parameters.sublist('Iterative Solver')
        iterative_solver_parameters.set('Output Stream', 0)
        maxit = _set_default_parameter(iterative_solver_parameters, 'Maximum Iterations', 1000)
        maxsize = _set_default_parameter(iterative_solver_parameters, 'Num Blocks', 100)
        _set_default_parameter(iterative_solver_parameters, 'Maximum Restarts', maxit // maxsize)
        _set_default_parameter(iterative_solver_parameters, 'Flexible Gmres', False)
        _set_default_parameter(iterative_solver_parameters, 'Convergence Tolerance', 1e-8)
        _set_default_parameter(iterative_solver_parameters, 'Output Frequency', 1)
        _set_default_parameter(iterative_solver_parameters, 'Show Maximum Residual Norm Only', False)
        _set_default_parameter(iterative_solver_parameters, 'Implicit Residual Scaling', 'Norm of RHS')
        _set_default_parameter(iterative_solver_parameters, 'Explicit Residual Scaling', 'Norm of RHS')

        prec_parameters = teuchos_parameters.sublist('Preconditioner')
        prec_parameters.set('Partitioner', 'Skew Cartesian')
        _set_default_parameter(prec_parameters, 'Separator Length', min(8, self.nx_global))
        _set_default_parameter(prec_parameters, 'Coarsening Factor', 2)
        _set_default_parameter(prec_parameters, 'Number of Levels', 1)

        coarse_solver_parameters = prec_parameters.sublist('Coarse Solver')
        _set_default_parameter(coarse_solver_parameters, "amesos: solver type", "Amesos_Superludist")

        return teuchos_parameters

    def _unset_parameter(self, name, original_parameters):
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
        later, rather than insert them.

        :meta private:

        '''
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

        self._compute_scaling()
        self._scale_jacobian()

        self.preconditioner = HYMLS.Preconditioner(self.jac, self.teuchos_parameters)
        self.preconditioner.Initialize()

        self.solver = HYMLS.BorderedSolver(self.jac, self.preconditioner, self.teuchos_parameters)

        self._unscale_jacobian()

        # Put back the original parameters
        for i in parameter_names:
            self._unset_parameter(i, original_parameters)

    def solve(self, jac, rhs, rhs2=None, V=None, W=None, C=None, solver=None):
        if solver is None:
            solver = self.solver

        rhs_sol = Vector(self.solve_map)
        rhs_sol.Import(rhs, self.solve_importer, Epetra.Insert)

        x_sol = Vector(rhs_sol)

        self._compute_scaling()

        if rhs2 is not None:
            rhs2_sol = Epetra.SerialDenseMatrix(1, 1)
            rhs2_sol[0, 0] = rhs2

            x2_sol = Epetra.SerialDenseMatrix(1, 1)

            V_sol = Vector(self.solve_map)
            V_sol.Import(V, self.solve_importer, Epetra.Insert)

            self._scale_rhs(V_sol)

            W_sol = Vector(self.solve_map)
            W_sol.Import(W, self.solve_importer, Epetra.Insert)

            self._unscale_lhs(W_sol)

            C_sol = Epetra.SerialDenseMatrix(1, 1)
            C_sol[0, 0] = C

            solver.SetBorder(V_sol, W_sol, C_sol)

        self._scale_jacobian()
        self._scale_rhs(rhs_sol)

        self.preconditioner.Compute()
        if rhs2 is not None:
            solver.ApplyInverse(rhs_sol, rhs2_sol, x_sol, x2_sol)

            x2 = x2_sol[0, 0]
        else:
            solver.ApplyInverse(rhs_sol, x_sol)

        solver.UnsetBorder()

        self._unscale_jacobian()
        self._unscale_lhs(x_sol)

        x = Vector(rhs)
        x.Export(x_sol, self.solve_importer, Epetra.Insert)

        if rhs2 is not None:
            return x, x2

        return x

    def eigs(self, state, return_eigenvectors=False, enable_recycling=False):
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

        self._compute_scaling()
        self._scale_matrix(jac_op)
        self._scale_matrix(mass_op)
        self._scale_jacobian()

        if arithmetic == 'complex':
            jada_interface = JaDaHYMLSInterface(self)
            prec = jada_interface.prec
        else:
            jada_interface = JaDaHYMLSInterface(self, preconditioned_solve=True)
            prec = None

        ret = self._eigs(jada_interface, jac_op, mass_op, prec,
                         state, return_eigenvectors, enable_recycling)

        if return_eigenvectors:
            self._unscale_lhs(ret[1])

        self._unscale_jacobian()

        return ret
