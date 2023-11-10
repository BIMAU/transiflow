import numpy
import functools

from scipy import sparse
from scipy.sparse import linalg

from transiflow import CrsMatrix

from transiflow.interface import BaseInterface


class Interface(BaseInterface):
    '''This class defines an interface to the SciPy backend for the
    discretization. We use this so we can write higher level methods
    such as pseudo-arclength continuation without knowing anything
    about the underlying methods such as the solvers that are present
    in the backend we are interfacing with.'''
    def __init__(self, parameters, nx, ny, nz, dim, dof, x=None, y=None, z=None):
        super().__init__(parameters, nx, ny, nz, dim, dof, x, y, z)

        # Select one pressure node to fix
        self.pressure_row = self.dim

        # Solver caching
        self.border_scaling = 1e-3
        self._lu = None
        self._prec = None

        self._gmres_iterations = 0

    def vector(self):
        return numpy.zeros(self.nx * self.ny * self.nz * self.dof)

    def vector_from_array(self, array):
        return array

    def rhs(self, state):
        '''Right-hand side in M * du / dt = F(u).'''
        return self.discretization.rhs(state)

    def jacobian(self, state):
        '''Jacobian J of F in M * du / dt = F(u).'''
        jac = self.discretization.jacobian(state)
        return sparse.csr_matrix((jac.coA, jac.jcoA, jac.begA))

    def mass_matrix(self):
        '''Mass matrix M in M * du / dt = F(u).'''
        mass = self.discretization.mass_matrix()
        return sparse.csr_matrix((mass.coA, mass.jcoA, mass.begA), (mass.n, mass.n))

    @staticmethod
    def _add_custom_methods(matrix):
        if hasattr(matrix, 'lu'):
            return

        matrix.solve = functools.partial(CrsMatrix._solve, matrix)
        matrix.bordered_lu = None
        matrix.lu = None
        matrix.n = matrix.shape[0]

    def compute_bordered_matrix(self, jac, V=None, W=None, C=None, fix_pressure_row=False):
        '''Helper to compute a bordered matrix of the form [A, V; W', C]'''
        def _get_value(V, i, j):
            if not hasattr(V, 'shape') or len(V.shape) < 1:
                return V

            if len(V.shape) < 2:
                return V[i]

            return V[i, j]

        self._add_custom_methods(jac)

        if fix_pressure_row and V is not None:
            self.debug_print(
                'Fixing pressure at row %d of the Jacobian matrix and adding the border' %
                self.pressure_row)
        elif V is not None:
            self.debug_print('Adding the border to the Jacobian matrix')
        else:
            self.debug_print('Fixing pressure at row %d of the Jacobian matrix' %
                             self.pressure_row)

        border_size = 0
        dtype = jac.dtype
        if V is not None and len(V.shape) > 1:
            border_size = V.shape[1]
            dtype = V.dtype
        elif V is not None:
            border_size = 1
            dtype = V.dtype

        extra_border_space = jac.n * border_size * 2 + border_size * border_size

        if W is None:
            W = V

        if C is None and border_size:
            C = numpy.zeros((border_size, border_size), dtype=jac.data.dtype)

        coA = numpy.zeros(jac.indptr[-1] + extra_border_space + 1, dtype=dtype)
        jcoA = numpy.zeros(jac.indptr[-1] + extra_border_space + 1, dtype=int)
        begA = numpy.zeros(len(jac.indptr) + border_size, dtype=int)

        idx = 0
        for i in range(jac.n):
            if fix_pressure_row and i == self.pressure_row:
                coA[idx] = -1.0
                jcoA[idx] = i
                idx += 1
                begA[i + 1] = idx
                continue

            for j in range(jac.indptr[i], jac.indptr[i + 1]):
                if not fix_pressure_row or jac.indices[j] != self.pressure_row:
                    coA[idx] = jac.data[j]
                    jcoA[idx] = jac.indices[j]
                    idx += 1

            for j in range(border_size):
                coA[idx] = self.border_scaling * _get_value(V, i, j)
                jcoA[idx] = jac.n + j
                idx += 1

            begA[i + 1] = idx

        for i in range(border_size):
            for j in range(jac.n):
                coA[idx] = self.border_scaling * _get_value(W, j, i)
                jcoA[idx] = j
                idx += 1

            for j in range(border_size):
                coA[idx] = self.border_scaling * self.border_scaling * _get_value(C, i, j)
                jcoA[idx] = jac.n + j
                idx += 1

            begA[jac.n + 1 + i] = idx

        return CrsMatrix(coA, jcoA, begA, False)

    def _compute_factorization(self, jac, V=None, W=None, C=None):
        '''Compute the LU factorization of the (bordered) jacobian.'''

        self._lu = None
        self._prec = None
        jac.lu = None

        fix_pressure_row = self.dof > self.dim and self.pressure_row is not None
        A = self.compute_bordered_matrix(jac, V, W, C, fix_pressure_row)

        # Convert the matrix to CSC format since splu expects that
        A = sparse.csr_matrix((A.coA, A.jcoA, A.begA)).tocsc()

        self.debug_print(
            'Computing the sparse LU factorization of the %s Jacobian matrix' % (
                'bordered ' if V is not None else ''))

        jac.lu = linalg.splu(A)
        jac.bordered_lu = V is not None

        # Cache the factorization for use in the iterative solver
        self._lu = jac.lu
        self._prec = linalg.LinearOperator((jac.n, jac.n),
                                           matvec=self._lu.solve,
                                           dtype=jac.dtype)

        self.debug_print(
            'Done computing the sparse LU factorization of the %s Jacobian matrix' % (
                'bordered ' if V is not None else ''))

    def _compute_preconditioner(self, jac, A):
        '''Compute the ILU factorization of the (bordered) jacobian.'''

        self._lu = None
        self._prec = None
        jac.lu = None

        # Convert the matrix to CSC format since spilu expects that
        A = sparse.csr_matrix((A.coA, A.jcoA, A.begA)).tocsc()

        self.debug_print('Computing the sparse ILU factorization of the Jacobian matrix')

        parameters = self.parameters.get('Preconditioner', {})
        jac.lu = linalg.spilu(A,
                              drop_tol=parameters.get('Drop Tolerance', None),
                              fill_factor=parameters.get('Fill Factor', None),
                              drop_rule=parameters.get('Drop Rule', None))
        jac.bordered_lu = A.shape != jac.shape

        # Cache the factorization for use in the iterative solver
        self._lu = jac.lu
        self._prec = linalg.LinearOperator(A.shape,
                                           matvec=self._lu.solve,
                                           dtype=A.dtype)

        self.debug_print('Done computing the sparse ILU factorization of the Jacobian matrix')

    def direct_solve(self, jac, rhs, rhs2=None, V=None, W=None, C=None):
        '''Solve J y = x for y.'''
        x = rhs.copy()

        # Fix one pressure node
        if self.dof > self.dim:
            if len(x.shape) < 2:
                x[self.pressure_row] = 0
            else:
                x[self.pressure_row, :] = 0

        # First try to use an iterative solver with the previous
        # direct solver as preconditioner
        if self._prec and jac.dtype == x.dtype and jac.dtype == self._prec.dtype and \
           self.parameters.get('Use Iterative Solver', False):
            out, info = linalg.gmres(jac,
                                     x,
                                     restart=5,
                                     maxiter=1,
                                     tol=1e-8,
                                     atol=0,
                                     M=self._prec)
            if info == 0:
                return out

        if rhs2 is not None:
            x = numpy.append(rhs, rhs2 * self.border_scaling)

        self._add_custom_methods(jac)

        # Use a direct solver instead
        if rhs2 is None and (not jac.lu or jac.bordered_lu):
            self._compute_factorization(jac)
        elif rhs2 is not None and (not jac.lu or not jac.bordered_lu):
            self._compute_factorization(jac, V, W, C)

        if jac.bordered_lu:
            self.debug_print('Solving a bordered linear system')

            y = jac.solve(x)

            border_size = 1
            if hasattr(rhs2, 'shape') and len(rhs2.shape) > 0:
                border_size = rhs2.shape[0]

            y1 = y[:-border_size]
            y2 = y[-border_size:] * self.border_scaling

            self.debug_print_residual('Done solving a bordered linear system with residual',
                                      jac, y1, rhs - V * y2)

            return y1, y2

        self.debug_print('Solving a linear system')

        y = jac.solve(x)

        self.debug_print_residual('Done solving a linear system with residual', jac, y, rhs)

        return y

    def iterative_solve(self, jac, rhs, rhs2=None, V=None, W=None, C=None):
        '''Solve J y = x for y.'''
        x = rhs.copy()

        # Fix one pressure node
        if self.dof > self.dim:
            if len(x.shape) < 2:
                x[self.pressure_row] = 0
            else:
                x[self.pressure_row, :] = 0

        if rhs2 is not None:
            x = numpy.append(rhs, rhs2 * self.border_scaling)

        fix_pressure_row = self.dof > self.dim and self.pressure_row is not None
        A = self.compute_bordered_matrix(jac, V, W, C, fix_pressure_row)

        self._add_custom_methods(jac)

        if rhs2 is None and (not jac.lu or jac.bordered_lu):
            self._compute_preconditioner(jac, A)
        elif rhs2 is not None and (not jac.lu or not jac.bordered_lu):
            self._compute_preconditioner(jac, A)

        self.debug_print('Solving a linear system')

        self._gmres_iterations = 0

        def callback(_r):
            self._gmres_iterations += 1

        parameters = self.parameters.get('Iterative Solver', {})
        restart = parameters.get('Restart', 100)
        maxiter = parameters.get('Maximum Iterations', 1000) / restart
        tol = parameters.get('Convergence Tolerance', 1e-6)

        y, info = linalg.gmres(A, x, restart=restart, maxiter=maxiter,
                               tol=tol, atol=0, M=self._prec,
                               callback=callback, callback_type='pr_norm')
        if info != 0:
            Exception('GMRES did not converge')

        if jac.bordered_lu:
            border_size = 1
            if hasattr(rhs2, 'shape') and len(rhs2.shape) > 0:
                border_size = rhs2.shape[0]

            y1 = y[:-border_size]
            y2 = y[-border_size:] * self.border_scaling

            self.debug_print_residual('Done solving a bordered linear system in %d iterations with residual' %
                                      self._gmres_iterations, jac, y1, rhs - V * y2)

            return y1, y2

        self.debug_print_residual('Done solving a linear system in %d iterations with residual' %
                                  self._gmres_iterations, jac, y, rhs)

        return y

    def solve(self, jac, rhs, rhs2=None, V=None, W=None, C=None):
        if self.parameters.get('Use Iterative Solver', False):
            return self.iterative_solve(jac, rhs, rhs2, V, W, C)

        return self.direct_solve(jac, rhs, rhs2, V, W, C)

    def eigs(self, state, return_eigenvectors=False, enable_recycling=False):
        '''Compute the generalized eigenvalues of beta * J(x) * v = alpha * M * v.'''

        from transiflow.interface.JaDa import Op

        parameters = self.parameters.get('Eigenvalue Solver', {})
        arithmetic = parameters.get('Arithmetic', 'complex')

        jac_op = Op(self.jacobian(state))
        mass_op = Op(self.mass_matrix())
        prec = None

        if self.parameters.get('Bordered Solver', False):
            from transiflow.interface.JaDa import BorderedInterface as JaDaInterface
        else:
            from transiflow.interface.JaDa import Interface as JaDaInterface

        jada_interface = JaDaInterface(self, jac_op, mass_op, jac_op.shape[0], numpy.complex128)
        if arithmetic == 'real':
            jada_interface = JaDaInterface(self, jac_op, mass_op, jac_op.shape[0])

        if not self.parameters.get('Bordered Solver', False):
            prec = jada_interface.shifted_prec

        return self._eigs(jada_interface, jac_op, mass_op, prec, state, return_eigenvectors,
                          enable_recycling)
