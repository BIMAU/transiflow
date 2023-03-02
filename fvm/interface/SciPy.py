import numpy

from scipy import sparse
from scipy.sparse import linalg

from fvm import CrsMatrix

from fvm.interface import BaseInterface


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

    def vector(self):
        return numpy.zeros(self.nx * self.ny * self.nz * self.dof)

    def vector_from_array(self, array):
        return array

    def rhs(self, state):
        '''Right-hand side in M * du / dt = F(u).'''
        return self.discretization.rhs(state)

    def jacobian(self, state):
        '''Jacobian J of F in M * du / dt = F(u).'''
        return self.discretization.jacobian(state)

    def mass_matrix(self):
        '''Mass matrix M in M * du / dt = F(u).'''
        return self.discretization.mass_matrix()

    def _compute_factorization(self, jac):
        '''Compute the LU factorization of jac.'''
        self._lu = None
        self._prec = None
        jac.lu = None

        coA = jac.coA
        jcoA = jac.jcoA
        begA = jac.begA

        # Fix one pressure node
        if self.dof > self.dim and self.pressure_row is not None:
            self.debug_print('Fixing pressure at row %d of the Jacobian matrix' %
                             self.pressure_row)

            coA = numpy.zeros(jac.begA[-1] + 1, dtype=jac.coA.dtype)
            jcoA = numpy.zeros(jac.begA[-1] + 1, dtype=int)
            begA = numpy.zeros(len(jac.begA), dtype=int)

            idx = 0
            for i in range(len(jac.begA) - 1):
                if i == self.pressure_row:
                    coA[idx] = -1.0
                    jcoA[idx] = i
                    idx += 1
                    begA[i + 1] = idx
                    continue
                for j in range(jac.begA[i], jac.begA[i + 1]):
                    if jac.jcoA[j] != self.pressure_row:
                        coA[idx] = jac.coA[j]
                        jcoA[idx] = jac.jcoA[j]
                        idx += 1

                begA[i + 1] = idx

        # Convert the matrix to CSC format since splu expects that
        A = sparse.csr_matrix((coA, jcoA, begA)).tocsc()

        self.debug_print('Computing the sparse LU factorization of the Jacobian matrix')

        jac.lu = linalg.splu(A)
        jac.bordered_lu = False

        # Cache the factorization for use in the iterative solver
        self._lu = jac.lu
        self._prec = linalg.LinearOperator((jac.n, jac.n),
                                           matvec=self._lu.solve,
                                           dtype=jac.dtype)

        self.debug_print('Done computing the sparse LU factorization of the Jacobian matrix')

    def compute_bordered_matrix(self, jac, V, W=None, C=None, fix_pressure_row=False):
        '''Helper to compute a bordered matrix of the form [A, V; W', C]'''
        def _get_value(V, i, j):
            if not hasattr(V, 'shape') or len(V.shape) < 1:
                return V

            if len(V.shape) < 2:
                return V[i]

            return V[i, j]

        if V is None:
            raise Exception('V is None')

        if fix_pressure_row:
            self.debug_print(
                'Fixing pressure at row %d of the Jacobian matrix and adding the border' %
                self.pressure_row)
        else:
            self.debug_print('Adding the border to the Jacobian matrix')

        border_size = 1
        if len(V.shape) > 1:
            border_size = V.shape[1]

        extra_border_space = jac.n * border_size * 2 + border_size * border_size

        if W is None:
            W = V

        if C is None:
            C = numpy.zeros((border_size, border_size), dtype=jac.coA.dtype)

        coA = numpy.zeros(jac.begA[-1] + extra_border_space + 1, dtype=V.dtype)
        jcoA = numpy.zeros(jac.begA[-1] + extra_border_space + 1, dtype=int)
        begA = numpy.zeros(len(jac.begA) + border_size, dtype=int)

        idx = 0
        for i in range(jac.n):
            if fix_pressure_row and i == self.pressure_row:
                coA[idx] = -1.0
                jcoA[idx] = i
                idx += 1
                begA[i + 1] = idx
                continue

            for j in range(jac.begA[i], jac.begA[i + 1]):
                if not fix_pressure_row or jac.jcoA[j] != self.pressure_row:
                    coA[idx] = jac.coA[j]
                    jcoA[idx] = jac.jcoA[j]
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

    def _compute_bordered_factorization(self, jac, V, W, C):
        '''Compute the LU factorization of the bordered jacobian.'''

        self._lu = None
        self._prec = None
        jac.lu = None

        fix_pressure_row = self.dof > self.dim and self.pressure_row is not None
        A = self.compute_bordered_matrix(jac, V, W, C, fix_pressure_row)

        # Convert the matrix to CSC format since splu expects that
        A = sparse.csr_matrix((A.coA, A.jcoA, A.begA)).tocsc()

        self.debug_print(
            'Computing the sparse LU factorization of the bordered Jacobian matrix')

        jac.lu = linalg.splu(A)
        jac.bordered_lu = True

        # Cache the factorization for use in the iterative solver
        self._lu = jac.lu
        self._prec = linalg.LinearOperator((jac.n, jac.n),
                                           matvec=self._lu.solve,
                                           dtype=jac.dtype)

        self.debug_print(
            'Done computing the sparse LU factorization of the bordered Jacobian matrix')

    def solve(self, jac, rhs, rhs2=None, V=None, W=None, C=None):
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

        # Use a direct solver instead
        if rhs2 is None and (not jac.lu or jac.bordered_lu):
            self._compute_factorization(jac)
        elif rhs2 is not None and (not jac.lu or not jac.bordered_lu):
            self._compute_bordered_factorization(jac, V, W, C)

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

    def eigs(self, state, return_eigenvectors=False, enable_recycling=False):
        '''Compute the generalized eigenvalues of beta * J(x) * v = alpha * M * v.'''

        from fvm.interface.JaDa import Op

        parameters = self.parameters.get('Eigenvalue Solver', {})
        arithmetic = parameters.get('Arithmetic', 'complex')

        jac_op = Op(self.jacobian(state))
        mass_op = Op(self.mass_matrix())
        prec = None

        if self.parameters.get('Bordered Solver', False):
            from fvm.interface.JaDa import BorderedInterface as JaDaInterface
        else:
            from fvm.interface.JaDa import Interface as JaDaInterface

        jada_interface = JaDaInterface(self, jac_op, mass_op, jac_op.shape[0], numpy.complex128)
        if arithmetic == 'real':
            jada_interface = JaDaInterface(self, jac_op, mass_op, jac_op.shape[0])

        if not self.parameters.get('Bordered Solver', False):
            prec = jada_interface.shifted_prec

        return self._eigs(jada_interface, jac_op, mass_op, prec, state, return_eigenvectors,
                          enable_recycling)
