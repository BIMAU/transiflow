import numpy

from scipy import sparse
from scipy.sparse import linalg

from transiflow.interface import BaseInterface


class Interface(BaseInterface):
    '''This class defines an interface to the SciPy backend for the
    discretization. This backend can be used for low-resolution serial
    runs as well as implementing new problems in the discretization.

    See :mod:`.Discretization` for the descriptions of the constructor
    arguments.

    Parameters
    ----------
    parameters : dict
        Key-value pairs that can be used to modify parameters in the
        discretization as well as the linear solver and eigenvalue
        solver.

    '''
    def __init__(self, parameters, nx, ny, nz=1, dim=None, dof=None,
                 x=None, y=None, z=None, boundary_conditions=None):
        super().__init__(parameters, nx, ny, nz, dim, dof, x, y, z, boundary_conditions)

        # Select one pressure node to fix
        self.pressure_row = self.dim

        # Solver caching
        self.border_scaling = 1e-3
        self._lu = None
        self._prec = None

    def vector(self):
        return numpy.zeros(self.nx * self.ny * self.nz * self.dof)

    def rhs(self, state):
        return self.discretization.rhs(state)

    def jacobian(self, state):
        jac = self.discretization.jacobian(state)
        return sparse.csr_matrix((jac.coA, jac.jcoA, jac.begA)).tocsc()

    def mass_matrix(self):
        mass = self.discretization.mass_matrix()
        return sparse.csr_matrix((mass.coA, mass.jcoA, mass.begA), (mass.n, mass.n)).tocsc()

    def compute_bordered_matrix(self, jac, V=None, W=None, C=None, fix_pressure_row=False):
        '''Helper to compute a bordered matrix of the form ``[A, V; W^T, C]``.'''
        def _get_value(V, i, j):
            if not hasattr(V, 'shape') or len(V.shape) < 1:
                return V

            if len(V.shape) < 2:
                return V[i]

            return V[i, j]

        if fix_pressure_row and V is not None:
            self._debug_print(
                'Fixing pressure at row %d of the Jacobian matrix and adding the border' %
                self.pressure_row)
        elif V is not None:
            self._debug_print('Adding the border to the Jacobian matrix')
        else:
            self._debug_print('Fixing pressure at row %d of the Jacobian matrix' %
                              self.pressure_row)

        border_size = 0
        dtype = jac.dtype
        if V is not None and len(V.shape) > 1:
            border_size = V.shape[1]
            dtype = V.dtype
        elif V is not None:
            border_size = 1
            dtype = V.dtype

        extra_border_space = jac.shape[0] * border_size * 2 + border_size * border_size

        if W is None:
            W = V

        if C is None and border_size:
            C = numpy.zeros((border_size, border_size), dtype=jac.data.dtype)

        coA = numpy.zeros(jac.indptr[-1] + extra_border_space + 1, dtype=dtype)
        icoA = numpy.zeros(jac.indptr[-1] + extra_border_space + 1, dtype=int)
        begA = numpy.zeros(len(jac.indptr) + border_size, dtype=int)

        idx = 0
        for i in range(jac.shape[0]):
            if fix_pressure_row and i == self.pressure_row:
                coA[idx] = -1.0
                icoA[idx] = i
                idx += 1
                begA[i + 1] = idx
                continue

            for j in range(jac.indptr[i], jac.indptr[i + 1]):
                if not fix_pressure_row or jac.indices[j] != self.pressure_row:
                    coA[idx] = jac.data[j]
                    icoA[idx] = jac.indices[j]
                    idx += 1

            for j in range(border_size):
                coA[idx] = self.border_scaling * _get_value(W, i, j)
                icoA[idx] = jac.shape[0] + j
                idx += 1

            begA[i + 1] = idx

        for i in range(border_size):
            for j in range(jac.shape[0]):
                coA[idx] = self.border_scaling * _get_value(V, j, i)
                icoA[idx] = j
                idx += 1

            for j in range(border_size):
                coA[idx] = self.border_scaling * self.border_scaling * _get_value(C, j, i)
                icoA[idx] = jac.shape[0] + j
                idx += 1

            begA[jac.shape[0] + 1 + i] = idx

        n = len(begA) - 1
        return sparse.csc_matrix((coA, icoA, begA), (n, n))

    def _compute_factorization(self, jac, V, W, C):
        '''Compute the LU factorization of the (bordered) jacobian.'''

        if V is not None and hasattr(jac, 'lu') and jac.lu is not None and jac.bordered_lu:
            return

        if V is None and hasattr(jac, 'lu') and jac.lu is not None and not jac.bordered_lu:
            return

        self._lu = None
        self._prec = None
        jac.lu = None

        fix_pressure_row = self.dof > self.dim and self.pressure_row is not None
        A = self.compute_bordered_matrix(jac, V, W, C, fix_pressure_row)

        self._debug_print(
            'Computing the sparse LU factorization of the %s Jacobian matrix' % (
                'bordered ' if V is not None else ''))

        jac.lu = linalg.splu(A)
        jac.bordered_lu = V is not None

        # Cache the factorization for use in the iterative solver
        self._lu = jac.lu
        self._prec = linalg.LinearOperator(jac.shape,
                                           matvec=self._lu.solve,
                                           dtype=jac.dtype)

        self._debug_print(
            'Done computing the sparse LU factorization of the %s Jacobian matrix' % (
                'bordered ' if V is not None else ''))

    def _compute_preconditioner(self, jac, A, V, _W, _C):
        '''Compute the ILU factorization of the (bordered) jacobian.'''

        if V is not None and hasattr(jac, 'lu') and jac.lu is not None and jac.bordered_lu:
            return

        if V is None and hasattr(jac, 'lu') and jac.lu is not None and not jac.bordered_lu:
            return

        self._lu = None
        self._prec = None
        jac.lu = None

        self._debug_print('Computing the sparse ILU factorization of the Jacobian matrix')

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

        self._debug_print('Done computing the sparse ILU factorization of the Jacobian matrix')

    def _lu_solve(self, A, rhs):
        if A.lu.L.dtype != rhs.dtype and numpy.dtype(rhs.dtype.char.upper()) == rhs.dtype:
            x = rhs.copy()
            x.real = self._lu_solve(A, rhs.real)
            x.imag = self._lu_solve(A, rhs.imag)
        else:
            x = A.lu.solve(rhs)

        return x

    def direct_solve(self, jac, rhs, rhs2=None, V=None, W=None, C=None):
        '''Solve $J(u, p) y = x$ for $y$ using a direct solver. See
        the :meth:`solve` function.

        '''
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
            out, info, _ = gmres(jac, x, 5, 1e-8, prec=self._prec)
            if info == 0:
                return out

        if rhs2 is not None:
            x = numpy.append(rhs, rhs2 * self.border_scaling)

        # Use a direct solver instead
        self._compute_factorization(jac, V, W, C)

        if jac.bordered_lu:
            self._debug_print('Solving a bordered linear system')

            y = self._lu_solve(jac, x)

            border_size = 1
            if hasattr(rhs2, 'shape') and len(rhs2.shape) > 0:
                border_size = rhs2.shape[0]

            y1 = y[:-border_size]
            y2 = y[-border_size:] * self.border_scaling

            if numpy.isscalar(rhs2):
                y2 = y2.item()

            self._debug_print_residual('Done solving a bordered linear system with residual',
                                       jac, y1, rhs - V * y2)

            return y1, y2

        self._debug_print('Solving a linear system')

        y = self._lu_solve(jac, x)

        self._debug_print_residual('Done solving a linear system with residual', jac, y, rhs)

        return y

    def iterative_solve(self, jac, rhs, rhs2=None, V=None, W=None, C=None):
        '''Solve $J(u, p) y = x$ for $y$ using an iterative solver.
        See the :meth:`solve` function.

        '''
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

        self._compute_preconditioner(jac, A, V, W, C)

        self._debug_print('Solving a linear system')

        parameters = self.parameters.get('Iterative Solver', {})
        restart = parameters.get('Restart', 100)
        maxit = parameters.get('Maximum Iterations', 1000)
        tol = parameters.get('Convergence Tolerance', 1e-6)

        y, info, gmres_iterations = gmres(A, x, maxit, tol, restart=restart, prec=self._prec)
        if info != 0:
            Exception('GMRES did not converge')

        if jac.bordered_lu:
            border_size = 1
            if hasattr(rhs2, 'shape') and len(rhs2.shape) > 0:
                border_size = rhs2.shape[0]

            y1 = y[:-border_size]
            y2 = y[-border_size:] * self.border_scaling

            self._debug_print_residual('Done solving a bordered linear system in %d iterations with residual' %
                                       gmres_iterations, jac, y1, rhs - V * y2)

            return y1, y2

        self._debug_print_residual('Done solving a linear system in %d iterations with residual' %
                                   gmres_iterations, jac, y, rhs)

        return y

    def solve(self, jac, rhs, rhs2=None, V=None, W=None, C=None):
        if self.parameters.get('Use Iterative Solver', False):
            return self.iterative_solve(jac, rhs, rhs2, V, W, C)

        return self.direct_solve(jac, rhs, rhs2, V, W, C)

    def eigs(self, state, return_eigenvectors=False, enable_recycling=False):
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


def gmres(A, b, maxit, tol, restart=None, prec=None):
    '''Helper for running GMRES with different SciPy versions.

    :meta private:

    '''
    iterations = 0

    def callback(_r):
        nonlocal iterations
        iterations += 1

    if restart is None:
        restart = min(maxit, 100)

    maxiter = (maxit - 1) // restart + 1

    try:
        y, info = linalg.gmres(A, b, restart=restart, maxiter=maxiter,
                               rtol=tol, atol=0, M=prec,
                               callback=callback, callback_type='pr_norm')
    except TypeError:
        # Compatibility with SciPy <= 1.11
        y, info = linalg.gmres(A, b, restart=restart, maxiter=maxiter,
                               tol=tol, atol=0, M=prec,
                               callback=callback, callback_type='pr_norm')

    return y, info, iterations
