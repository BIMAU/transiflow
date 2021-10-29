import numpy

from scipy import sparse
from scipy.sparse import linalg

from fvm import Discretization, CylindricalDiscretization

class Interface:
    '''This class defines an interface to the NumPy backend for the
    discretization. We use this so we can write higher level methods
    such as pseudo-arclength continuation without knowing anything
    about the underlying methods such as the solvers that are present
    in the backend we are interfacing with.'''

    def __init__(self, parameters, nx, ny, nz, dim, dof, x=None, y=None, z=None):
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.dim = dim
        self.dof = dof

        self.discretization = Discretization(parameters, nx, ny, nz, dim, dof, x, y, z)

        if 'Problem Type' in parameters and 'Taylor-Couette' in parameters['Problem Type']:
            self.discretization = CylindricalDiscretization(parameters, nx, ny, nz, dim, dof, x, y, z)

        self.parameters = parameters

        # Select one pressure node to fix
        self.pressure_row = self.dim
        if self.parameters.get('Verbose', False):
            print('Fixing pressure at row %d' % self.pressure_row)

        # Solver caching
        self._lu = None
        self._prec = None

        # Eigenvalue solver caching
        self._subspaces = None

    def set_parameter(self, name, value):
        '''Set a parameter in self.parameters while also letting the
        discretization know that we changed a parameter. '''
        self.discretization.set_parameter(name, value)

    def get_parameter(self, name):
        '''Get a parameter from self.parameters through the discretization.'''
        return self.discretization.get_parameter(name)

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
            coA = numpy.zeros(jac.begA[-1] + 1, dtype=jac.coA.dtype)
            jcoA = numpy.zeros(jac.begA[-1] + 1, dtype=int)
            begA = numpy.zeros(len(jac.begA), dtype=int)

            idx = 0
            for i in range(len(jac.begA)-1):
                if i == self.pressure_row:
                    coA[idx] = -1.0
                    jcoA[idx] = i
                    idx += 1
                    begA[i+1] = idx
                    continue
                for j in range(jac.begA[i], jac.begA[i+1]):
                    if jac.jcoA[j] != self.pressure_row:
                        coA[idx] = jac.coA[j]
                        jcoA[idx] = jac.jcoA[j]
                        idx += 1

                begA[i+1] = idx

        # Convert the matrix to CSC format since splu expects that
        A = sparse.csr_matrix((coA, jcoA, begA)).tocsc()

        jac.lu = linalg.splu(A)
        jac.bordered_lu = False

        # Cache the factorization for use in the iterative solver
        self._lu = jac.lu
        self._prec = linalg.LinearOperator((jac.n, jac.n), matvec=self._lu.solve, dtype=jac.dtype)

    def _compute_bordered_factorization(self, jac, V, W, C):
        '''Compute the LU factorization of the bordered jacobian.'''

        def _get_value(V, i, j):
            if not hasattr(V, 'shape') or len(V.shape) < 1:
                return V

            if len(V.shape) < 2:
                return V[i]

            return V[i, j]

        self._lu = None
        self._prec = None
        jac.lu = None

        if V is None:
            raise Exception('V is None')

        border_size = 1
        if len(V.shape) > 1:
            border_size = V.shape[1]

        extra_border_space = jac.n * border_size * 2 + border_size * border_size

        if W is None:
            W = V

        if C is None:
            C = numpy.zeros((border_size, border_size), dtype=jac.coA.dtype)

        coA = numpy.zeros(jac.begA[-1] + extra_border_space + 1, dtype=jac.coA.dtype)
        jcoA = numpy.zeros(jac.begA[-1] + extra_border_space + 1, dtype=int)
        begA = numpy.zeros(len(jac.begA) + border_size, dtype=int)

        fix_pressure_row = self.dof > self.dim and self.pressure_row is not None

        idx = 0
        for i in range(jac.n):
            if fix_pressure_row and i == self.pressure_row:
                coA[idx] = -1.0
                jcoA[idx] = i
                idx += 1
                begA[i+1] = idx
                continue

            for j in range(jac.begA[i], jac.begA[i+1]):
                if not fix_pressure_row or jac.jcoA[j] != self.pressure_row:
                    coA[idx] = jac.coA[j]
                    jcoA[idx] = jac.jcoA[j]
                    idx += 1

            for j in range(border_size):
                coA[idx] = _get_value(V, i, j)
                jcoA[idx] = jac.n + j
                idx += 1

            begA[i+1] = idx

        for i in range(border_size):
            for j in range(jac.n):
                coA[idx] = _get_value(W, j, i)
                jcoA[idx] = j
                idx += 1

            for j in range(border_size):
                coA[idx] = _get_value(C, i, j)
                jcoA[idx] = jac.n + j
                idx += 1

            begA[jac.n+1+i] = idx

        # Convert the matrix to CSC format since splu expects that
        A = sparse.csr_matrix((coA, jcoA, begA)).tocsc()

        jac.lu = linalg.splu(A)
        jac.bordered_lu = True

        # Cache the factorization for use in the iterative solver
        self._lu = jac.lu
        self._prec = linalg.LinearOperator((jac.n, jac.n), matvec=self._lu.solve, dtype=jac.dtype)

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
            out, info = linalg.gmres(jac, x, restart=5, maxiter=1, tol=1e-8, atol=0, M=self._prec)
            if info == 0:
                return out

        if rhs2 is not None:
            x = numpy.append(rhs, rhs2)

        # Use a direct solver instead
        if rhs2 is None and (not jac.lu or jac.bordered_lu):
            self._compute_factorization(jac)
        elif rhs2 is not None and (not jac.lu or not jac.bordered_lu):
            self._compute_bordered_factorization(jac, V, W, C)

        if jac.bordered_lu:
            y = jac.solve(x)

            border_size = 1
            if hasattr(rhs2, 'shape') and len(rhs2.shape) > 0:
                border_size = rhs2.shape[0]

            return y[:-border_size], y[-border_size:]

        return jac.solve(x)

    def _eigs(self, jada_interface, jac_op, mass_op, prec_op, state, return_eigenvectors):
        '''Internal helper for eigs()'''

        from jadapy import jdqz, orthogonalization

        parameters = self.parameters.get('Eigenvalue Solver', {})
        arithmetic = parameters.get('Arithmetic', 'complex')
        target = parameters.get('Target', 0.0)
        initial_subspace_dimension = parameters.get('Initial Subspace Dimension', 0)
        subspace_dimensions = [parameters.get('Minimum Subspace Dimension', 30),
                               parameters.get('Maximum Subspace Dimension', 60)]
        tol = parameters.get('Tolerance', 1e-7)
        num = parameters.get('Number of Eigenvalues', 5)

        if not self._subspaces and initial_subspace_dimension > 0:
            # Use an inverse iteration to find guesses
            # for the eigenvectors closest to the target
            V = jada_interface.vector(initial_subspace_dimension)
            V[:, 0] = jada_interface.random()
            orthogonalization.normalize(V[:, 0])

            for i in range(1, initial_subspace_dimension):
                V[:, i] = jada_interface.prec(V[:, i-1])
                orthogonalization.orthonormalize(V[:, 0:i], V[:, i])

            self._subspaces = [V]

        result = jdqz.jdqz(jac_op, mass_op, num, tol=tol, subspace_dimensions=subspace_dimensions, target=target,
                           interface=jada_interface, arithmetic=arithmetic, prec=prec_op,
                           return_eigenvectors=return_eigenvectors, return_subspaces=True,
                           initial_subspaces=self._subspaces)

        if return_eigenvectors:
            alpha, beta, v, q, z = result
            self._subspaces = [q, z]
            idx = range(len(alpha))
            idx = sorted(idx, key=lambda i: -(alpha[i] / beta[i]).real if (alpha[i] / beta[i]).real < 100 else 100)

            w = v.copy()
            eigs = alpha.copy()
            for i in range(len(idx)):
                w[:, i] = v[:, idx[i]]
                eigs[i] = alpha[idx[i]] / beta[idx[i]]
            return eigs, w
        else:
            alpha, beta, q, z = result
            self._subspaces = [q, z]
            return numpy.array(sorted(alpha / beta, key=lambda x: -x.real if x.real < 100 else 100))

    def eigs(self, state, return_eigenvectors=False):
        '''Compute the generalized eigenvalues of beta * J(x) * v = alpha * M * v.'''

        from fvm.JadaInterface import JadaOp, JadaInterface

        parameters = self.parameters.get('Eigenvalue Solver', {})
        arithmetic = parameters.get('Arithmetic', 'complex')

        jac_op = JadaOp(self.jacobian(state))
        mass_op = JadaOp(self.mass_matrix())
        jada_interface = JadaInterface(self, jac_op, mass_op, jac_op.shape[0], numpy.complex128)
        if arithmetic == 'real':
            jada_interface = JadaInterface(self, jac_op, mass_op, jac_op.shape[0])

        return self._eigs(jada_interface, jac_op, mass_op, jada_interface.shifted_prec,
                          state, return_eigenvectors)
