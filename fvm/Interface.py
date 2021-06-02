import numpy

from scipy import sparse
from scipy.sparse import linalg

from fvm import Discretization

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

        self.parameters = parameters

        # Select one pressure node to fix
        self.pressure_row = self.dim + (self.nx // min(8, self.nx) - 1) * self.dof \
            + (self.ny // min(8, self.ny) - 1) * self.nx * self.dof \
            + (self.nz // min(8, self.nz) - 1) * self.ny * self.nx * self.dof
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
            self._lu = None
            self._prec = None
            jac.lu = None

            coA = jac.coA
            jcoA = jac.jcoA
            begA = jac.begA

            # Fix one pressure node
            if self.dof > self.dim:
                coA = numpy.zeros(jac.begA[-1], dtype=jac.coA.dtype)
                jcoA = numpy.zeros(jac.begA[-1], dtype=int)
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
        elif rhs2 is not None and (not jac.lu or not jac.bordered_lu):
            self._lu = None
            self._prec = None
            jac.lu = None

            coA = numpy.zeros(jac.begA[-1] + 2 * jac.n + 1, dtype=jac.coA.dtype)
            jcoA = numpy.zeros(jac.begA[-1] + 2 * jac.n + 1, dtype=int)
            begA = numpy.zeros(len(jac.begA) + 1, dtype=int)

            idx = 0
            for i in range(jac.n):
                if i == self.pressure_row and self.dof > self.dim:
                    coA[idx] = -1.0
                    jcoA[idx] = i
                    idx += 1
                    begA[i+1] = idx
                    continue
                for j in range(jac.begA[i], jac.begA[i+1]):
                    if jac.jcoA[j] != self.pressure_row or not self.dof > self.dim:
                        coA[idx] = jac.coA[j]
                        jcoA[idx] = jac.jcoA[j]
                        idx += 1
                coA[idx] = V[i]
                jcoA[idx] = jac.n
                idx += 1

                begA[i+1] = idx

            for i in range(jac.n):
                coA[idx] = W[i]
                jcoA[idx] = i
                idx += 1

            coA[idx] = C
            jcoA[idx] = jac.n
            idx += 1

            begA[jac.n+1] = idx

            # Convert the matrix to CSC format since splu expects that
            A = sparse.csr_matrix((coA, jcoA, begA)).tocsc()

            jac.lu = linalg.splu(A)
            jac.bordered_lu = True

            # Cache the factorization for use in the iterative solver
            self._lu = jac.lu
            self._prec = linalg.LinearOperator((jac.n, jac.n), matvec=self._lu.solve, dtype=jac.dtype)

        if jac.bordered_lu:
            y = jac.solve(x)
            return y[:-1], y[-1]

        return jac.solve(x)

    def eigs(self, state, return_eigenvectors=False):
        '''Compute the generalized eigenvalues of beta * J(x) * v = alpha * M * v.'''

        from jadapy import jdqz, Target
        from fvm.JadaInterface import JadaOp, JadaInterface

        jac_op = JadaOp(self.jacobian(state))
        mass_op = JadaOp(self.mass_matrix())
        jada_interface = JadaInterface(self, jac_op, mass_op, jac_op.shape[0], numpy.complex128)

        parameters = self.parameters.get('Eigenvalue Solver', {})
        target = parameters.get('Target', Target.LargestRealPart)
        subspace_dimensions = [parameters.get('Minimum Subspace Dimension', 30),
                               parameters.get('Maximum Subspace Dimension', 60)]
        tol = parameters.get('Tolerance', 1e-7)
        num = parameters.get('Number of Eigenvalues', 5)

        result = jdqz.jdqz(jac_op, mass_op, num, tol=tol, subspace_dimensions=subspace_dimensions, target=target,
                           interface=jada_interface, arithmetic='complex', prec=jada_interface.shifted_prec,
                           return_eigenvectors=return_eigenvectors, return_subspaces=True,
                           initial_subspaces=self._subspaces)

        if return_eigenvectors:
            alpha, beta, v, q, z = result
            self._subspaces = [q, z]
            idx = range(len(alpha))
            idx = sorted(idx, key=lambda i: -(alpha[i] / beta[i]).real)

            w = v.copy()
            eigs = alpha.copy()
            for i in range(len(idx)):
                w[:, i] = v[:, idx[i]]
                eigs[i] = alpha[idx[i]] / beta[idx[i]]
            return eigs, w
        else:
            alpha, beta, q, z = result
            self._subspaces = [q, z]
            return numpy.array(sorted(alpha / beta, key=lambda x: -x.real))
