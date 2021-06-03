import numpy

from scipy import sparse
from scipy.sparse import linalg

from fvm import Discretization


# see the number of gmres iterations
class gmres_counter(object):
    def __init__(self, disp=True):
        self._disp = disp
        self.niter = 0

    def __call__(self, rk=None):
        self.niter += 1
        if self._disp:
            print('iter %3i\trk = %s' % (self.niter, str(rk)))


class Interface:
    def __init__(self, parameters, nx, ny, nz, dim, dof):
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.dim = dim
        self.dof = dof
        self.discretization = Discretization(parameters, nx, ny, nz, dim, dof)

        self.parameters = parameters

        # Solver caching
        self._lu = None
        self._prec = None

        # Eigenvalue solver caching
        self._subspaces = None

    def set_parameter(self, name, value):
        self.discretization.set_parameter(name, value)

    def get_parameter(self, name):
        return self.discretization.get_parameter(name)

    def rhs(self, state):
        return self.discretization.rhs(state)

    def jacobian(self, state):
        return self.discretization.jacobian(state)

    def mass_matrix(self):
        return self.discretization.mass_matrix()

    # def solve(self, jac, rhs):
    #     coA = numpy.zeros(jac.begA[-1], dtype=jac.coA.dtype)
    #     jcoA = numpy.zeros(jac.begA[-1], dtype=int)
    #     begA = numpy.zeros(len(jac.begA), dtype=int)
    #
    #     idx = 0
    #     for i in range(len(jac.begA)-1):
    #         if i == self.dim:
    #             coA[idx] = -1.0
    #             jcoA[idx] = i
    #             idx += 1
    #             begA[i+1] = idx
    #             continue
    #         for j in range(jac.begA[i], jac.begA[i+1]):
    #             if jac.jcoA[j] != self.dim:
    #                 coA[idx] = jac.coA[j]
    #                 jcoA[idx] = jac.jcoA[j]
    #                 idx += 1
    #         begA[i+1] = idx
    #
    #     A = sparse.csr_matrix((coA, jcoA, begA))
    #     if len(rhs.shape) < 2:
    #         rhs[self.dim] = 0
    #         x = linalg.spsolve(A, rhs)
    #     else:
    #         x = rhs.copy()
    #         #rhs[self.dim, :] = 0
    #         for i in range(x.shape[1]):
    #             x[:, i] = linalg.spsolve(A, rhs[:, i])
    #     return x

    # TODO wei
    def solve(self, jac, x):
        rhs = x.copy()

        # Fix one pressure node
        if self.dof > self.dim:
            if len(rhs.shape) < 2:
                rhs[self.dim] = 0
            else:
                rhs[self.dim, :] = 0

        # Use a direct solver instead
        if True:
            coA = jac.coA
            jcoA = jac.jcoA
            begA = jac.begA

            # Fix one pressure node
            if self.dof > self.dim:
                coA = numpy.zeros(jac.begA[-1], dtype=jac.coA.dtype)
                jcoA = numpy.zeros(jac.begA[-1], dtype=int)
                begA = numpy.zeros(len(jac.begA), dtype=int)

                idx = 0
                for i in range(len(jac.begA) - 1):
                    if i == self.dim:
                        coA[idx] = -1.0
                        jcoA[idx] = i
                        idx += 1
                        begA[i + 1] = idx
                        continue
                    for j in range(jac.begA[i], jac.begA[i + 1]):
                        if jac.jcoA[j] != self.dim:
                            coA[idx] = jac.coA[j]
                            jcoA[idx] = jac.jcoA[j]
                            idx += 1
                    begA[i + 1] = idx

            # Convert the matrix to CSC format since splu expects that
            A = sparse.csr_matrix((coA, jcoA, begA)).tocsc()

            # jac.lu = linalg.splu(A)

            if self.parameters.get('Use Iterative Solver', False):
                if self.parameters.get('Use Preconditioner', False):
                    if self.parameters.get('Use ILU Preconditioner', False):
                        self._prec = linalg.LinearOperator((jac.n, jac.n), matvec=linalg.spilu(A).solve, dtype=jac.dtype)

                    if self._prec and jac.dtype == rhs.dtype and jac.dtype == self._prec.dtype:
                        out, info = linalg.gmres(A, rhs, M=self._prec, callback=gmres_counter())
                        if info == 0:
                            return out
                else:
                    out, info = linalg.gmres(A, rhs, callback=gmres_counter())
                    if info == 0:
                        return out
            A_lu = linalg.splu(A)

        return A_lu.solve(rhs)

    def solve_bordered(self, jac, fval, dfval, r_x, r_mu, r):
        rhs = fval.copy()

        # Fix one pressure node
        if self.dof > self.dim:
            if len(rhs.shape) < 2:
                rhs[self.dim] = 0
            else:
                rhs[self.dim, :] = 0

        # Use a direct solver instead
        if True:
            coA = jac.coA
            jcoA = jac.jcoA
            begA = jac.begA

            # Fix one pressure node
            if self.dof > self.dim:
                coA = numpy.zeros(jac.begA[-1], dtype=jac.coA.dtype)
                jcoA = numpy.zeros(jac.begA[-1], dtype=int)
                begA = numpy.zeros(len(jac.begA), dtype=int)

                idx = 0
                for i in range(len(jac.begA) - 1):
                    if i == self.dim:
                        coA[idx] = -1.0
                        jcoA[idx] = i
                        idx += 1
                        begA[i + 1] = idx
                        continue
                    for j in range(jac.begA[i], jac.begA[i + 1]):
                        if jac.jcoA[j] != self.dim:
                            coA[idx] = jac.coA[j]
                            jcoA[idx] = jac.jcoA[j]
                            idx += 1
                    begA[i + 1] = idx

            # Convert the matrix to CSC format since splu expects that
            A = sparse.csr_matrix((coA, jcoA, begA)).tocsc().toarray()

            a = numpy.concatenate((A, dfval[:, numpy.newaxis]), axis=1)
            b = numpy.append(r_x, r_mu)
            A = numpy.concatenate((a, b[:, numpy.newaxis].T), axis=0)

            b = numpy.append(-fval, r)

            A_sparse = sparse.csc_matrix(A)
            A_lu = linalg.splu(A_sparse)
            # jac.lu = linalg.splu(A)

            if self.parameters.get('Use Iterative Solver', False):
                if self.parameters.get('Use Preconditioner', False):
                    if self.parameters.get('Use LU Preconditioner', False):
                        self._prec = linalg.LinearOperator((jac.n + 1, jac.n + 1), matvec=linalg.splu(A).solve, dtype=jac.dtype)
                    elif self.parameters.get('Use ILU Preconditioner', False):
                        self._prec = linalg.LinearOperator((jac.n + 1, jac.n + 1), matvec=linalg.spilu(A).solve,
                                                           dtype=jac.dtype)

                    if self._prec and jac.dtype == rhs.dtype and jac.dtype == self._prec.dtype:
                        out, info = linalg.gmres(A_sparse, b, M=self._prec, callback=gmres_counter())
                        if info == 0:
                            return out
                else:
                    out, info = linalg.gmres(A_sparse, b, callback=gmres_counter())
                    if info == 0:
                        return out

        return A_lu.solve(b)
    #
    # def solve(self, jac, x):
    #     rhs = x.copy()
    #
    #     # Fix one pressure node
    #     if self.dof > self.dim:
    #         if len(rhs.shape) < 2:
    #             rhs[self.dim] = 0
    #         else:
    #             rhs[self.dim, :] = 0
    #
    #     # First try to use an iterative solver with the previous
    #     # direct solver as preconditioner
    #     if self._prec and jac.dtype == rhs.dtype and jac.dtype == self._prec.dtype and \
    #             self.parameters.get('Use Iterative Solver', False):
    #         out, info = linalg.gmres(jac, rhs, restart=5, maxiter=1, tol=1e-8, atol=0, M=self._prec)
    #         if info == 0:
    #             return out
    #
    #     # no preconditioner
    #     # if self._prec and jac.dtype == rhs.dtype and jac.dtype == self._prec.dtype and \
    #     #         self.parameters.get('Use Iterative Solver', False):
    #     #     out, info = linalg.gmres(jac, rhs, restart=5, maxiter=1, tol=1e-8, atol=0)
    #     #     if info == 0:
    #     #         return out
    #
    #     # Use a direct solver instead
    #     if not jac.lu:
    #         coA = jac.coA
    #         jcoA = jac.jcoA
    #         begA = jac.begA
    #
    #         # Fix one pressure node
    #         if self.dof > self.dim:
    #             coA = numpy.zeros(jac.begA[-1], dtype=jac.coA.dtype)
    #             jcoA = numpy.zeros(jac.begA[-1], dtype=int)
    #             begA = numpy.zeros(len(jac.begA), dtype=int)
    #
    #             idx = 0
    #             for i in range(len(jac.begA) - 1):
    #                 if i == self.dim:
    #                     coA[idx] = -1.0
    #                     jcoA[idx] = i
    #                     idx += 1
    #                     begA[i + 1] = idx
    #                     continue
    #                 for j in range(jac.begA[i], jac.begA[i + 1]):
    #                     if jac.jcoA[j] != self.dim:
    #                         coA[idx] = jac.coA[j]
    #                         jcoA[idx] = jac.jcoA[j]
    #                         idx += 1
    #                 begA[i + 1] = idx
    #
    #         # Convert the matrix to CSC format since splu expects that
    #         A = sparse.csr_matrix((coA, jcoA, begA)).tocsc()
    #
    #         jac.lu = linalg.splu(A)
    #
    #         # Cache the factorization for use in the iterative solver
    #         self._lu = jac.lu
    #         # LU decomposition as the preconditioner
    #         self._prec = linalg.LinearOperator((jac.n, jac.n), matvec=self._lu.solve, dtype=jac.dtype)
    #
    #         # ILU decomposition as the preconditioner
    #         # self._prec = linalg.LinearOperator((jac.n, jac.n), matvec=linalg.spilu(A).solve, dtype=jac.dtype)
    #
    #     return jac.solve(rhs)

    def eigs(self, state, return_eigenvectors=False):
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
