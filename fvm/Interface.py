import numpy

from scipy import sparse
from scipy.sparse import linalg

from fvm import Discretization

class Interface:
    def __init__(self, parameters, nx, ny, nz, dim, dof, x=None, y=None, z=None):
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.dim = dim
        self.dof = dof
        self.discretization = Discretization(parameters, nx, ny, nz, dim, dof, x, y, z)

        self.parameters = parameters
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

    def solve(self, jac, x):
        if not jac.lu:
            coA = numpy.zeros(jac.begA[-1], dtype=jac.coA.dtype)
            jcoA = numpy.zeros(jac.begA[-1], dtype=int)
            begA = numpy.zeros(len(jac.begA), dtype=int)

            idx = 0
            for i in range(len(jac.begA)-1):
                if i == self.dim:
                    coA[idx] = -1.0
                    jcoA[idx] = i
                    idx += 1
                    begA[i+1] = idx
                    continue
                for j in range(jac.begA[i], jac.begA[i+1]):
                    if jac.jcoA[j] != self.dim:
                        coA[idx] = jac.coA[j]
                        jcoA[idx] = jac.jcoA[j]
                        idx += 1
                begA[i+1] = idx

            # Convert the matrix to CSC format since splu expects that
            A = sparse.csr_matrix((coA, jcoA, begA)).tocsc()

            jac.lu = linalg.splu(A)

        rhs = x.copy()
        if len(rhs.shape) < 2:
            rhs[self.dim] = 0
        else:
            rhs[self.dim, :] = 0

        return jac.solve(rhs)

    def eigs(self, state, return_eigenvectors=False):
        from jadapy import jdqz
        from fvm.JadaInterface import JadaOp, JadaInterface

        jac_op = JadaOp(self.jacobian(state))
        mass_op = JadaOp(self.mass_matrix())
        jada_interface = JadaInterface(self, jac_op, mass_op, jac_op.shape[0], numpy.complex128)

        parameters = self.parameters.get('Eigenvalue Solver', {})
        target = parameters.get('Target', 0)
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
