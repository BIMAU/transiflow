from transiflow import CrsMatrix

import time
import numpy
import warnings

from jadapy import NumPyInterface
from jadapy.orthogonalization import orthogonalize

from scipy import sparse

def _get_scalars(alpha, beta):
    try:
        if len(alpha.shape) == 2:
            alpha = alpha[0, 0]
            beta = beta[0, 0]
        elif len(alpha.shape) == 1:
            alpha = alpha[0]
            beta = beta[0]
    except AttributeError:
        pass

    return alpha, beta

class Op:
    def __init__(self, mat):
        self.fvm_mat = mat
        self.mat = sparse.csr_matrix((mat.coA, mat.jcoA, mat.begA), shape=(mat.n, mat.n))
        self.dtype = mat.coA.dtype
        self.shape = (mat.n, mat.n)

    def matvec(self, x):
        return self.mat * x

    def __matmul__(self, x):
        return self.mat * x

class CachedMatrix:
    def __init__(self, matrix, alpha, beta):
        self.matrix = matrix
        self.alpha = alpha
        self.beta = beta
        self.last_used = time.time()

    def same_shifts(self, alpha, beta):
        eps = 1e-10
        eps2 = 1e-1

        if abs(alpha) < eps:
            if abs(alpha - self.alpha) < eps2 and abs(beta - self.beta) < eps2:
                return True
        elif abs(beta) > eps and abs(self.beta) > eps:
            if abs(alpha / beta - self.alpha / self.beta) / abs(alpha / beta) < eps2:
                return True

        return False

    def same_shapes(self, matrix):
        return matrix is None or self.matrix.shape == matrix.shape

    def get_matrix(self):
        self.last_used = time.time()
        return self.matrix

class MatrixCache:
    def __init__(self, jac_op, mass_op):
        self.jac_op = jac_op
        self.mass_op = mass_op

        self.matrices = []
        self.max_matrices = 1

    def get_shifted_matrix(self, alpha, beta, shifted_matrix=None):
        alpha, beta = _get_scalars(alpha, beta)

        if shifted_matrix is None and alpha == 0.0 and beta == 1.0:
            return self.jac_op.fvm_mat

        # Cache previous preconditioners
        for i, cached_matrix in enumerate(self.matrices):
            if cached_matrix.same_shifts(alpha, beta) and cached_matrix.same_shapes(shifted_matrix):
                return cached_matrix.get_matrix()

        # Remove the cached preconditioner that was not used last
        if len(self.matrices) >= self.max_matrices:
            if len(self.matrices) > 1 and self.matrices[0].last_used > self.matrices[1].last_used:
                self.matrices.pop(1)
            elif len(self.matrices) > 0:
                self.matrices.pop(0)

        if shifted_matrix is None:
            mat = beta * self.jac_op.mat - alpha * self.mass_op.mat
            shifted_matrix = CrsMatrix(mat.data, mat.indices, mat.indptr, False)

        if self.max_matrices > 0:
            self.matrices.append(CachedMatrix(shifted_matrix, alpha, beta))

        return shifted_matrix

class PrecOp(object):
    def __init__(self, op, prec_op):
        self.op = op
        self.prec_op = prec_op

        self.dtype = self.op.dtype
        self.shape = self.op.shape

    def matvec(self, x):
        return self.op.proj(self.prec_op(x, self.op.alpha, self.op.beta))

class Interface(NumPyInterface.NumPyInterface):
    def __init__(self, interface, jac_op, mass_op, *args, **kwargs):
        super().__init__(*args)
        self.interface = interface
        self.jac_op = jac_op

        self.preconditioned_solve = kwargs.get('preconditioned_solve', False)
        self.shifted = kwargs.get('shifted', False)

        self._matrix_cache = MatrixCache(jac_op, mass_op)

    def solve(self, op, x, tol, maxit):
        if op.dtype.char != op.dtype.char.upper():
            # Real case
            if abs(op.alpha.real) < abs(op.alpha.imag):
                op.alpha = op.alpha.imag
            else:
                op.alpha = op.alpha.real
            op.beta = op.beta.real

        prec_op = None
        if self.preconditioned_solve:
            if self.shifted:
                prec_op = PrecOp(op, self.shifted_prec)
            else:
                prec_op = PrecOp(op, self.prec)

        out = x.copy()
        for i in range(x.shape[1]):

            restart = min(maxit, 100)
            maxiter = (maxit - 1) // restart + 1
            out[:, i], info = sparse.linalg.gmres(op, x[:, i], restart=restart, maxiter=maxiter, tol=tol, atol=0, M=prec_op)
            if info < 0:
                raise Exception('GMRES returned ' + str(info))
            elif info > 0:
                warnings.warn('GMRES did not converge in ' + str(info) + ' iterations')
        return out

    def prec(self, x, *args):
        return self.interface.solve(self.jac_op.fvm_mat, x)

    def shifted_prec(self, x, alpha, beta):
        shifted_matrix = self._matrix_cache.get_shifted_matrix(alpha, beta)
        return self.interface.solve(shifted_matrix, x)

class BorderedPrecOp(object):
    def __init__(self, interface, prec):
        self.interface = interface
        self.prec = prec

        self.dtype = prec.dtype
        self.shape = prec.shape

    def matvec(self, x):
        return self.interface.solve(self.prec, x)

class BorderedInterface(NumPyInterface.NumPyInterface):
    def __init__(self, interface, jac_op, mass_op, *args, **kwargs):
        super().__init__(*args)
        self.interface = interface
        self.jac_op = jac_op
        self.mass_op = mass_op

        self._matrix_cache = MatrixCache(jac_op, mass_op)

    def solve(self, op, x, tol, maxit):
        alpha = op.alpha
        beta = op.beta

        if op.dtype.char != op.dtype.char.upper():
            # Real case
            if abs(op.alpha.real) < abs(op.alpha.imag):
                alpha = op.alpha.imag
            else:
                alpha = op.alpha.real
            beta = op.beta.real

        alpha, beta = _get_scalars(alpha, beta)

        mat = beta * self.jac_op.mat - alpha * self.mass_op.mat
        shifted_matrix = CrsMatrix(mat.data, mat.indices, mat.indptr, False)
        shifted_bordered_matrix = self.interface.compute_bordered_matrix(shifted_matrix, op.Z, op.Q)
        shifted_bordered_op = Op(shifted_bordered_matrix)

        prec = self._matrix_cache.get_shifted_matrix(alpha, beta, shifted_bordered_matrix)
        prec_op = BorderedPrecOp(self.interface, prec)

        out = x.copy()
        for i in range(x.shape[1]):
            restart = min(maxit, 100)
            maxiter = (maxit - 1) // restart + 1
            x2 = numpy.append(x[:, i], numpy.zeros(op.Q.shape[1], x.dtype))
            y, info = sparse.linalg.gmres(shifted_bordered_op, x2, restart=restart,
                                          maxiter=maxiter, tol=tol, atol=0, M=prec_op)
            out[:, i] = y[:-op.Q.shape[1]]

            if info < 0:
                raise Exception('GMRES returned ' + str(info))
            elif info > 0:
                warnings.warn('GMRES did not converge in ' + str(info) + ' iterations')

        orthogonalize(op.Q, out)

        return out

    def prec(self, x, *args):
        return self.interface.solve(self.jac_op.fvm_mat, x)
