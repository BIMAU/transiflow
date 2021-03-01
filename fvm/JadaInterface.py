import numpy

from fvm import CrsMatrix

from jadapy import NumPyInterface

from scipy import sparse

class JadaOp:
    def __init__(self, mat):
        self.mat = sparse.csr_matrix((mat.coA, mat.jcoA, mat.begA), shape=(mat.n, mat.n))
        self.dtype = numpy.dtype('D')
        self.shape = (mat.n, mat.n)

    def matvec(self, x):
        return self.mat * x

    def __matmul__(self, x):
        return self.mat * x

class JadaPrecOp(object):
    def __init__(self, op, interface):
        self.op = op
        self.interface = interface

        self.dtype = self.op.dtype
        self.shape = self.op.shape

    def matvec(self, x):
        alpha = self.op.alpha
        beta = self.op.beta
        if len(alpha.shape) == 2:
            alpha = alpha[0, 0]
            beta = beta[0, 0]
        elif len(alpha.shape) == 1:
            alpha = alpha[0]
            beta = beta[0]

        rhs = x.copy()
        mat = beta * self.op.A.mat - alpha * self.op.B.mat
        crs_mat = CrsMatrix(mat.data, mat.indices, mat.indptr)
        return self.op.proj(self.interface.solve(crs_mat, rhs))

class JadaInterface(NumPyInterface.NumPyInterface):
    def __init__(self, interface, x, *args):
        super().__init__(*args)
        self.interface = interface
        self.jac = self.interface.jacobian(x)
        self.jac_op = JadaOp(self.jac)
        self.mass_op = JadaOp(self.interface.mass_matrix())

    # def solve(self, op, x, tol):
    #     out = x.copy()
    #     for i in range(x.shape[1]):
    #         out[:, i] , info = sparse.linalg.gmres(op, x[:, i], restart=100, tol=tol, atol=0,
    #                                                M=JadaPrecOp(op, self.interface))
    #         if info != 0:
    #             raise Exception('GMRES returned ' + str(info))
    #     return out

    def prec(self, x, alpha, beta):
        if len(alpha.shape) == 2:
            alpha = alpha[0, 0]
            beta = beta[0, 0]
        elif len(alpha.shape) == 1:
            alpha = alpha[0]
            beta = beta[0]

        rhs = x.copy()
        mat = beta * self.jac_op.mat - alpha * self.mass_op.mat
        crs_mat = CrsMatrix(mat.data, mat.indices, mat.indptr)
        return self.interface.solve(crs_mat, rhs)
