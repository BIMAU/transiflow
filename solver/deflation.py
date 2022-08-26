import scipy
import scipy.sparse.linalg as spla

def dpcg(A_d, b, x0=None, tol=None, maxiter=None, M=None):

    it = 0
    def count_iter(xj):
        nonlocal it
        it += 1
    xtil = A_d.applyQ(b)
    btil = b - A_d.applyA(xtil)
    xbar, flag = spla.cg(A_d, btil, x0, tol=tol, maxiter=maxiter, M=M, callback=count_iter)
    x=A_d.proj(xbar) + xtil
    return x, flag, it

def dgmres(A_d, b, x0=None, tol=None, maxiter=None, restart=None, M=None):

    it = 0
    def count_iter(xj):
        nonlocal it
        it += 1
    xtil = A_d.applyQ(b)
    btil = b - A_d.applyA(xtil)
    xbar, flag = spla.gmres(A_d, btil, x0, tol=tol, maxiter=maxiter, M=M, restart=restart, callback=count_iter)
    x=A_d.proj(xbar) + xtil
    return x, flag, it


class ProjectedOperator(spla.LinearOperator):

    def __init__(self, A, V):
        self.A = A
        self.V = V
        self.dtype = A.dtype
        self.shape=A.shape

    def _matvec(self, x):
        y1 = x - self.V @ (self.V.T @ x)
        y2 = self.A @ y1
        y3 = y2 - self.V @ (self.V.T @ y2)
        return y3

class DeflatedOperator(spla.LinearOperator):

    def __init__(self, A, V, V0=None):
        '''
        Construct deflated operator with given space V. V is checked for orthonormality on input.
        The operator is defined as

              op*x = (I-Q0)(I - AQ)A(I-QA)(I-Q0)x,
        with Q     = V (V'AV)\V'
             Q0    = V0*V0', with V0 an optinoal exact nullspace of A (if provided).
        
        '''
        if A.shape[0] != A.shape[1]:
            raise Exception('A must be square!')

        k = V.shape[1]
        ortho_err = spla.norm( (V.T @ V) - scipy.sparse.eye(k) )
        if ortho_err>1e-10:
            print('Warning: input V to DeflatedOperator is not orthonormal: ||V^TV-I||=%g'%(ortho_err))
        self.A = A
        self.V = V
        self.V0 = V0
        self.dtype = A.dtype
        self.shape = A.shape

        self.E = scipy.sparse.csc_matrix(self.V.T @ (self.A @ self.V))
        self.Elu = spla.splu(self.E)

    def applyA(self, x):
        '''
        apply the original matrix A to a vector (without deflation)
        '''
        return self.A @ x

    def applyQ(self, x):
        '''
        computes y = Q*x
        with Q = V(V'AV)\V'
        '''
        q = self.V.T @ x
        return self.V @ self.Elu.solve(q)

    def proj(self, x):
        '''
        computes y = (I - (QA+Q0))x
        with Q = V(V'AV)\V', Q0=V0*V0'
        '''
        if self.V0 is not None:
            y = x - self.V0 @ (self.V0.T @ x)
        else:
            y = x
        y = y - self.applyQ(self.A @ y)
        return y

    def projT(self, x):
        '''
        computes y = (I - (AQ+Q0))x
        with Q = V(V'AV)\V'
        '''
        y = x - self.A @ self.applyQ(x)
        if self.V0 is not None:
            y = y - self.V0 @ (self.V0.T @ y)
        return y

    def _matvec(self, x):
        '''
        apply deflated operator A,
          y = (I - AQ)A(I - QA)x,
          Q = V (V'AV)\(V'x)
        '''
        y = self.projT(self.A @ self.proj(x))
        return y
