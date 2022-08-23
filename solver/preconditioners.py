import numpy
import scipy

import scipy.sparse.linalg as spla

class AdditiveSchwarz(spla.LinearOperator):

    def num_subdomains(self):
        return len(self.index_lists)
    def indices(self, sd):
        return self.index_lists[sd]

    def __init__(self, A, index_lists):
        '''
        Constructs Additive Schwarz preconditioner
        with given index lists:

        P = diag(A(i[0],i[0]), A(i[1],i[1]), ...)
        Index lists i (sd_indices) may be overlapping.

        For example,

        AdditiveSchwarz(A, [[0,1,2],[2,3,4],[5,6],7])

        will create two overlapping 3x3 blocks and a separate 2x2 and 1x1 block:

            |a11 a12 a13|
            |a21 a22 a23|_______
            |a31 a32|a33|a34 a35|
                    |a43 a44 a45|
                    |a53 a54 a55|______
                    ------------|a66 a67|
                                |a76 a77|
                                 ------- a88
        '''
        self.index_lists = index_lists
        self.factor(A)

    def __init__(self, A, block_size, dof):
        '''
        Construct block Jacobi preconditioner with given block size.
        If dof>1, the block size is multiplied by dof (that is, each
        block will contain block_size grid points/cells with multiple
        degrees of freedom each).
        We use an LU factorization for the blocks because scipy doesn't
        have Cholesky.
        '''
        if A.shape[0] % (block_size*dof):
            raise Exception('size of A must be a multiple of the block_size')

        sd_size = block_size*dof
        num_sd = int(A.shape[0] / sd_size)
        self.index_lists = []
        for sd in range(num_sd):
            self.index_lists += [range(sd*sd_size, min((sd+1)*sd_size, A.shape[0]))]
        self.factor(A)

    def factor(self, A):
        '''
        Factor the block diagonal of A as determined
        by the index sets passed to the constructor,
        to prepare the preconditioner for application.

        This function can be called repeatedly if the
        values of A change (but not the partitioning/
        block structure)
        '''
        if A.shape[0] != A.shape[1]:
            raise Exception('A must be square!')
        self.shape = A.shape
        self.LU = []

        for i in range(self.num_subdomains()):
            idx = self.indices(i)
            self.LU.append( spla.splu(scipy.sparse.csc_matrix(A[idx,:][:,idx])))

    def solve(self,x):
        '''
        Apply inverse operator to solve a linear system with the block diagonal of A
        '''
        y = numpy.zeros(x.shape)
        for i in range(self.num_subdomains()):
            idx = self.indices(i)
            y[idx]=self.LU[i].solve(x[idx])
        return y

    def _matvec(self, x):
        return self.solve(x)

