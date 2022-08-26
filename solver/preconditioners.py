import numpy
import scipy
import sys

import scipy.sparse.linalg as spla
from solver_util import StokesDD, partition_of_unity
from deflation import DeflatedOperator

class AdditiveSchwarz(spla.LinearOperator):

    def num_subdomains(self):
        return len(self.index_lists)
    def indices(self, sd):
        return self.index_lists[sd]

    def __init__(self, A, arg1, arg2=None):
        '''
        There are two flavors of this constructor:

        1. AdditiveSchwarz(A, index_lists)
        2. AdditiveSchwarz(A, 

        ========================================================
        Variant 1 constructs Additive Schwarz
        preconditioner with given index lists:

        P = diag(A(i[0],i[0]), A(i[1],i[1]), ...)
        Index lists i (sd_indices) may be overlapping.

        For example,

        AdditiveSchwarz(A, [[0,1,2],[2,3,4],[5,6],7])

        will create two overlapping 3x3 blocks and a separate 
        2x2 and 1x1 block:

            |a11 a12 a13|
            |a21 a22 a23|_______
            |a31 a32|a33|a34 a35|
                    |a43 a44 a45|
                    |a53 a54 a55|______
                    ------------|a66 a67|
                                |a76 a77|
                                 ------- a88

        ========================================================
        Variant 2 construct block Jacobi preconditioner with
        given block size. If dof>1, the block size is multiplied
        by dof (that is, each block will contain block_size grid
        points/cells with multiple degrees of freedom each).
        '''

        def create_lists(A, block_size, dof):

            if A.shape[0] % (block_size*dof):
                raise Exception('size of A must be a multiple of the block_size')

            sd_size = block_size*dof
            num_sd = int(A.shape[0] / sd_size)
            index_lists = []
            for sd in range(num_sd):
                index_lists += [range(sd*sd_size, min((sd+1)*sd_size, A.shape[0]))]
            return index_lists


        if type(arg1) is list:
            self.index_lists = arg1
        elif type(arg1) is int and arg2 is not None:
            self.index_lists = create_lists(A, arg1, arg2)
        else:
            raise Exception('invalid use of constructor.')

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



def build_stokes_preconditioner(A, nx, ny, sx, sy, V0=None):
    '''
    M, A_d = build_stokes_preconditioner(nx,ny,sx,sy)

    will create an Additive Schwarz preconditioner M with minimal overlap,
    and a GDSW-type deflated operator A_d to be used with solvers like
    deflation.dpcg or deflation.dgmres
    '''
    N = A.shape[0]
    if A.shape[1] != N or N != nx*ny*3:
        raise Exception('A must be square and have nx*ny*3 rows/columns')

    # helper object to get index lists:
    DD = StokesDD(nx,ny,sx,sy)
    idx0 = [] # overlapping subdomains

    valV  = numpy.zeros(N)
    rowsV = numpy.array(range(N),dtype='int')
    colsV = numpy.zeros(N,dtype='int')

    for sd in range(DD.num_subdomains()):
        i0, i1, i2, i3 = DD.indices(sd)

        # this list will be passed to AdditiveSchwarz to define
        # the overlapping subdomains
        idx0 += [i0]

        A11 = A[i1,:][:,i1]
        # Replace the lat row by a sum(p)=0 equation.
        # This row is a pressure equation in the corner
        # of a subdomain and therefore only connects to
        # velocities ('full conservation cell')
        n1=A11.shape[1]
        if A11[n1-1,:].nnz == 0:
            A11[n1-1,range(2,n1,3)]=1
        try:
            LU11 = spla.splu(A11, permc_spec='NATURAL', diag_pivot_thresh=0)
        except:
            print('interior matrix %d is singular, going into debug mode'%(sd))
            breakpoint()
            raise Exception('singular interior matrix A11 encountered')

        valV[i1] = 0
        colV[i1] = sd

        for sep_nodes in i2:
            l = len(sep_nodes)
            sep_id = DD.get_group_id(sep_nodes)
            A12 = A[i1,:][:,sep_nodes]
            e2=numpy.ones(l)/sqrt(l)
            valV[sep_nodes] = e2
            colV[sep_nodes] = sep_id
            e1 = LU11.solve(A12 @ e2)
            valV[i1] += e1


    # minimally overlapping Additive Schwarz preconditioner
    M_as=AdditiveSchwarz(A, idx0)

    V = scipy.sparse.csc_matrix( (valV, (rowV, colV)), shape=[N,k])

    A_d=DeflatedOperator(A, V, V0)
    return M, A_d
