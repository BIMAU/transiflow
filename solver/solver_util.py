import numpy
import scipy
import pymorton

import scipy.sparse.linalg as spla
from math import sqrt

def get_z_ordering(nx, ny, nz=1, dof=1):
    ''' reorder by a Morton (z-)curve to get subdomains with contguous indices:

    1--2  5--6  16...
      /  /  /   /
     /  /  /   |
    3--4  7--8 |
     _______/  |
    /          |
    9--*  *--* |
      /  /  /  |
     /  /  /  /
    *--*  *--*

    Returns z_idx and z_inv such that if x is in cartesian ordering,
    y=x(z_idx) is in z-ordering, and x=y(z_inv)

    This function works in 3D (octree) or 2D (nz=1, quadtree), and for
    problems with multiple degrees of freedom per grid point (dof>1).
    In that case, the dofs per grid point are kept together.
    '''

    z_idx = numpy.zeros(nx*ny*nz*dof)
    z_inv = numpy.zeros(nx*ny*nz*dof)

    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                c_id = (k*ny+j)*nx+i
                if nz == 1:
                    z_id = pymorton.interleave2(i,j)
                else:
                    z_id = pymorton.interleave3(i,j,k)

                for var in range(dof):
                    z_idx[z_id*dof+var] = c_id*dof+var
                    z_inv[c_id*dof+var] = z_id*dof+var

    return z_idx, z_inv

def get_separator_groups(nx,sx,ny,sy,nz=1,sz=1,dof=1):
    '''
    Returns a sparse matrix with N=nx*ny*nz*dof rows
    and m<<N columns. Each column index represents a
    'separator group'
    '''
    raise Exception('not implemented')

def get_subdomain_groups(dims, sd_dims, dof):
    '''
    Returns an orthonormal sparse matrix V with, for block size=prod(sd_dims),
    one column ber contiguous block with row entries 1 inside that block and 0 outside.

        V = [1
             1
             :
             1
             0 1
             0 1
             : :
               1
               0 1
               0 1
               : :
                 1 ...]

    If dof>1, a separate group (column) is inserted for each variable in a block,
    and block size refers to grid points/cells (with multiple dof per point/cell).
    E.g., for dof=3, bolock_size=4:

        V = [1 0 0
             0 1 0
             0 0 1
             1 0 0
             0 1 0
             0 0 1
             : : :
             1 0 0
             0 1 0
             0 0 1
                  1 0 0
                  0 1 0
                  0 0 1
                  1 0 0
                  0 1 0
                  0 0 1
                  : : :
                  0 1 0
                  1 0 0
                  0 0 1...]
    '''
    block_size = numpy.prod(sd_dims)
    n = numpy.prod(dims)
    N = n*dof
    k = int(N/block_size)

    def indices(sd, var):
        return range(sd*block_size*dof+var, (sd+1)*block_size*dof+var, dof)

    valV  = numpy.ones(N) / sqrt(block_size)
    rows = numpy.array(range(N))
    cols = numpy.zeros(N)
    for i in range(k):
        var = i % dof
        sd  = int(( i - var)/dof)
        idx = indices(sd, var)
        cols[idx] = i
    V = scipy.sparse.csc_matrix( (valV, (rows, cols)), shape=[N,k])
    return V

class BlockJacobi(spla.LinearOperator):

    def indices(self, sd):
        return range(sd*self.block_size, (sd+1)*self.block_size)

    def __init__(self, A, block_size, dof=1):
        '''
        Construct block Jacobi preconditioner with given block size.
        If dof>1, the block size is multiplied by dof (that is, each
        block will contain block_size grid points/cells with multiple
        degrees of freedom each).
        We use an LU factorization for the blocks because scipy doesn't
        have Cholesky.
        '''
        if A.shape[0] != A.shape[1]:
            raise Exception('A must be square!')
        if A.shape[0] % (block_size*dof):
            raise Exception('size of A must be a multiple of the block_size')

        self.shape = A.shape
        self.block_size = block_size*dof
        self.num_blocks = int(A.shape[0] / self.block_size)
        self.LU = []

        for i in range(self.num_blocks):
            idx = self.indices(i)
            self.LU.append( spla.splu(scipy.sparse.csc_matrix(A[idx,:][:,idx])))

    def solve(self,x):
        '''
        Apply inverse operator to solve a linear system with the block diagonal of A
        '''
        y = numpy.zeros(x.shape)
        for i in range(self.num_blocks):
            idx = self.indices(i)
            y[idx]=self.LU[i].solve(x[idx])
        return y

    def _matvec(self, x):
        return self.solve(x)

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

    def __init__(self, A, V):
        '''
        Construct deflated operator with given space V. V is checked for orthonormality on input.
        The operator is defined as

              op*x = (I - AQ)A(I-QA)x,
        with Q     = V (V'AV)\V'
        
        '''
        if A.shape[0] != A.shape[1]:
            raise Exception('A must be square!')

        k = V.shape[1]
        ortho_err = spla.norm( (V.T @ V) - scipy.sparse.eye(k) )
        if ortho_err>1e-10:
            raise Exception('input V to DeflatedOperator is not orthonormal: ||V^TV-I||=%g'%(ortho_err))
        self.A = A
        self.V = V
        self.dtype = A.dtype
        self.shape = A.shape

        self.E = scipy.sparse.csc_matrix(self.V.T @ (self.A @ self.V))
        self.Elu = spla.splu(self.E)

    def applyQ(self, x):
        '''
        computes y = Q*x
        with Q = V(V'AV)\V'
        '''
        q = self.V.T @ x
        return self.V @ self.Elu.solve(q)

    def proj(self, x):
        '''
        computes y = (I - QA)x
        with Q = V(V'AV)\V'
        '''
        return x - self.applyQ(self.A @ x)

    def projT(self, x):
        '''
        computes y = (I - AQ)x
        with Q = V(V'AV)\V'
        '''
        return x - self.A @ self.applyQ(x)

    def _matvec(self, x):
        '''
        apply deflated operator A,
          y = (I - AQ)A(I - QA)x,
          Q = V (V'AV)\(V'x)
        '''
        y = self.projT(self.A @ self.proj(x))
        return y

class DomainDecomposition(spla.LinearOperator):
    '''
    Given a matrix that represents some discretized PDE on an nx x ny mesh,
    with dof equations and unknowns per grid cell/point, and ordered according
    to a z-curve ordering, this class implements an Additive Schwarz preconditioner
    with optionally overlap or underlap between the subdomains.

    Furthermore, the object can be used to get GDSW-like deflation vectors:

        | A11 \ A12*V2 |
    V = |              |, where V2 are separator groups of 1's ('partition of unity')
        |        V2    |

    This space can then be passed to DeflatedOperator to get additional acceleration by a coarse
    operator (V'AV)^{-1}. Note that all of this is currently assuming symmetric matrices (otherwise we'll
    need left and right deflation spces)
    '''
    def __init__(self, A, block_size, dof=1, overlap=0):
        if A.shape[0] != A.shape[1]:
            raise Exception('A must be square!')
        if A.shape[0] % block_size:
            raise Exception('size of A must be a multiple of the block_size')
        if A.shape[0] % dof:
            raise Exception('size of A must be a multiple of the dof')

        self.shape = A.shape
        self.block_size = block_size*dof
        self.dof = dof
        self.num_blocks = int(A.shape[0] / self.block_size)
        self.LU11 = []
        #...

    def ind2sub(self, idx):
        '''
        Given a global index, returns
        Cartesian indices 0<=i<nx, 0<=j<ny and the variable type 0<=var<dof
        '''
        var  = idx % dof
        cell = int((idx-var)/self.dof)
        i, j = pymorton.deinterleave2(cell)
        return i, j, var

    def sub2ind(self,i,j,var):
        '''
        Given Cartesian coordinates and the variable type,
        returns the global index.
        '''
        cell = pymorton.interleave2(i,j)
        return cell * self.dof + var

    def subdomain(self, idx):
        '''
        returns the subdomain index of a global index idx
        '''
        return int(idx/self.block_size)

    def subdomain(self, i, j, var):
        '''
        returns the subdomain index of a subscript (Cartesian coordinates + variable type)
        '''
        return self.subdomain(sub2ind(i,j,var))

    def indices(self, sd):
        '''
        returns i1, i2: A11=A[i1,:][:,i1]for subdomain sd,
        A12=A[i1,:][:,i2], etc.

        How exactly we define the '2' variables is to be clarified.
        '''
        raise Exception('not implemented')
#        z_idx = range(sd*self.block_size, (sd+1)*self.block_size)
#        i1 = []
#        i2 = []
#        for ii in z_idx:
#            i,j,var = ind2sub(ii)
#            ii_left = subdomain(sub2ind(i+1,j,var))

