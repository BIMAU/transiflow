import numpy
import scipy
import pymorton

import scipy.sparse.linalg as spla
from math import sqrt, ceil, log

def get_z_ordering(nx, ny, dof=1, x_offset=0, y_offset=0):
    '''
    z_idx = get_z_ordering(nx,ny, dof=1) [last few arguments are only for internal use]
    
    Reorder by a Morton (z-)curve to get subdomains with contiguous indices:

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

    Returns z_idx and z_inv such that if v is in cartesian ordering,
    w=v[z_idx] is in z-ordering, and v=w[z_inv].
    To reorder a sparse (scipy)  matrix, you can use A[z_idx,:][:,z_idx].

    This function is currently restricted to 2D but allows
    problems with multiple degrees of freedom per grid point (dof>1).
    In that case, the dofs per grid point are kept together.

    It is possible that the z ordering has 'holes' if the grid has dimensions
    that are unequal or not a power of 2. For example, for nx=ny=3, we get

    0 1 4
    2 3 6
    8 9 12

    In that case, we re-map them to:

    0 1 4
    2 3 5
    6 7 8

    This is done by recursively applying the z-ordering to
    blocks of s^2 cells, where s is a power of 2.

    '''

    if any(map(lambda item: item<=0, [nx,ny,dof])):
        raise Exception('all input arguments must be strictly positive')

    N = nx*ny*dof
    z_idx = numpy.zeros(N)

    nx0 = pow(2,(int(log(nx,2))))
    ny0 = pow(2,(int(log(ny,2))))

    s=min(nx0,ny0)

    for j in range(s):
        for i in range(s):
            c_id = (j+y_offset)*(x_offset+nx)+x_offset+i
            z_id = pymorton.interleave2(i,j)
            for var in range(dof):
                z_idx[z_id*dof+var] = c_id*dof+var

    offset=s*s*dof

    if nx>s:
        # east
        z_idx2 = get_z_ordering(nx-s, s, dof, x_offset=x_offset+s,y_offset=y_offset)
        len=z_idx2.size
        z_idx[range(offset,offset+len)] = z_idx2
        offset = offset + len
    if ny>s:
        # south
        z_idx3 = get_z_ordering(nx, ny-s, dof, x_offset=x_offset, y_offset=y_offset+s)
        len=z_idx3.size
        z_idx[range(offset,offset+len)] = z_idx3

    return z_idx

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

def get_separator_groups(nx,sx,ny,sy,nz=1,sz=1,dof=1):
    '''
    Returns a sparse matrix in CSC format with N=nx*ny*nz*dof rows
    and m<<N columns. Each column index represents a 'separator group',
    marked in the following sketch by its column index. Note that interior
    variables are all grouped together in group 0.
    For nx=ny=8, sx=sy=3:

    0  0  1  0  0  4  0  0
    0  0  1  0  0  4  0  0
    2  2  3  5  5  6 14 14
    0  0  7  0  0 10  0  0
    0  0  7  0  0 10  0  0
    8  8  9 11 11 12 16 16
    0  0 17  0  0 20  0  0
    0  0 17  0  0 20  0  0

    These indices are obtained using a z-curve for the subdomains,
    so there may be 'holes' in the indexing, e.g. at the right boundary
    where there are no separators. We therefore renumber them after construction
    to give coniguous column indices.

    '''

    dim = 2
    if nz != 1:
        dim = 3
        raise Exception('not implemented for 3D yet')

    if dof > dim:
        raise Exception('not implemented for Stokes-like problems')


    # we use z-ordering of the subdomain IDs, which will produce holes in the
    # numbering of subdomains unless nx is a multiple of sx, etc. This can be
    # fixed in a future 'proper' implementation.
    if nx%sx or ny%sy:
        raise Exception('up to now, nx must be a multiple of sx, and similar for ny resp. sy.')

    num_sd_x = ceil(nx/sx)
    num_sd_y = ceil(ny/sy)

    sd_z_idx, sd_z_inv = get_z_ordering(num_sd_x, num_sd_y, dof=3*dof)

    print(sd_z_idx)
    print(sd_z_inv)


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
            raise Exception('input V to DeflatedOperator is not orthonormal: ||V^TV-I||=%g'%(ortho_err))
        self.A = A
        self.V = V
        self.V0 = V0
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

