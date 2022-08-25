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

    Returns z_idx such that if v is in cartesian ordering,
    w=v[z_idx] is in z-ordering.
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
    z_idx = numpy.zeros(N,dtype='int')

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

def get_inv_ordering(idx):
    '''
    Given a 1-to-1 mapping idx such that x = y[idx],
    returns the inverse mapping inv: y = x[inv]
    '''
    inv = numpy.zeros(idx.size,idx.dtype)
    inv[idx]=range(idx.size)
    return inv

def flatten(list_of_lists):
    '''
    Given something like [[1,2],[3],[4,5,6]],
    returns [1,2,3,4,5,6]
    '''
    return [item for sublist in list_of_lists for item in sublist]

def partition_of_unity(index_lists):
    '''
    Creates a sparse matrix V with
    N rows and k columns, where
    k is the number of sublists in 'index_lists',
    and N-1 is the maximum index found in any sublist.

    If lj = index_lists[j],
    column j of V contains entries

    V[i,j]=0                 if  i not in lj
    V[i,j]=1/sqrt(len(lj))   if  i in lj

    If the index sets are disjunct and non-empty, V is structurally orthonormal.
    '''
    k = len(index_lists)
    N = max(i for lst in index_lists for i in lst)+1

    rows = flatten(index_lists)
    nnz=len(rows)

    valV  = numpy.zeros(nnz)
    cols = numpy.zeros(nnz,dtype='int')
    pos = 0
    for i in range(k):
        idx = index_lists[i]
        cols[range(pos,pos+len(idx))] = i
        valV[range(pos,pos+len(idx))] = 1/sqrt(len(idx))
        pos += len(idx)

    V = scipy.sparse.csc_matrix( (valV, (rows, cols)), shape=[N,k])
    return V

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
    index_lists=[]
    for sd in range(int(k/dof)):
        for var in range(dof):
            index_lists.append(indices(sd,var))
    return partition_of_unity(index_lists)

class StokesDD:
    '''
    Assuming a 2D Stokes equation on an nx x ny C-grid,
    and a subdomain size of sx*sy cells per subdomain,
    and z-curve ordering, this function returns i1, i2 and i3
    such that
    A(i1,i1) represents interior variables of the subdomain,
    and the i2 variables are the velocity separators. i3 are
    the pressure indices, i3 is a subset of i1.

    The intended use of these arrays is that

    V2  with V2[i2,sd]=v2 is a partition of unity of the velocity separators of the domain,
    and V1 = A11\(A12 V2) is the 'harmonic extension' to the interior variables.
    V3 with
    '''

    def __init__(self, nx, ny, sx, sy):
        '''
        '''
        self.dim=2
        self.dof=3
        self.nx=nx
        self.ny=ny
        self.sx=sx
        self.sy=sy
        self.group_ID=dict()

        self.z_idx = get_z_ordering(nx,ny,dof=1)
        self.z_inv = get_inv_ordering(self.z_idx)
        self.Z = self.z_inv.reshape(ny,nx).transpose()

        self.num_sd_x = ceil(nx/sx)
        self.num_sd_y = ceil(ny/sy)

        # create a z-ordering for the subdomains
        self.sd_idx = get_z_ordering(self.num_sd_x, self.num_sd_y, dof=1)
        self.sd_inv = get_inv_ordering(self.sd_idx)
        self.Zsd = self.sd_inv.reshape(self.num_sd_y,self.num_sd_x).transpose()

    def plot_cgrid(self, indices, colors, title=None, markersize=12, ax=None, plotgrid=True, plotindices=False):
        '''
        given a set of indices into the c-grid
        (e.g. as returned by the indices() function),
        make a plot of the C-grid with these variables
        marked. Both indices and colors should be lists
        (indices a list of lists) of equal length.
        '''
        from matplotlib import pyplot as plt

        if type(indices) is not list:
            raise Exception('indices must be a list or list of lists')

        if len(indices)==0:
            return

        if type(indices[0]) is not list:
            indices = [indices]

        if type(colors) is not list:
            colors = [colors]

        if type(markersize) is not list:
            markersize = [markersize]

        # cyclic extension of list of colors and markersize
        while len(markersize) < len(indices):
            markersize += markersize

        if ax is None:
            plotgrid=True
            ax=plt.axes()

        if plotgrid:
            for j in range(self.ny+1):
                ax.plot([-0.5,self.nx-0.5],[-j+0.5,-j+0.5],'-')
            for i in range(self.nx+1):
                ax.plot([i-0.5,i-0.5],[0.5-self.ny,0.5],'-')

        if title is not None:
            plt.title(title)

        MarkerDict = dict(([0,'>'],[1,'v'],[2,'o']))
        ioff = [0.5, 0,   0]
        joff = [0,   -0.5, 0]

        for set in range(len(indices)):
            idx = indices[set]
            clr = colors[set]
            ms  = markersize[set]
            for z_id in idx:
                cell  = int(z_id/self.dof)
                var   = z_id % self.dof
                i, j = self.cell_coordinates(cell)
                ax.plot([i+ioff[var]],[-j+joff[var]],MarkerDict[var], markersize=ms, markerfacecolor=clr)
                if plotindices:
                    ax.text(i+ioff[var],-j+joff[var],str(z_id), ha='center',va='center',fontsize=ms-2)
        plt.draw()



    def subdomain_coordinates(self, sd):
        '''
        Given a subdomain ID (in z-ordering),
        returns indices i and j of that subdomain
        in a Cartesian grid of subdomains.
        '''
        sd_cart = self.sd_idx[sd]
        ic = sd_cart % self.num_sd_x
        jc = int(sd_cart/self.num_sd_x)
        return ic, jc

    def cell_coordinates(self, cell):
        '''
        Given a grid cell ID (in z-ordering),
        returns indices i and j of that grid cell
        in a Cartesian grid of nx x ny cells.
        '''
        c_cart = self.z_idx[cell]
        ic = c_cart % self.nx
        jc = int(c_cart/self.nx)
        return ic, jc


    def num_subdomains(self):
        return self.num_sd_x*self.num_sd_y


    def get_group_id(index_list):
        '''
        Given a list of cell indices, returns
        a group index. If the particular group
        of variables does not have a group ID yet,
        one is created and stored. 

        Group IDs

        - start at self.num_subdomains() to make sure subdomain
          interiors can be handled consistently as groups, too
        - are assigned incrementally, so two consecutive calls
          with new index sets give ids j and j+1, j>=num_subdomains()
        - are independent of the ordering of the unknowns,
          so the groups [1,2,3] and [3,1,2], for instance,
          will give the same group ID, but [1,2,3,4] will
          receive a new one.
        '''
        key = tuple(index_list.sorted())
        self.group_ID.setdefault(self.num_subdomains()+len(self.group_ID))
        return self.group_ID[key]

    def indices(self, sd):
        '''
        For subdomain 'sd' (0<=sd<=self.num_subdomains),
        returns idx0, idx1, idx2  and idx3 such that, if A is the Jacobian of a 'Stokes-like' problem
        on a C-grid, 

        A[idx0,:][:,idx0] is the subdomain matrix with minimal overlap, i.e. for this 2x2 domain,
        the marked variables (o: p, T; >: u; '^': v)

            +-^-+-^-+
            |   |   |
            > o > o >
            |   |   |
            +-^-+-^-+
            |   |   |
            > o > o >
            |   |   |
            +-^-+-^-+

        A[idx1,:][:,idx1] is the interior (*: excluded variables)

            +-*-+-*-+
            |   |   |
            * o > o *
            |   |   |
            +-^-+-*-+
            |   |   |
            * o * o *
            |   |   |
            +-*-+-*-+

            A[idx2,:][:,idx2] are the separators,


        of the subdomain, and idx2 are the indices of the separator velocities.
        idx3 are the pressure nodes of the subdomain (a subset of idx1).

            +-2-+-2-+
            |   |   |
            2 3 > 3 2
            |   |   |
            +-^-+-2-+
            |   |   |
            2 3 2 3 2
            |   |   |
            +-2-+-2-+

        To make further processing simpler, the idx2 variables (velocity separators)
        are returned as a 'list of lists', grouped by variable type and subdomain face.

        '''
        # get the coordinates of the first cell in the subdomain
        # (this is a robust but slow implementation, we need it for now
        # because our z-ordering constructs an array rather than just
        # providing a function, as a 'real' implementation would)
        ic, jc = self.subdomain_coordinates(sd)

        irng = range(ic*self.sx, min((ic+1)*self.sx, self.nx))
        jrng = range(jc*self.sx, min((jc+1)*self.sy, self.ny))
        #i-index of grid cells to the left of the subdomain
        im1 = []
        if ic > 0:
            im1 += [irng[0]-1]
        # j-index of the grids cell on top of the current subdomain
        jm1 = []
        if jc > 0:
            jm1 += [jrng[0]-1]

        # all subdomain variables (including minimal overlap to neighboring subdomains):
        idx0  = []
        idx0 += list((0+self.dof*self.Z[im1+list(irng),:][:,jrng]).flat)
        idx0 += list((1+self.dof*self.Z[irng,:][:,jm1+list(jrng)]).flat)
        idx0 += list((2+self.dof*self.Z[irng,:][:,list(jrng)]).flat)

        # interior nodes: u-velocities
        idx1 = list((0+self.dof*self.Z[range(irng[0],irng[-1]),:][:,range(jrng[0],jrng[-1])]).flat)
        # interior nodes: v-velocities
        idx1 += list((1+self.dof*self.Z[range(irng[0],irng[-1]),:][:,range(jrng[0],jrng[-1])]).flat)
        # pressure
        idx3 = list((2+self.dof*self.Z[irng,:][:,jrng]).flat)
        idx1 += idx3

        # last row and column of u's / v's are either separator velocities or interior
        idx2 = []
        last_u1 = list((0+self.dof*self.Z[irng[-1],:][jrng]).flat)
        last_v1 = list((1+self.dof*self.Z[irng[-1],:][range(jrng[0],jrng[-1])]).flat)
        last_u2 = list((0+self.dof*self.Z[range(irng[0],irng[-1]),:][:,jrng[-1]]).flat)
        last_v2 = list((1+self.dof*self.Z[irng,:][:,jrng[-1]]).flat)

        if ic < self.num_sd_x-1:
            idx2.append(last_u1)
            idx2.append(last_v1)
        else:
            idx1 += last_u1[:]
            idx1 += last_v1[:]
            print(idx1)

        if jc < self.num_sd_y-1:
            idx2.append(last_u2)
            idx2.append(last_v2)
        else:
            idx1 += last_u2
            idx1 += last_v2

        return sorted(idx0), sorted(idx1), idx2, idx3


