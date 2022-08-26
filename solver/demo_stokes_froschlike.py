import numpy
import scipy
import pickle

import matplotlib.pyplot as plt

from fvm import Interface
from fvm import plot_utils
from fvm import utils

from demo_stokes import stokes_matrix, proj

from solver_util import *
from preconditioners import *
from deflation import *
from numpy.linalg import norm

def main(nx, sx, plot_matrices=False):

    ny = nx
    sy = sx
    dof = 3
    N = nx*ny*dof
    print('\nnx=%d, ny=%d'%(nx,ny))
    print('sx=%d, sy=%d'%(sx,sy))

    A0 = stokes_matrix(nx,ny)
    z_idx = get_z_ordering(nx,ny,dof=dof)
    A=A0[z_idx,:][:,z_idx]

    xex = numpy.random.randn(N)
    # project out the null space (constant pressure mode)
    V0 = numpy.zeros((N,1))
    V0[range(dof-1,N,dof)]=1/sqrt(nx*ny)
    xex = proj(xex, V0)
    rhs=A*xex

    x0=numpy.zeros(N)
    tol=1e-10
    maxit=4000
    maxbas=250

    print('%16s\titer\tresidual\terror (error orth V0)'%('label'))
    def report(label,x,iter):
        err=x-xex
        err_u = norm(err[range(0,N,3)])
        err_v = norm(err[range(1,N,3)])
        err_p = norm(err[range(2,N,3)])
        err_p2= norm(err[range(2,N,3)]-err[2])

        print('%16s\t%d\t%5.3e\t%5.3e (%5.3e)'%(label,iter,norm(A*x-rhs), norm(x-xex), norm(proj(x,V0) - xex)))
        #print('separate u/v/p/p0: '+str([err_u, err_v, err_p, err_p2]))
        #print('V0^Txex = '+str(V0.T@xex))
        #print('V0^Tx   = '+str(V0.T@x))

    def count_iter(xj):
        nonlocal it
        it += 1

    #2-level Schwarz method
    M_as, A_d = build_stokes_preconditioner(A, nx, ny, sx, sy)

    x, flag, it = dgmres(A_d, rhs, x0, tol, maxiter=maxit, restart=maxbas)
    report('GDSW deflation (gmres)',x,it);

    x, flag, it = dgmres(A_d, rhs, x0, tol, maxiter=maxit, restart=maxbas, M=M_as)
    report('2-level Schwarz/GDSW (gmres)',x,it);

if __name__ == '__main__':

    for nx in [8,16, 32, 64]:
        main(nx, 2, False)
