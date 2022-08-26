import numpy
import scipy
import pickle

import matplotlib.pyplot as plt

from fvm import Interface
from fvm import plot_utils
from fvm import utils

from solver_util import *
from preconditioners import *
from deflation import *
from numpy.linalg import norm

def stokes_matrix(nx, ny):

    nz = 1
    dim = 2
    dof = 3
    n = dof * nx * ny * nz

    # Define the problem
    parameters = {'Problem Type': 'Lid-driven Cavity',
                  'Reynolds Number': 0,
                  'X-max': 1,
                  'Y-max': 1,
                  # Give back extra output (this is also more expensive)
                  'Verbose': False}

    interface = Interface(parameters, nx, ny, nz, dim, dof)


    x = numpy.zeros(n)

    A = interface.jacobian(x)

    A = scipy.sparse.csr_matrix((A.coA, A.jcoA, A.begA)).tocsc()

    return A

def proj(x, V):
    return x - V @ (V.T @ x)


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

    if plot_matrices:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(2,1)
        ax[0].spy(A0)
        ax[0].set_title('A, cartesian')
        ax[1].spy(A)
        ax[1].set_title('A, z-ordering')
        plt.show()

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

    it = 0
    x, flag = spla.gmres(A,rhs,x0,tol,maxiter=maxit,restart=maxbas,callback=count_iter)
    x = proj(x,V0)
    report('Ax=b (gmres)',x,it)

    it = 0
    Aproj = ProjectedOperator(A, V0)
    x, flag = spla.cg(Aproj,rhs,x0,tol,maxiter=maxit,callback=count_iter)
    report('(I-VV\')A(I-VV\')x=b (cg)',x,it)

    it = 0
    sx = 4
    sy = 4
    M_bj = AdditiveSchwarz(A, sx*sy, dof)
    x, flag = spla.gmres(A,rhs,x0,tol,maxiter=maxit,restart=maxbas, M=M_bj,callback=count_iter)
    report('M\\Ax=M\\b (gmres)',x,it)


    # deflation
    V = get_subdomain_groups((nx,ny),(sx,sy),dof)
    A_d = DeflatedOperator(A, V)
    x, flag, it = dgmres(A_d, rhs, x0, tol, maxiter=maxit, restart=maxbas)
    report('PAx=Pb (gmres)',x,it);

    # deflated with explicit null-space removal
    A_d0 = DeflatedOperator(A, V, V0)
    x, flag, it = dpcg(A_d0, rhs, x0, tol, maxiter=maxit)
    report('P0Ax=P0b (cg)',x,it);

#   # deflated and preconditioned method
    x, flag, it = dgmres(A_d, rhs, x0, tol, maxiter=maxit, restart=maxbas, M=M_bj)
    report('M\\PAx=M\\Pb (gmres)',x,it);

if __name__ == '__main__':

    for nx in [16, 32, 64]:
        main(nx, 2, False)
