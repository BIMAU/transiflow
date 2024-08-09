import numpy

from transiflow.utils import norm

from transiflow.Discretization import Discretization
from transiflow.CylindricalDiscretization import CylindricalDiscretization


class BaseInterface:
    '''This class defines a base interface to the NumPy backend for the
    discretization. We use this so we can write higher level methods
    such as pseudo-arclength continuation without knowing anything
    about the underlying methods such as the solvers that are present
    in the backend we are interfacing with.'''

    def __init__(self, parameters, nx, ny, nz, dim=None, dof=None, x=None, y=None, z=None):
        self.nx = nx
        self.ny = ny
        self.nz = nz

        self.discretization = Discretization(parameters, nx, ny, nz, dim, dof, x, y, z)

        if 'Problem Type' in parameters and 'Taylor-Couette' in parameters['Problem Type']:
            self.discretization = CylindricalDiscretization(parameters, nx, ny, nz, dim, dof, x, y, z)

        self.dim = dim or self.discretization.dim
        self.dof = dof or self.discretization.dof

        self.parameters = parameters

        # Eigenvalue solver caching
        self._subspaces = None

    def debug_print(self, *args):
        if self.parameters.get('Verbose', False):
            print('Debug:', *args, flush=True)

    def debug_print_residual(self, string, jac, x, rhs):
        if self.parameters.get('Verbose', False):
            r = norm(jac @ x - rhs)
            self.debug_print(string, '{}'.format(r))

    def set_parameter(self, name, value):
        '''Set a parameter in self.parameters while also letting the
        discretization know that we changed a parameter. '''
        self.discretization.set_parameter(name, value)

    def get_parameter(self, name):
        '''Get a parameter from self.parameters through the discretization.'''
        return self.discretization.get_parameter(name)

    def rhs(self, state):
        '''Right-hand side in M * du / dt = F(u).'''
        raise NotImplementedError()

    def jacobian(self, state):
        '''Jacobian J of F in M * du / dt = F(u).'''
        raise NotImplementedError()

    def mass_matrix(self):
        '''Mass matrix M in M * du / dt = F(u).'''
        raise NotImplementedError()

    def solve(self, jac, rhs, rhs2=None, V=None, W=None, C=None):
        '''Solve J y = x for y.'''
        raise NotImplementedError()

    def _eigs(self, jada_interface, jac_op, mass_op, prec_op, state, return_eigenvectors, enable_recycling):
        '''Internal helper for eigs()'''

        from jadapy import jdqz, orthogonalization

        parameters = self.parameters.get('Eigenvalue Solver', {})
        arithmetic = parameters.get('Arithmetic', 'complex')
        target = parameters.get('Target', 0.0)
        initial_subspace_dimension = parameters.get('Initial Subspace Dimension', 0)
        subspace_dimensions = [parameters.get('Minimum Subspace Dimension', 30),
                               parameters.get('Maximum Subspace Dimension', 60)]
        enable_recycling = parameters.get('Recycle Subspaces', enable_recycling)
        tol = parameters.get('Tolerance', 1e-7)
        num = parameters.get('Number of Eigenvalues', 5)

        if not enable_recycling:
            self._subspaces = None

        if not self._subspaces and initial_subspace_dimension > 0:
            # Use an inverse iteration to find guesses
            # for the eigenvectors closest to the target
            V = jada_interface.vector(initial_subspace_dimension)
            V[:, 0] = jada_interface.random()
            orthogonalization.normalize(V[:, 0])

            for i in range(1, initial_subspace_dimension):
                V[:, i] = jada_interface.prec(mass_op @ V[:, i-1])
                orthogonalization.orthonormalize(V[:, 0:i], V[:, i])

            self._subspaces = [V]
        elif self._subspaces:
            k = self._subspaces[0].shape[1]
            V = jada_interface.vector(k + 1)
            V[:, 0:k] = self._subspaces[0]
            V[:, k] = jada_interface.random()
            orthogonalization.orthonormalize(V[:, 0:k], V[:, k])

            gamma = numpy.sqrt(1 + abs(target) ** 2)
            W = jada_interface.vector(k + 1)
            W[:, 0:k] = self._subspaces[1]
            W[:, k] = (jac_op @ V[:, k]) * (1 / gamma) - (mass_op @ V[:, k]) * (target / gamma)
            orthogonalization.orthonormalize(W[:, 0:k], W[:, k])

            self._subspaces = [V, W]

        result = jdqz.jdqz(jac_op, mass_op, num, tol=tol, subspace_dimensions=subspace_dimensions, target=target,
                           interface=jada_interface, arithmetic=arithmetic, prec=prec_op,
                           return_eigenvectors=return_eigenvectors, return_subspaces=True,
                           initial_subspaces=self._subspaces)

        if return_eigenvectors:
            alpha, beta, v, q, z = result
            self._subspaces = [q, z]
            idx = range(len(alpha))
            idx = sorted(idx, key=lambda i: -(alpha[i] / beta[i]).real if (alpha[i] / beta[i]).real < 100 else 100)

            w = v.copy()
            eigs = alpha.copy()
            for i in range(len(idx)):
                w[:, i] = v[:, idx[i]]
                eigs[i] = alpha[idx[i]] / beta[idx[i]]
            return eigs[:num], w
        else:
            alpha, beta, q, z = result
            self._subspaces = [q, z]
            eigs = numpy.array(sorted(alpha / beta, key=lambda x: -x.real if x.real < 100 else 100))
            return eigs[:num]

    def eigs(self, state, return_eigenvectors=False, enable_recycling=False):
        '''Compute the generalized eigenvalues of beta * J(x) * v = alpha * M * v.'''
        raise NotImplementedError()
