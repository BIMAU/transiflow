import json
import numpy

from transiflow.utils import norm

from transiflow.Discretization import Discretization
from transiflow.CylindricalDiscretization import CylindricalDiscretization


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (numpy.int8, numpy.int16, numpy.int32, numpy.int64,
                            numpy.uint8, numpy.uint16, numpy.uint32, numpy.uint64)):

            return int(obj)
        elif isinstance(obj, (numpy.float16, numpy.float32, numpy.float64)):
            return float(obj)
        elif isinstance(obj, (complex, numpy.complex64, numpy.complex128)):
            return {'__complex__': True, 'real': obj.real, 'imag': obj.imag}
        elif isinstance(obj, numpy.ndarray):
            return obj.tolist()

        return json.JSONEncoder.default(self, obj)


def numpy_decoder(obj):
    if '__complex__' in obj:
        return complex(obj['real'], obj['imag'])
    return obj


class BaseInterface:
    '''This is the base class for all interfaces. We use this so we
    can write higher level methods such as pseudo-arclength
    continuation without knowing anything about the underlying methods
    such as the solvers that are present in the backend we are
    interfacing with.

    This class provides some basic implementations, which generally
    only work in serial, as well as some abstract methods that need to
    be implemented by the derived interface.

    '''

    def __init__(self, parameters, nx, ny, nz=1, dim=None, dof=None,
                 x=None, y=None, z=None, boundary_conditions=None):
        self.nx = nx
        self.ny = ny
        self.nz = nz

        if 'Problem Type' in parameters and 'Taylor-Couette' in parameters['Problem Type']:
            self.discretization = CylindricalDiscretization(parameters, nx, ny, nz, dim, dof,
                                                            x, y, z, boundary_conditions)
        else:
            self.discretization = Discretization(parameters, nx, ny, nz, dim, dof,
                                                 x, y, z, boundary_conditions)

        self.dim = dim or self.discretization.dim
        self.dof = dof or self.discretization.dof

        self.x = self.discretization.x
        self.y = self.discretization.y
        self.z = self.discretization.z

        self.parameters = parameters

        # Eigenvalue solver caching
        self._subspaces = None

    def _debug_print(self, *args):
        if self.parameters.get('Verbose', False):
            print('Debug:', *args, flush=True)

    def _debug_print_residual(self, string, jac, x, rhs):
        if self.parameters.get('Verbose', False):
            r = norm(jac @ x - rhs)
            self.debug_print(string, '{}'.format(r))

    def vector(self):
        '''Return a zero-initialized state vector suitable for the
        currently defined problem type.'''
        raise NotImplementedError()

    def vector_from_array(self, array):
        '''Create a state vector suitable for this interface from an
        array.'''
        return array

    def array_from_vector(self, vector):
        '''Create a numpy array suitable for postprocessing from a
        state vector.'''
        return vector

    def set_parameter(self, name, value):
        '''Set a parameter in the discretization.

        Parameters
        ----------
        name : str
            Name of the parameter in the parameter list.
        value : scalar
            Value of the parameter.

        '''
        self.discretization.set_parameter(name, value)

    def get_parameter(self, name):
        '''Get a parameter value from the discretization.

        Parameters
        ----------
        name : str
            Name of the parameter in the parameter list.

        '''
        return self.discretization.get_parameter(name)

    def save_json(self, name, obj):
        '''Save an object to a json file

        Parameters
        ----------
        name : str
            Name of the file.
        obj : Any
            Serializable object to save to the file.
        '''

        with open(name, 'w') as f:
            json.dump(obj, f, cls=NumpyEncoder)

    def save_parameters(self, name):
        '''Save the current parameter set to a file.

        Parameters
        ----------
        name : str
            Name of the file without extension.

        '''

        params_name = name + '.params'
        self.save_json(params_name, self.parameters)
        print('Wrote parameters to', params_name, flush=True)

    def save_state(self, name, x):
        '''Save the state ``x`` along with the current parameter set.

        Parameters
        ----------
        name : str
            Name of the file.
        x : array_like
            State at the current parameter values.

        '''
        self.save_parameters(name)

        if not name.endswith('.npy'):
            name += '.npy'

        numpy.save(name, x)
        print('Wrote state to', name, flush=True)

    def load_json(self, name):
        '''Load the an object from a json file.

        Parameters
        ----------
        name : str
            Name of the file without extension.

        '''
        with open(name, 'r') as f:
            return json.load(f, object_hook=numpy_decoder)

    def load_parameters(self, name):
        '''Load the parameter set from a file.

        Parameters
        ----------
        name : str
            Name of the file without extension.

        '''
        params_name = name + '.params'

        old_parameters = self.parameters.copy()
        self.parameters.update(self.load_json(params_name))
        print('Read parameters from', params_name, flush=True)

        for parameter_name, value in self.parameters.items():
            old_value = old_parameters.get(parameter_name)
            if value != old_value:
                print('Updated \'{}\' from {} to {}'.format(parameter_name, old_value, value),
                      flush=True)

    def load_state(self, name):
        '''Load the state ``x`` along with the current parameter set.

        Parameters
        ----------
        name : str
            Name of the file.

        '''
        self.load_parameters(name)

        if not name.endswith('.npy'):
            name += '.npy'

        x = numpy.load(name)
        print('Read state from', name, flush=True)

        return x

    def rhs(self, state):
        r'''Compute the right-hand side of the DAE. That is the
        right-hand side $F(u, p)$ in

        .. math:: M(p) \frac{\d u}{\d t} = F(u, p)

        Parameters
        ----------
        state : array_like
            State $u$ at which to evaluate $F(u, p)$

        Returns
        -------
        rhs : array_like
            The value of $F(u, p)$

        '''
        raise NotImplementedError()

    def jacobian(self, state):
        r'''Compute the Jacobian matrix $J(u, p)$ of the right-hand
        side of the DAE. That is the Jacobian matrix of $F(u, p)$ in

        .. math:: M(p) \frac{\d u}{\d t} = F(u, p)

        Parameters
        ----------
        state : array_like
            State $u$ at which to evaluate $J(u, p)$

        Returns
        -------
        jac : Matrix
            The matrix $J(u, p)$ in a suitable sparse format

        '''
        raise NotImplementedError()

    def mass_matrix(self):
        r'''Compute the mass matrix of the DAE. That is the mass
        matrix $M(p)$ in

        .. math:: M(p) \frac{\d u}{\d t} = F(u, p)

        Returns
        -------
        mass : Matrix
            The matrix $M(p)$ in a suitable sparse format

        '''
        raise NotImplementedError()

    def solve(self, jac, rhs, rhs2=None, V=None, W=None, C=None):
        '''Solve $J(u, p) y = x$ for $y$.

        Parameters
        ----------
        jac : Matrix
            The Jacobian matrix $J(u, p)$ as returned by
            ``jacobian()``.
        rhs : array_like
           The right-hand side vector $x$.
        rhs2 : array_like, optional
            Extension of ``rhs`` in case a bordered system is solved.
        V : array_like, optional
            Border case a bordered system ``[A, V; W^T, C] [y; y2] =
            [x; x2]`` is solved.
        W : array_like, optional
            Border case a bordered system ``[A, V; W^T, C] [y; y2] =
            [x; x2]`` is solved.
        C : matrix_like, optional
            Border case a bordered system ``[A, V; W^T, C] [y; y2] =
            [x; x2]`` is solved.

        '''
        raise NotImplementedError()

    def _eigs(self, jada_interface, jac_op, mass_op, prec_op, state,
              return_eigenvectors, enable_recycling):
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
        r'''Compute the generalized eigenvalues of $\\beta * J(u, p) *
        v = \\alpha * M(p) * v$.

        Parameters
        ----------
        state : array_like
            State $u$ at which to evaluate $J(u, p)$.
        return_eigenvectors : bool, optional
            Whether to return the eigenvectors $v$.
        enable_recycling : bool, optional
            Whether to use the previous eigenvalue space as initial
            search space.

        Returns
        -------
        eigs : array_like
            Eigenvalues $\\alpha / \\beta$.
        v : array_like
            Corresponding eigenvectors in case ``return_eigenvectors``
            is ``True``.

        '''
        raise NotImplementedError()
