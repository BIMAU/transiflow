Adding a custom backend
=======================
Adding a custom backend can be useful when using an external package for providing a model, or when using a custom solver.
As mentioned in :ref:`choosing a backend`, we provide some basic parallel backends (:mod:`Epetra <.interface.Epetra>` and :mod:`PETSc <.interface.PETSc>`) that can be used as base classes.

There are only a few methods that should be implemented when adding a custom backend.
These methods are the ones that raise a ``NotImplementedError`` in :mod:`BaseInterface <.interface.BaseInterface>`.
Of course all other methods in :mod:`BaseInterface <.interface.BaseInterface>` can also be overloaded, but they are all just convenience methods and are not required by this package.

``solve(self, jac, rhs)`` or optionally ``solve(self, jac, rhs, rhs2=None, V=None, W=None, C=None)``
  Should solve a linear system with matrix ``jac`` and right-hand side ``rhs`` and then return the solution.

``jacobian(self, state)``
  Should return a matrix object for the current state that can be used by the linear solver.

``rhs(self, state)``
  Should return the right-hand side for the current state.

``vector(self)``
  Should return a new vector that can interact with the linear solver.

``mass_matrix(self)``
  Should return a mass matrix object as used for time integration and eigenvalue computation.
  This is optional if neither of those is used.

``eigs(self, state, return_eigenvectors=False, enable_recycling=False)``
  Used for eigenvalue computation during the continuation.
  This is optional and only used when automatically detecting bifurcations.

``set_parameter(self, name, value)``
  Used during the continuation to set the continuation parameter.
  This already has a default implementation that can be used if no custom discretization is being provided.

..
    Explicitly enable math mode
.. math::
