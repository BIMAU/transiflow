Choosing a backend
==================
TransiFlow implements different computational backends for different purposes, e.g. model development or large scale simulations.
We provide an overview here on what backends are available and which backend to choose in which situation.
See :mod:`transiflow.interface` for implementational details.

SciPy
-----
The default backend.
Suitable for model development and small problems (less than 1 million unknowns).
This backend is feature complete.

Epetra
------
A parallel backend for the Epetra package in Trilinos.
It requires the PyTrilinos Python package to be installed.
This does not include a parallel preconditioner or eigenvalue solver and can therefore not be used for solving large problems.
This backend is for development purposes only and can be used as a base class for other parallel Trilinos-based backends.

HYMLS
-----
A parallel backend for the Epetra-based HYMLS preconditioner.
It requires the PyTrilinos and HYMLS Python packages to be installed.
This backend is most suitable for large-scale parallel computations.
The preconditioner has shown good scalability with 4096 cores and over 500 million unknowns.
This backend is feature complete.

PETSc
-----
Similar to the Epetra backend, but for PETSc-based solvers.
It requires the petsc4py and mpi4py Python packages to be installed.
This does not include a parallel preconditioner or eigenvalue solver and can therefore not be used for solving large problems.
