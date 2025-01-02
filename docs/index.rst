.. TransiFlow documentation master file, created by
   sphinx-quickstart on Thu May 23 10:44:30 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to TransiFlow's documentation!
======================================

TransiFlow is a Python package for performing analysis on transitions in fluid flows.
For this purpose, we implement a pseudo-arclength :mod:`continuation <.Continuation>` method, an implicit :mod:`time integration <.TimeIntegration>` method, as well as finite-volume :mod:`discretizations <.Discretization>` of various fluid flow problems.
The package is fully agnostic of the computational backend, which allows for easy switching between e.g. SciPy for performing small scale simulations and PyTrilinos for large scale parallel computations.

.. toctree::
   :maxdepth: 3
   :caption: Usage

   continuation
   backends
   custom-bc
   custom-model

.. _problem definitions:
.. toctree::
   :maxdepth: 3
   :caption: Problem definitions

   problems/ldc
   problems/rb
   problems/dhc
   problems/tc
   problems/qg
   problems/amoc

.. toctree::
   :maxdepth: 2
   :caption: API documentation

   reference/transiflow

.. toctree::
   :caption: Symbols

   symbols

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
