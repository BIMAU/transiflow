Examples
==================================
The ``examples`` directory contains several examples on how to use TransiFlow.
They do this based on a certain problem definition, but in theory, all examples are useful for any problem definition.
On this page, we give an overview of what every example is supposed to demonstrate.

``ldc.py``
  The most basic example that shows the detection of a bifurcation point in a 2D lid-driven cavity, and computes the eigenvector at that point.

``ldc2.py``
  An example that goes more into detail on how to compute generalized eigenvalues and eigenvectors for the problem definition.

``ldc3.py``
  An example that shows how to perform time integration.
  This may for instance be useful when computing periodic orbits after a Hopf-Bifurcation.

``ldc3_3d.py``
  An example that shows how to use a parallel back-end (HYMLS in this case) for a 3D problem.

``dhc.py``
  Essentially the same as ``ldc.py`` but now for a differentially heated cavity, meaning this now includes temperature.

``qg.py``
  An example that shows how to move onto a stable branch after a pitchfork bifurcation by adding asymmetry to the problem.

``amoc.py``
  A more advanced version of ``qg.py`` with added temperature and salinity and more advanced post-processing.

..
    Explicitly enable math mode
.. math::
