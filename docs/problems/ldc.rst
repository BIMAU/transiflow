Lid-driven Cavity
=================
In the lid-driven cavity a fluid contained in a box is driven by a lid located at the top of the box that moves at a constant velocity $U$.

Governing Equations
-------------------
In a domain of dimensions $L \\times B \\times D$, the governing incompressive Navier-Stokes equations are given by

.. math:: \frac{\partial \mathbf{u}}{\partial t} + (\mathbf{u} \cdot \nabla) \mathbf{u} &= -\frac{1}{\rho_0}\nabla p + \nu \nabla^2\mathbf{u} - ge_z\\
          \nabla \cdot \mathbf{u} &= 0

with boundary conditions

.. math:: x &= 0, L &:~& u = v = w = 0\\
          y &= 0, B &:~& u = v = w = 0\\
          z &= 0 &:~& u = v = w = 0\\
          z &= D &:~& u = U, v = w = 0

See :ref:`symbols` for the definition of all quantities used here.

Nondimensional formulation
--------------------------
In TransiFlow, we implement the following nondimensionalized equations

.. math:: \frac{\partial \tilde{\mathbf{u}}}{\partial \tilde t} + (\tilde{\mathbf{u}} \cdot \tilde\nabla) \tilde{\mathbf{u}} &= -\tilde\nabla \tilde p +\frac{1}{\mathrm{Re}}\tilde\nabla^2\tilde{\mathbf{u}}\\
          \tilde\nabla \cdot \tilde{\mathbf{u}} &= 0

with boundary conditions

.. math:: \tilde x &= 0, 1 &:~& \tilde u = \tilde v = \tilde w = 0\\
          \tilde y &= 0, A_y &:~& \tilde u = \tilde v = \tilde w = 0\\
          \tilde z &= 0 &:~& \tilde u = \tilde v = \tilde w = 0\\
          \tilde z &= A_z &:~& \tilde u = 1, \tilde v = \tilde w = 0

Here $x$, $y$ and $z$ are scaled by $L$, and hence $A_y = B / L$ and $A_z = D / L$. The other quantities are scaled using $u = \\tilde u U$, $t = \\tilde t L / U$, $p = \\tilde p \\rho_0 U^2 - \\rho_0gz$. Moreover, the Reynolds number is given by $\\mathrm{Re} = UL / \\nu$. All other quantities are defined in :ref:`symbols`.

Parameters
----------
These are the relevant parameters in the ``parameters`` dictionary for this problem type.

===================== ============= =====
Parameter name        Default value Notes
===================== ============= =====
``'Problem Type'``                  Set to ``'Lid-driven Cavity'``
``'Lid velocity'``    1.0
``'Reynolds Number'`` 1.0
===================== ============= =====
