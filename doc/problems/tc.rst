Taylor-Couette Flow
===================
Taylor-Couette flow describes the flow of a liquid enclodes by two rotating cylinders. The inner cylinder of radius $r_i$ rotates with angular frequency $\\omega_i$, the outer cylinder of radius $r_o$ rotates with angular frequency $\\omega_o$.

Governing Equations
-------------------
In a cylindrical domain of dimensions $d \\times 2\\pi \\times L$, where $d = r_o - r_i$, the governing equations for Taylor-Couette flow are given by

.. math:: \frac{\partial \mathbf{u}}{\partial t} + (\mathbf{u} \cdot \nabla) \mathbf{u} &= -\frac{1}{\rho_0}\nabla p + \nu \nabla^2\mathbf{u} - ge_z\\
          \nabla \cdot \mathbf{u} &= 0

with boundary conditions

.. math:: r &= r_i &:~& u = w = 0, v = \omega_i r_i\\
          r &= r_o &:~& u = w = 0, v = \omega_o r_o\\
          z &= 0, L &:~& u = v = w = 0

where $\\omega_o$ and $\\omega_i$ are the angular frequencies of respectively the outer and inner cylinders. In the $\\theta$-direction we apply periodic conditions, i.e. $u(r, 0, z, t) = u(r, 2\\pi, z, t)$ and similar for other quantities. See :ref:`symbols` for the definition of all quantities used here.

Nondimensional formulation
--------------------------
In TransiFlow, we implement the following nondimensionalized equations

.. math:: \frac{\partial \tilde{\mathbf{u}}}{\partial \tilde t} + (\tilde{\mathbf{u}} \cdot \tilde\nabla) \tilde{\mathbf{u}} &= -\tilde\nabla \tilde p +\frac{1}{\mathrm{Ta}}\tilde\nabla^2\tilde{\mathbf{u}}\\
          \tilde\nabla \cdot \tilde{\mathbf{u}} &= 0

with boundary conditions

.. math:: \tilde r &= 1 &:~& \tilde u = \tilde w = 0, \tilde v = \tilde \omega_i\\
          \tilde r &= \eta^{-1} &:~& \tilde u = \tilde w = 0, \tilde v = \tilde \omega_o \eta\\
          \tilde z &= 0, L / r_i &:~& \tilde u = \tilde v = \tilde w = 0

Here length is scaled by $r_i$, $u = \\tilde u v_i$, $t = \\tilde t r_i / v_i$, $p = \\tilde p \\rho_0 v_i^2 - \\rho_0gz$, $\\omega = \\tilde \\omega v_i / r_i$. Moreover, the Taylor number is given by $\\mathrm{Ta} = v_i r_i / \\nu = \\mathrm{Re}_i r_i / d = \\mathrm{Re}_i (\\eta^{-1} - 1)^{-1}$ with $\\eta = r_i / r_o$. All other quantities are defined in :ref:`symbols`.

Parameters
----------
These are the relevant parameters in the ``parameters`` dictionary for this problem type.

===================== ================== =====
Parameter name        Default value      Notes
===================== ================== =====
``'Problem Type'``                       Set to ``'Taylor-Couette'``
``'Reynolds Number'`` 1.0                Unused if Ta is defined
``'Taylor Number'``   Re / (1 / eta - 1)
===================== ================== =====
