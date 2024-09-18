Rayleigh-Bénard Convection
==========================
Rayleigh-Bénard convection describes the flow in a liquid that is heated from below.

Governing Equations
-------------------
In a domain of dimensions $L \\times B \\times D$, the governing equations for Rayleigh-Bénard convection are given by

.. math:: \frac{\partial \mathbf{u}}{\partial t} + (\mathbf{u} \cdot \nabla) \mathbf{u} &= -\frac{1}{\rho_0}\nabla p + \nu \nabla^2\mathbf{u} - \frac{\rho g}{\rho_0}e_z\\
          \nabla \cdot \mathbf{u} &= 0\\
          \frac{\partial T}{\partial t} + (\mathbf{u} \cdot \nabla) T &= \kappa_T \nabla^2T

with boundary conditions

.. math:: x &= 0, L &:~& u = v = w = \frac{\partial T}{\partial x} = 0\\
          y &= 0, B &:~& u = v = w = \frac{\partial T}{\partial y} = 0\\
          z &= 0 &:~& u = v = w = 0, T = T_B\\
          z &= D &:~& u = v = w = 0, k\frac{\partial T}{\partial z} = h(T_A-T)

where $T_A$ is the temperature of the gas just above the interface. See :ref:`symbols` for the definition of all quantities used here.

Nondimensional formulation
--------------------------
In TransiFlow, we implement the following nondimensionalized equations

.. math:: \frac{\partial \tilde{\mathbf{u}}}{\partial \tilde t} + (\tilde{\mathbf{u}} \cdot \tilde\nabla) \tilde{\mathbf{u}} &= -\tilde\nabla \tilde p +\frac{1}{\mathrm{Gr}^{1/2}}\tilde\nabla^2\tilde{\mathbf{u}} + \tilde Te_z\\
          \tilde\nabla \cdot \tilde{\mathbf{u}} &= 0\\
          \frac{\partial \tilde T}{\partial \tilde t} + (\tilde{\mathbf{u}} \cdot \tilde\nabla) \tilde T &= \frac{1}{\mathrm{Pr}\mathrm{Gr}^{1/2}}\tilde\nabla^2\tilde T

with boundary conditions

.. math:: \tilde x &= 0, A_x &:~& \tilde u = \tilde v = \tilde w = \frac{\partial \tilde T}{\partial \tilde x} = 0\\
          \tilde y &= 0, A_y &:~& \tilde u = \tilde v = \tilde w = \frac{\partial \tilde T}{\partial \tilde y} = 0\\
          \tilde z &= 0 &:~& \tilde u = \tilde v = \tilde w = 0, \tilde T = 1\\
          \tilde z &= 1 &:~& \frac{\partial \tilde u}{\partial \tilde z} = \frac{\partial \tilde v}{\partial \tilde z} = \tilde w = 0, \frac{\partial \tilde T}{\partial \tilde z} = \mathrm{Bi} \tilde T

Here $x$, $y$ and $z$ are scaled by $D$, and hence $A_x = L / D$ and $A_y = B / D$. The other quantities are scaled using $u = \\hat u \\nu / D$, $t = \\hat t D^2 / \\nu$, $p = \\hat p (\\mu \\nu / D^2) - \\rho_0 g z$, $T = (T_B - T_A) \\hat T + T_A$ and additionally $\\hat u = \\tilde u \\mathrm{Gr}^{1/2}$, $\\hat t = \\tilde t \\mathrm{Gr}^{-1/2}$, $\\hat p = \\tilde p \\mathrm{Gr}$, $\\hat T = \\tilde T$. Moreover, the Prandtl number is given by $\\mathrm{Pr} = \\nu / \\kappa_T$, the Rayleigh number by $\\mathrm{Ra} = (\\alpha_T g \\Delta T D^3) / (\\nu \\kappa_T)$, the Grashof number by $\\mathrm{Gr} = \\mathrm{Ra} / \\mathrm{Pr}$ and the Biot number is given by $\\mathrm{Bi} = h D / k$. All other quantities are defined in :ref:`symbols`.

Perturbation formulation
------------------------
Alternatively, we can also look at the perturbation from the motionless solution

.. math:: \bar T(\tilde z) = 1 - \frac{\mathrm{Bi}}{\mathrm{Bi} + 1}\tilde z

which is less prone to numerical errors. This results in the following nondimensional formulation

.. math:: \frac{\partial \tilde{\mathbf{u}}}{\partial \tilde t} + (\tilde{\mathbf{u}} \cdot \tilde\nabla) \tilde{\mathbf{u}} &= -\tilde\nabla \tilde p +\frac{1}{\mathrm{Gr}^{1/2}}\tilde\nabla^2\tilde{\mathbf{u}} + \tilde Te_z\\
          \tilde\nabla \cdot \tilde{\mathbf{u}} &= 0\\
          \frac{\partial \tilde T}{\partial \tilde t} + (\tilde{\mathbf{u}} \cdot \tilde\nabla) \tilde T &= \frac{1}{\mathrm{Pr}\mathrm{Gr}^{1/2}}\tilde\nabla^2\tilde T + \frac{\mathrm{Bi}}{\mathrm{Bi + 1}}\tilde w

with boundary conditions

.. math:: \tilde x &= 0, A_x &:~& \tilde u = \tilde v = \tilde w = \frac{\partial \tilde T}{\partial \tilde x} = 0\\
          \tilde y &= 0, A_y &:~& \tilde u = \tilde v = \tilde w = \frac{\partial \tilde T}{\partial \tilde y} = 0\\
          \tilde z &= 0 &:~& \tilde u = \tilde v = \tilde w = \tilde T = 0\\
          \tilde z &= 1 &:~& \frac{\partial \tilde u}{\partial \tilde z} = \frac{\partial \tilde v}{\partial \tilde z} = \tilde w = 0, \frac{\partial \tilde T}{\partial \tilde z} = \mathrm{Bi} \tilde T


Parameters
----------
These are the relevant parameters in the ``parameters`` dictionary for this problem type.

===================== ============= =====
Parameter name        Default value Notes
===================== ============= =====
``'Problem Type'``                  | Set to ``'Rayleigh-Benard'`` or
                                    | ``'Rayleigh-Benard Perturbation'``
``'Rayleigh Number'`` 1.0           Unused if Gr is defined
``'Prandtl Number'``  1.0
``'Grashof Number'``  Ra / Pr       Overrides Ra if defined
``'Biot Number'``     0.0
===================== ============= =====
