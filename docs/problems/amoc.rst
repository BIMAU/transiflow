2D Atlantic Meridional Ocean Circulation
========================================
This describes a rectangular 2D model of the Atlantic meridional ocean circulation (AMOC).

Governing Equations
-------------------
In a domain of dimensions $L \\times D$, the governing equations are given by

.. math:: \frac{\partial \mathbf{u}}{\partial t} + (\mathbf{u} \cdot \nabla) \mathbf{u} &= -\frac{1}{\rho_0} \nabla p + \nu \nabla^2 \mathbf{u} - \frac{\rho g}{\rho_0} e_z\\
          \nabla \cdot \mathbf{u} &= 0\\
          \frac{\partial T}{\partial t} + (\mathbf{u} \cdot \nabla) T &= \kappa_T \nabla^2 T\\
          \frac{\partial S}{\partial t} + (\mathbf{u} \cdot \nabla) S &= \kappa_S \nabla^2 S

with boundary conditions

.. math:: x &= 0, L &:~& u = w = \frac{\partial T}{\partial x} = \frac{\partial S}{\partial x} = 0\\
          z &= 0 &:~& u = w = \frac{\partial T}{\partial z} = \frac{\partial S}{\partial z} = 0\\
          z &= D &:~& u = w = 0, T = T_S(x), \frac{\partial S}{\partial z} = \sigma Q_S(x)

where $T_S(x)$ is a prescribed temperature distribution along the surface. The parameter $\\sigma$ is the strength of the surface fresh-water flux and $Q_S(x)$ represents its spatial structure. See :ref:`symbols` for the definition of all quantities used here.

Nondimensional formulation
--------------------------
In TransiFlow, we implement the following nondimensionalized equations

.. math:: \frac{\partial \tilde{\mathbf{u}}}{\partial \tilde t} + (\tilde{\mathbf{u}} \cdot \tilde \nabla) \tilde{\mathbf{u}} &= -\tilde \nabla \tilde p +\frac{1}{\mathrm{Gr}^{1/2}}\tilde \nabla^2 \tilde{\mathbf{u}} + (\tilde T - \tilde S)e_z\\
          \tilde\nabla \cdot \tilde{\mathbf{u}} &= 0\\
          \frac{\partial \tilde T}{\partial \tilde t} + (\tilde{\mathbf{u}} \cdot \tilde\nabla) \tilde T &= \frac{1}{\mathrm{Pr} \mathrm{Gr}^{1/2}} \tilde \nabla^2 \tilde T\\
          \frac{\partial \tilde S}{\partial \tilde t} + (\tilde{\mathbf{u}} \cdot \tilde\nabla) \tilde S &= \frac{1}{\mathrm{Pr} \mathrm{Le} \mathrm{Gr}^{1/2}} \tilde \nabla^2 \tilde S

with boundary conditions

.. math:: \tilde x &= 0, A_x &:~& \tilde u = \tilde w = \frac{\partial \tilde T}{\partial \tilde x} = \frac{\partial \tilde S}{\partial \tilde x} = 0\\
          \tilde z &= 0 &:~& \tilde u = \tilde w = \frac{\partial \tilde T}{\partial \tilde z} = \frac{\partial \tilde S}{\partial \tilde z} = 0\\
          \tilde z &= 1 &:~& \tilde u = \tilde w = 0, \tilde T = \tilde T_S(x), \frac{\partial \tilde S}{\partial \tilde z} = \sigma Q_S(x)

Here $x$ and $z$ are scaled by $D$, and hence $A_x = L / D$. The other quantities are scaled using $u = \\hat u \\nu / D$, $t = \\hat t D^2 / \\nu$, $p = \\hat p (\\mu \\nu / D^2) - \\rho_0 g z$, $T = \\Delta T \\hat T$, $S = \\Delta S / \\lambda \\hat S$ with $\\lambda = \\alpha_S \\Delta S / (\\alpha_T \\Delta T)$ and additionally $\\hat u = \\tilde u \\mathrm{Gr}^{1/2}$, $\\hat t = \\tilde t \\mathrm{Gr}^{-1/2}$, $\\hat p = \\tilde p \\mathrm{Gr}$, $\\hat T = \\tilde T$, $\\hat S = \\tilde S$. Moreover, the Prandtl number is given by $\\mathrm{Pr} = \\nu / \\kappa_T$, the Rayleigh number by $\\mathrm{Ra} = (\\alpha_T g \\Delta T D^3) / (\\nu \\kappa_T)$, the Grashof number by $\\mathrm{Gr} = \\mathrm{Ra} / \\mathrm{Pr}$ and the Lewis number by $\\mathrm{Le}  = \\kappa_T / \\kappa_S$. All other quantities are defined in :ref:`symbols`.

Parameters
----------
These are the relevant parameters in the ``parameters`` dictionary for this problem type.

===================== ============= =====
Parameter name        Default value Notes
===================== ============= =====
``'Problem Type'``                  Set to ``'AMOC'``
``'Rayleigh Number'`` 1.0           Unused if Gr is defined
``'Prandtl Number'``  1.0
``'Grashof Number'``  Ra / Pr       Overrides Ra if defined
``'Lewis Number'``    1.0
===================== ============= =====
