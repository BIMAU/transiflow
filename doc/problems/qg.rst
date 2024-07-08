Double-gyre wind-driven circulation
===================================
TransiFlow also implements the idealized barotropic quasi-geostrophic (QG) model of the double-gyre (DG) wind-driven circulation.

Governing Equations
-------------------
Consider a rectangular ocean basin of size $L \\times L \\times D$ situated on a midlatitube $\\beta$-plane with central latitude $\\theta_0 = 45^\\circ \\mathrm{N}$ and Coriolis parameter $f_0=2 \\Omega \\sin{\\theta_0}$, where $\\Omega$ is the rotation rate of the Earth.
The flow is forced at the surface through a wind-stress vector $(\\tau^x(x, y), \\tau^y(x, y))$.
Flows in the basin are governed by the barotropic vorticity equation (with the streamfunction $\\psi$ and the vertical component of the vorticity $\\zeta$) is 

.. math:: \left(\frac{\partial}{\partial t} + \frac{\partial \psi}{\partial x} \frac{\partial}{\partial y} - \frac{\partial \psi}{\partial y} \frac{\partial}{\partial x}\right) (\zeta - \lambda_0 \psi) + \beta_0 \frac{\partial \psi}{\partial x} &= \frac{1}{\rho D} \left(\frac{\partial \tau^y}{\partial x} - \frac{\partial \tau^x}{\partial y}\right) - \epsilon_0 \zeta + A_H \nabla^2 \zeta\\
          \zeta &= \nabla^2 \psi

with boundary conditions

.. math:: x &= 0, L &:~& \psi = \frac{\partial \psi}{\partial x} = 0\\
          y &= 0, L &:~& \psi = \zeta = 0

where $\\epsilon_0$ is a damping coefficient, $\\lambda_0 = f_0^2 / (g D)$ and $A_H$ is the lateral friction coefficient.

Nondimensional formulation
--------------------------
The above equations can be non-dimensionalized as

.. math:: \left(\frac{\partial}{\partial \tilde t} + \tilde u \frac{\partial}{\partial \tilde x} + \tilde v \frac{\partial}{\partial \tilde y}\right) (\tilde \zeta - F \tilde \psi) + \beta \frac{\partial \tilde \psi}{\partial \tilde x} &= \alpha \left(\frac{\partial \tilde \tau^y}{\partial \tilde x} - \frac{\partial \tilde \tau^x}{\partial \tilde y}\right) - r_0 \tilde \zeta + \frac{1}{\mathrm{Re}} \tilde \nabla^2 \tilde \zeta\\
          \tilde \zeta &= \tilde \nabla^2 \tilde \psi

with boundary conditions

.. math:: \tilde x &= 0, 1 &:~& \tilde \psi = \frac{\partial \tilde \psi}{\partial \tilde x} = 0\\
          \tilde y &= 0, 1 &:~& \tilde \psi = \tilde \zeta = 0

Here $x$, and $y$ are scaled by $L$, $\\psi = \\tilde \\psi U L$, $\\zeta = \\tilde \\zeta U / L$, $t = \\tilde t L / U$, $\\tau = \\tilde \\tau \\tau_0$. Moreover, $u = -\\partial \\psi / \\partial y = -U \\partial \\tilde \\psi / \\partial \\tilde y = \\tilde u U$ and $v = \\partial \\psi / \\partial x = U \\partial \\tilde \\psi / \\partial \\tilde x = \\tilde v U$. We also define the parameters

.. math:: Re = \frac{U L}{A_H} ~;~ r_0 = \frac{\epsilon_0 L}{U} ~;~ \beta = \frac{\beta_0 L^2}{U} ~;~ \alpha = \frac{\tau_0 L}{\rho D U^2} ~;~ F = \frac{f^2_0 L^2}{g D}

The wind-stress forcing is prescribed as

.. math:: \tilde \tau^x(\tilde x, \tilde y) &= \frac{-1}{2 \pi} \cos{(2 \pi \tilde y)}\\
          \tilde \tau^y(\tilde x, \tilde y) &= 0

All other quantities are defined in :ref:`symbols`.

Velocity-pressure formulation
-----------------------------
Taking $F = r_0 = 0$, the equations we implement in TransiFlow are formulated using the velocity $\\mathbf{u} = (u, v)$ and pressure $p$ as follows

.. math:: \frac{\partial \tilde{\mathbf{u}}}{\partial \tilde t} + \tilde{\mathbf{u}} \cdot \tilde \nabla \tilde{\mathbf{u}} + \beta e_z \times \tilde{\mathbf{u}} &= \alpha \left(\frac{\partial \tilde \tau^y}{\partial \tilde x} - \frac{\partial \tilde \tau^x}{\partial \tilde y}\right) - \tilde \nabla p + \frac{1}{\mathrm{Re}} \tilde \nabla^2 \tilde{\mathbf{u}}\\
          \tilde \nabla \cdot \tilde{\mathbf{u}} &= 0

with boundary conditions

.. math:: \tilde x &= 0, 1 &:~& \tilde u = \tilde v = 0\\
          \tilde y &= 0, 1 &:~& \frac{\partial \tilde u}{\partial \tilde y} = 0, \tilde v = 0

Parameters
----------
These are the relevant parameters in the ``parameters`` dictionary for this problem type.

=========================== ============= =====
Parameter name              Default value Notes
=========================== ============= =====
``'Problem Type'``                        Set to ``'Double Gyre'``
``'Asymmetry Parameter'``   0.0           Used to get switch branches
``'Reynolds Number'``       1.0
``'Rossby Parameter'``      0.0           $\\beta$
``'Wind Stress Parameter'`` 0.0           $\\alpha$
=========================== ============= =====
