Adding a custom boundary condition
==================================
TransiFlow implements many pre-defined problem types, but you can easily add your own using a custom boundary condition.
An example would be a 2D lid-driven cavity with a moving lid on the bottom instead of the top.
For the purpose of this example, we code the moving lid manually, instead of using the provided ``moving_lid_south`` method.
Say we have a lid that moves with velocity $U$, then at the boundary $(u_j + u_{j-1}) / 2 = U$, $v_j = 0$.
So we replace $u_{j-1}$ by $2U - u_j$ and then apply the no slip condition.
We do this by subtracting the part of ``atom`` that would be applied to $u_{j-1}$ from $u_j$, and adding $2U$ times the part of ``atom`` that would be applied to $u_{j-1}$ to the forcing.

.. code-block:: Python

    def boundaries(boundary_conditions, atom):
        lid_velocity = parameters['Lid Velocity']
        boundary_conditions.frc[0, :, :, 1] += \
            2 * lid_velocity * numpy.sum(atom[0, :, :, 1, 1, 0, :, :], axis=(-1, -2))

        atom[0, :, :, :, 1, 1, :, :] -= atom[0, :, :, :, 1, 0, :, :]
        atom[0, :, :, :, 1, 0, :, :] = 0

        boundary_conditions.no_slip_east(atom)
        boundary_conditions.no_slip_west(atom)

        boundary_conditions.no_slip_north(atom)
        boundary_conditions.no_slip_south(atom)

        return boundary_conditions.get_forcing()

    interface = Interface(parameters, nx, ny, boundary_conditions=boundaries)

The arguments for the ``boundaries`` function are a :class:`.BoundaryConditions` object and an ``atom`` object as described in the :class:`.Discretization` class.

..
    Explicitly enable math mode
.. math::
