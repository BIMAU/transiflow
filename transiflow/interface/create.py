def _icmp(first, second):
    return first.lower() == second.lower()

def create(parameters, nx, ny, nz=1, dim=None, dof=None,
           boundary_conditions=None, backend='SciPy'):
    '''Helper function to create an interface with a certain backend.
    This functions is aliased as ``Interface``. It can be called as

    .. code-block:: Python

        from transiflow import Interface
        interface = Interface(parameters, nx, ny)

    Parameters
    ----------
    parameters : dict
        Key-value pairs describing parameters of the model, for
        instance the Renolds number and the problem type. Possible
        values can be found in :ref:`problem definitions`.
    nx : int
        Grid size in the x direction.
    ny : int
        Grid size in the y direction.
    nz : int, optional
        Grid size in the z direction. 1 for 2-dimensional problems.
        This is the default.
    dim : int, optional
        Physical dimension of the problem. In case this is set to 2, w
        is not referenced in the state vector. The default is based on
        the value of nz.
    dof : int, optional
        Degrees of freedom for this problem. This should be set to dim
        plus 1 for each of pressure, temperature and salinity, if they
        are required for your problem. For example a 3D differentially
        heated cavity has dof = 3 + 1 + 1 = 5.
    boundary_conditions : function, optional
        User-supplied function that implements the boundary
        conditions. It is called as ``boundary_conditions(bc, atom)``
        where ``bc`` is an instance of the :class:`.BoundaryConditions`
        class.
    backend : str, optional
        The backend to use. Can be ``Epetra``, ``HYMLS``, ``PETSc``, ``SciPy``.

    Returns
    -------
    interface : Interface
        An interface instance.

    '''

    if _icmp(backend, 'Epetra'):
        from .Epetra import Interface
        return Interface(parameters, nx, ny, nz, dim, dof,
                         boundary_conditions=boundary_conditions)

    if _icmp(backend, 'HYMLS'):
        from .HYMLS import Interface
        return Interface(parameters, nx, ny, nz, dim, dof,
                         boundary_conditions=boundary_conditions)

    if _icmp(backend, 'PETSc'):
        from .PETSc import Interface
        return Interface(parameters, nx, ny, nz, dim, dof,
                         boundary_conditions=boundary_conditions)

    from .SciPy import Interface
    return Interface(parameters, nx, ny, nz, dim, dof,
                     boundary_conditions=boundary_conditions)
