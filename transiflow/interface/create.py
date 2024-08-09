def _icmp(first, second):
    return first.lower() == second.lower()

def create(parameters, nx, ny, nz=1, dim=None, dof=None, backend='SciPy'):
    if _icmp(backend, 'Epetra'):
        from .Epetra import Interface
        return Interface(parameters, nx, ny, nz, dim, dof)

    if _icmp(backend, 'HYMLS'):
        from .HYMLS import Interface
        return Interface(parameters, nx, ny, nz, dim, dof)

    if _icmp(backend, 'PETSc'):
        from .PETSc import Interface
        return Interface(parameters, nx, ny, nz, dim, dof)

    from .SciPy import Interface
    return Interface(parameters, nx, ny, nz, dim, dof)
