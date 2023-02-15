from .CrsMatrix import CrsMatrix
from .BoundaryConditions import BoundaryConditions
from .Discretization import Discretization
from .CylindricalDiscretization import CylindricalDiscretization
from .Continuation import Continuation
from .TimeIntegration import TimeIntegration

__all__ = [
    'CrsMatrix', 'BoundaryConditions', 'Discretization', 'CylindricalDiscretization',
    'Continuation', 'TimeIntegration'
]

# Backward compatibility
from .interface.SciPy import Interface # noqa: F401
from .interface import SciPy as SciPyInterface # noqa: F401

__all__.extend(['Interface', 'SciPyInterface'])

try:
    from .interface import Epetra as EpetraInterface # noqa: F401
    __all__.extend(['EpetraInterface'])
except ImportError:
    pass

try:
    from .interface import HYMLS as HYMLSInterface # noqa: F401
    __all__.extend(['HYMLSInterface'])
except ImportError:
    pass

try:
    from .interface import PETSc as PETScInterface # noqa: F401
    __all__.extend(['PETScInterface'])
except ImportError:
    pass
