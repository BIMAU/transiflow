import numpy
from petsc4py import PETSc

import fvm


class Vector(PETSc.Vec):
    """Distributed Epetra_Vector with some extra methods added to it for convenience."""

    @staticmethod
    def from_array(x):
        vec = Vector().createMPI(len(x))
        vec.setArray(x)
        return vec

    size = property(PETSc.Vec.getSize)

    @property
    def shape(self):
        return (self.size,)


def ind2sub(nx, ny, nz, idx, dof=1):
    rem = idx
    var = rem % dof
    rem = rem // dof
    i = rem % nx
    rem = rem // nx
    j = rem % ny
    rem = rem // ny
    k = rem % nz
    return (i, j, k, var)


def sub2ind(nx, ny, nz, dof, i, j, k, var):
    return ((k * ny + j) * nx + i) * dof + var


def set_default_parameter(parameterlist, name, value):
    if name not in parameterlist:
        parameterlist[name] = value

    return parameterlist[name]


def get_local_coordinate_vector(x, nx_offset, nx_local):
    x = numpy.roll(x, 2)
    x = x[nx_offset : nx_offset + nx_local + 3]
    return numpy.roll(x, -2)


class Interface(fvm.Interface):
    """This class defines an interface to the PETSc backend for the
    discretization. We use this so we can write higher level methods
    such as pseudo-arclength continuation without knowing anything
    about the underlying methods such as the solvers that are present
    in the backend we are interfacing with.

    The PETSc backend partitions the domain into Cartesian subdomains,
    while solving linear systems on skew Cartesian subdomains to deal
    with the C-grid discretization. The subdomains will be distributed
    over multiple processors if MPI is used to run the application."""

    def __init__(self, comm, parameters, nx, ny, nz, dim, dof):
        fvm.Interface.__init__(self, parameters, nx, ny, nz, dim, dof)

        self.jac = None

        self.nx_global = nx
        self.ny_global = ny
        self.nz_global = nz
        self.dof = dof

        self.comm = comm

        self.parameters = parameters

    def rhs(self, state):
        """Right-hand side in M * du / dt = F(u) defined on the
        non-overlapping discretization domain map."""

        rhs = fvm.Interface.rhs(self, state.getArray())
        rhs = Vector.from_array(rhs)

        return rhs

    def jacobian(self, state):
        """Jacobian J of F in M * du / dt = F(u) defined on the
        domain map used by PETSc."""

        local_jac = fvm.Interface.jacobian(self, state.getArray())

        if self.jac is None:
            self.jac = PETSc.Mat().createAIJ((state.size, state.size), comm=self.comm)
            self.jac.setUp()
        else:
            self.jac.zeroEntries()

        self.jac.setValuesCSR(
            numpy.array(local_jac.begA, dtype=numpy.int32),
            numpy.array(local_jac.jcoA[: local_jac.begA[-1]], dtype=numpy.int32),
            local_jac.coA[: local_jac.begA[-1]],
        )

        self.jac.assemble()

        return self.jac

    def mass_matrix(self):
        """Mass matrix M in M * du / dt = F(u) defined on the
        domain map used by HYMLS."""
        return None

    def solve(self, jac, rhs):
        """Currently unused direct solver that was used for testing."""

        ksp = PETSc.KSP().create()
        ksp.setType("preonly")
        pc = ksp.getPC()
        pc.setType("lu")
        pc.setFactorSolverType("mumps")

        ksp.setFromOptions()

        ksp.setOperators(jac)
        x = rhs.copy()

        ksp.solve(rhs, x)
        return x

    def eigs(self, state, return_eigenvectors=False, enable_recycling=False):
        """Compute the generalized eigenvalues of beta * J(x) * v = alpha * M * v."""
        raise NotImplementedError()
