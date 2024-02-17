import numpy
import petsc4py
from petsc4py import PETSc

from transiflow.interface import ParallelBaseInterface

petsc4py.init()


class Vector(PETSc.Vec):
    """Distributed ghosted PETSc Vector with some extra methods added to it for convenience."""

    @staticmethod
    def from_array(m, x, ghosts=(), comm=PETSc.COMM_WORLD):
        vec = Vector().createGhost(ghosts, len(x), comm=comm)
        vec.setLGMap(m)
        with vec.localForm() as lf:
            lf.set(0.0)
        vec.setValues(range(*vec.getOwnershipRange()), x[m.getIndices()])
        vec.assemble()
        vec.ghostUpdate(PETSc.InsertMode.INSERT, PETSc.ScatterMode.FORWARD)
        return vec

    size = property(PETSc.Vec.getSize)

    @property
    def shape(self):
        return (self.size,)


class Interface(ParallelBaseInterface):
    """This class defines an interface to the PETSc backend for the
    discretization. We use this so we can write higher level methods
    such as pseudo-arclength continuation without knowing anything
    about the underlying methods such as the solvers that are present
    in the backend we are interfacing with.

    The PETSc backend partitions the domain into Cartesian subdomains,
    while solving linear systems on skew Cartesian subdomains to deal
    with the C-grid discretization. The subdomains will be distributed
    over multiple processors if MPI is used to run the application."""

    def __init__(self, parameters, nx, ny, nz, dim, dof, comm=PETSc.COMM_WORLD):
        super().__init__(comm, parameters, nx, ny, nz, dim, dof)

        self.size_global = nx * ny * nz * dof

        self.map = self.create_map()
        self.assembly_map = self.create_map(True)
        self.ghosts = list(set(self.assembly_map.indices).difference(self.map.indices))

        self.jac = None

    def vector(self):
        vec = Vector().createGhost(self.ghosts, self.size_global, comm=self.comm)
        vec.setLGMap(self.map)
        return vec

    def vector_from_array(self, array):
        return Vector.from_array(self.map, array, self.ghosts)

    def create_map(self, overlapping=False):
        """Create a PETSc map on which the local discretization domain is defined.
        The overlapping part is only used for computing the discretization."""

        local_elements = ParallelBaseInterface.create_map(self, overlapping)
        return PETSc.LGMap().create(local_elements, bsize=None, comm=self.comm)

    def rhs(self, state):
        """Right-hand side in M * du / dt = F(u) defined on the
        non-overlapping discretization domain map."""

        state.ghostUpdate(PETSc.InsertMode.INSERT, PETSc.ScatterMode.FORWARD)
        rhs = state.duplicate()
        with rhs.localForm() as local_rhs, state.localForm() as local_state:
            # all entries after n = rhs.size are ghost points
            # permute local_state such that indices are in correct order (sorted, ascending)
            indices = numpy.r_[self.map.indices, numpy.array(self.ghosts, dtype=int)]
            ind_permute = numpy.argsort(indices)
            local_arr = numpy.empty_like(local_rhs.array)
            local_arr[ind_permute] = self.discretization.rhs(local_state.array[ind_permute])
            local_rhs.setArray(local_arr)

        rhs.assemble()

        return rhs

    def jacobian(self, state):
        """Jacobian J of F in M * du / dt = F(u) defined on the
        domain map used by PETSc."""

        with state.localForm() as lf:
            indices = numpy.r_[self.map.indices, numpy.array(self.ghosts, dtype=int)]
            ind_permute = numpy.argsort(indices)
            local_jac = self.discretization.jacobian(lf.array[ind_permute])

        if self.jac is None:
            self.jac = PETSc.Mat().create(comm=self.comm)
            self.jac.setType("aij")
            N = state.size
            self.jac.setSizes((N, N))
            self.jac.setLGMap(self.map)
            self.jac.setUp()
        else:
            self.jac.zeroEntries()

        self.jac.assemblyBegin()
        for i in range(len(local_jac.begA) - 1):
            if self.is_ghost(i):
                continue

            col_idx = numpy.array(
                local_jac.jcoA[local_jac.begA[i]: local_jac.begA[i + 1]],
                dtype=PETSc.IntType,
            )
            values = local_jac.coA[local_jac.begA[i]: local_jac.begA[i + 1]]
            self.jac.setValues(i, col_idx, values)
        self.jac.assemblyEnd()

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
