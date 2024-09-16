import numpy
import petsc4py
from petsc4py import PETSc

from transiflow.interface import ParallelBaseInterface

petsc4py.init()


class Vector(PETSc.Vec):
    """Distributed ghosted PETSc Vector with some extra methods added to it for convenience."""

    @staticmethod
    def from_array(m, x, ghosts=(), comm=PETSc.COMM_WORLD):
        size = len(m.indices)
        vec = Vector().createGhost(ghosts, (size, None), comm=comm)
        vec.setValues(m.indices, x)
        vec.assemble()
        vec.ghostUpdate(addv=PETSc.InsertMode.INSERT_VALUES, mode=PETSc.ScatterMode.FORWARD)
        return vec

    size = property(PETSc.Vec.getSize)

    def gather(self):
        sct, vector_global = PETSc.Scatter().toZero(self)
        sct.scatter(self, vector_global)
        return vector_global

    def all_gather(self):
        sct, vector_global = PETSc.Scatter().toAll(self)
        sct.scatter(self, vector_global)
        return vector_global

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
    over multiple processors if MPI is used to run the application.

    PETSc parallel vectors (and matrices) are distributed such that each process owns a
    contiguous range of indices (rows) without holes. The natural ordering is therefore
    mapped to PETSc's ordering.
    """

    def __init__(self, parameters, nx, ny, nz=1, dim=None, dof=None, comm=PETSc.COMM_WORLD):
        super().__init__(comm, parameters, nx, ny, nz, dim, dof)

        self.size_global = nx * ny * nz * self.dof

        self.map_natural = self.create_map()
        self.assembly_map = self.create_map(True)

        ghosts_natural = list(
            set(self.assembly_map.indices).difference(self.map_natural.indices)
        )

        indices_natural = Vector().createWithArray(self.map_natural.indices)
        sct, indices_natural_global = PETSc.Scatter().toAll(indices_natural)
        sct.scatter(indices_natural, indices_natural_global)
        self.index_natural_global_permut = numpy.argsort(indices_natural_global)

        self.ghosts = self.ghosts_petsc(ghosts_natural, indices_natural_global)

        self.index_ordering = PETSc.AO().createMapping(self.map_natural.indices)
        self.map = self.create_map_petsc()

        self.index_natural_local_permut = numpy.argsort(
            numpy.r_[self.map_natural.indices, ghosts_natural]
        )

        self.index_ordering_assembly = PETSc.AO().createMapping(
            numpy.r_[self.map_natural.indices, ghosts_natural].astype(PETSc.IntType),
            numpy.r_[self.map.indices, self.ghosts].astype(PETSc.IntType),
        )

        self.jac = None

    def vector(self):
        return Vector().createGhost(self.ghosts, (len(self.map.indices), None), comm=self.comm)

    def vector_from_array(self, array):
        return Vector.from_array(self.map, array[self.map_natural.indices], ghosts=self.ghosts)

    def array_from_vector(self, vector):
        return Vector.all_gather(vector).array[self.index_natural_global_permut]

    def ghosts_petsc(self, ghosts_natural, indices_natural_global):
        """Get global indices of ghosts in PETSc ordering."""
        ghosts_petsc = []
        for ghost in ghosts_natural:
            ghosts_petsc.append(numpy.where(indices_natural_global.array == ghost)[0][0])

        return ghosts_petsc

    def create_map_petsc(self):
        """Create a PETSc index map, using PETSc's ordering (contiguous row ownership),
        from the naturally ordered indices."""
        return PETSc.LGMap().create(
            self.index_ordering.app2petsc(self.map_natural.indices), bsize=None, comm=self.comm
        )

    def create_map(self, overlapping=False):
        """Create a PETSc map on which the local discretization domain is defined.
        The overlapping part is only used for computing the discretization."""
        local_elements = ParallelBaseInterface.create_map(self, overlapping)
        return PETSc.LGMap().create(local_elements, bsize=None, comm=self.comm)

    def rhs(self, state):
        """Right-hand side in M * du / dt = F(u) defined on the
        non-overlapping discretization domain map."""

        state.ghostUpdate(addv=PETSc.InsertMode.INSERT_VALUES, mode=PETSc.ScatterMode.FORWARD)
        ind_local_nat = self.index_natural_local_permut

        with state.localForm() as lf:
            local_rhs = self.discretization.rhs(lf.array[ind_local_nat])

        rhs_vec = state.duplicate()

        with rhs_vec.localForm() as lf:
            lf.array[ind_local_nat] = local_rhs

        return rhs_vec

    def jacobian(self, state):
        """Jacobian J of F in M * du / dt = F(u) defined on the
        domain map used by PETSc."""

        ind_local_nat = self.index_natural_local_permut

        state.ghostUpdate(addv=PETSc.InsertMode.INSERT_VALUES, mode=PETSc.ScatterMode.FORWARD)

        with state.localForm() as lf:
            local_jac = self.discretization.jacobian(lf.array[ind_local_nat])

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
        for i_loc in range(len(local_jac.begA) - 1):

            if self.is_ghost(i_loc):
                continue

            col_idx_nat = self.assembly_map.indices[
                local_jac.jcoA[local_jac.begA[i_loc]: local_jac.begA[i_loc + 1]]
            ]
            values = local_jac.coA[local_jac.begA[i_loc]: local_jac.begA[i_loc + 1]]

            i = self.index_ordering.app2petsc(self.assembly_map.indices[i_loc])
            col_idx = self.index_ordering_assembly.app2petsc(col_idx_nat)
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

        dof = self.dim
        self.set_dirichlet_bc(jac, rhs, dof, 0.0)

        ksp.setOperators(jac)
        x = self.vector()

        ksp.solve(rhs, x)

        x.ghostUpdate(addv=PETSc.InsertMode.INSERT_VALUES, mode=PETSc.ScatterMode.FORWARD)
        return x

    def set_dirichlet_bc(self, A, b, i, value):
        """Set Dirichlet BC to specified value at index i.
        Single values or sequences of indices and values are accepted.
        """
        A.zeroRows(i)
        b.setValues(i, value)
        b.assemble()
        b.ghostUpdate(addv=PETSc.InsertMode.INSERT_VALUES, mode=PETSc.ScatterMode.FORWARD)

    def eigs(self, state, return_eigenvectors=False, enable_recycling=False):
        """Compute the generalized eigenvalues of beta * J(x) * v = alpha * M * v."""
        raise NotImplementedError()
