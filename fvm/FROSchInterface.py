from PyTrilinos import Epetra

import sys
import os
import numpy

import fvm

from fvm.HYMLSInterface import Vector, ind2sub, sub2ind, convert_parameters, set_default_parameter

import FROSch

class Interface(fvm.Interface):
    '''This class defines an interface to the FROSch backend for the
    discretization. We use this so we can write higher level methods
    such as pseudo-arclength continuation without knowing anything
    about the underlying methods such as the solvers that are present
    in the backend we are interfacing with.'''

    def __init__(self, comm, parameters, nx, ny, nz, dim, dof, x=None, y=None, z=None):
        fvm.Interface.__init__(self, parameters, nx, ny, nz, dim, dof)

        self.nx_global = nx
        self.ny_global = ny
        self.nz_global = nz
        self.dof = dof

        self.comm = comm

        # disable Python output on procs other than 0:
        if self.comm.MyPID() != 0:
            self._original_stdout = sys.stdout
            print('PID %d will now disable output to stdout' % (self.comm.MyPID()))
            sys.stdout = open(os.devnull, 'w')

        self.parameters = parameters
        self.teuchos_parameters = self.get_teuchos_parameters()

        self.partition_domain()
        self.map = self.create_map()

        self.assembly_map = self.create_map(True)
        self.assembly_importer = Epetra.Import(self.assembly_map, self.map)

        # FIXME: Map that FROSch uses for solving
        self.solve_map = self.map
        self.solve_importer = Epetra.Import(self.solve_map, self.map)

        # Create local coordinate vectors
        x_length = self.parameters.get('X-max', 1.0) - self.parameters.get('X-min', 0.0)
        x_start = self.parameters.get('X-min', 0.0) + self.nx_offset / self.nx_global * x_length
        x_end = x_start + self.nx_local / self.nx_global * x_length

        y_length = self.parameters.get('Y-max', 1.0) - self.parameters.get('Y-min', 0.0)
        y_start = self.parameters.get('Y-min', 0.0) + self.ny_offset / self.ny_global * y_length
        y_end = y_start + self.ny_local / self.ny_global * y_length

        z_length = self.parameters.get('Z-max', 1.0) - self.parameters.get('Z-min', 0.0)
        z_start = self.parameters.get('Z-min', 0.0) + self.nz_offset / self.nz_global * z_length
        z_end = z_start + self.nz_local / self.nz_global * z_length

        if self.parameters.get('Grid Stretching', False) or self.teuchos_parameters.isParameter('Grid Stretching Factor'):
            sigma = self.parameters.get('Grid Stretching Factor', 1.5)
            x = fvm.utils.create_stretched_coordinate_vector(x_start, x_end, self.nx_local, sigma) if x is None else x
            y = fvm.utils.create_stretched_coordinate_vector(y_start, y_end, self.ny_local, sigma) if y is None else y
            z = fvm.utils.create_stretched_coordinate_vector(z_start, z_end, self.nz_local, sigma) if z is None else z
        else:
            x = fvm.utils.create_uniform_coordinate_vector(x_start, x_end, self.nx_local) if x is None else x
            y = fvm.utils.create_uniform_coordinate_vector(y_start, y_end, self.ny_local) if y is None else y
            z = fvm.utils.create_uniform_coordinate_vector(z_start, z_end, self.nz_local) if z is None else z

        # Re-initialize the fvm.Interface parameters
        self.nx = self.nx_local
        self.ny = self.ny_local
        self.nz = self.nz_local
        self.discretization = fvm.Discretization(self.parameters, self.nx_local, self.ny_local, self.nz_local,
                                                 self.dim, self.dof, x, y, z)

        self.left_scaling = None
        self.inv_left_scaling = None
        self.right_scaling = None
        self.inv_right_scaling = None

        self.jac = None
        self.mass = None
        self.initialize()

    def __del__(self):
        if self.comm.MyPID() != 0:
            print('PID %d will now re-enable output to stdout' % (self.comm.MyPID()))
            sys.stdout.close()
            sys.stdout = self._original_stdout

    def get_teuchos_parameters(self):
        teuchos_parameters = convert_parameters(self.parameters)

        # HYMLS::Solver parameters
        solver_parameters = teuchos_parameters.sublist('Solver')
        solver_parameters.set('Initial Vector', 'Zero')
        solver_parameters.set('Left or Right Preconditioning', 'Right')

        iterative_solver_parameters = solver_parameters.sublist('Iterative Solver')
        iterative_solver_parameters.set('Output Stream', 0)
        set_default_parameter(iterative_solver_parameters, 'Maximum Iterations', 1000)
        set_default_parameter(iterative_solver_parameters, 'Maximum Restarts', 20)
        set_default_parameter(iterative_solver_parameters, 'Num Blocks', 100)
        set_default_parameter(iterative_solver_parameters, 'Flexible Gmres', False)
        set_default_parameter(iterative_solver_parameters, 'Convergence Tolerance', 1e-8)
        set_default_parameter(iterative_solver_parameters, 'Output Frequency', 1)
        set_default_parameter(iterative_solver_parameters, 'Show Maximum Residual Norm Only', False)

        # FIXME: Set default FROSch parameters
        preconditioner_parameters = teuchos_parameters.sublist('Preconditioner')

        set_default_parameter(teuchos_parameters, 'Dimension', self.dim)
        set_default_parameter(preconditioner_parameters, 'Use Offset', False)

        set_default_parameter(preconditioner_parameters, 'OverlappingOperator Type', 'AlgebraicOverlappingOperator')
        overlappigoperator_parameters = preconditioner_parameters.sublist('AlgebraicOverlappingOperator')

        set_default_parameter(teuchos_parameters, 'CoarseOperator Type', 'IPOUHarmonicCoarseOperator')
        coarseoperator_parameters = preconditioner_parameters.sublist('IPOUHarmonicCoarseOperator')
        set_default_parameter(coarseoperator_parameters, 'Reuse: Coarse Basis' , True)

        preconditioner_parameters.updateParametersFromXmlFile('frosch_preconditioner.xml');

        return teuchos_parameters

    def get_teuchos_parameters(self):
        teuchos_parameters = convert_parameters(self.parameters)

        # HYMLS::Solver parameters
        solver_parameters = teuchos_parameters.sublist('Solver')
        solver_parameters.set('Initial Vector', 'Zero')
        solver_parameters.set('Left or Right Preconditioning', 'Right')

        iterative_solver_parameters = solver_parameters.sublist('Iterative Solver')
        iterative_solver_parameters.set('Output Stream', 0)
        set_default_parameter(iterative_solver_parameters, 'Maximum Iterations', 1000)
        set_default_parameter(iterative_solver_parameters, 'Maximum Restarts', 20)
        set_default_parameter(iterative_solver_parameters, 'Num Blocks', 100)
        set_default_parameter(iterative_solver_parameters, 'Flexible Gmres', False)
        set_default_parameter(iterative_solver_parameters, 'Convergence Tolerance', 1e-8)
        set_default_parameter(iterative_solver_parameters, 'Output Frequency', 1)
        set_default_parameter(iterative_solver_parameters, 'Show Maximum Residual Norm Only', False)

        # FIXME: Set default FROSch parameters
        preconditioner_parameters = teuchos_parameters.sublist('Preconditioner')

        set_default_parameter(preconditioner_parameters, 'Use Offset', False)

        set_default_parameter(preconditioner_parameters, 'OverlappingOperator Type', 'AlgebraicOverlappingOperator')
        overlappigoperator_parameters = preconditioner_parameters.sublist('AlgebraicOverlappingOperator')

        set_default_parameter(teuchos_parameters, 'CoarseOperator Type', 'IPOUHarmonicCoarseOperator')
        coarseoperator_parameters = preconditioner_parameters.sublist('IPOUHarmonicCoarseOperator')
        set_default_parameter(coarseoperator_parameters, 'Reuse: Coarse Basis' , True)

        preconditioner_parameters.updateParametersFromXmlFile('frosch_preconditioner.xml');

        return teuchos_parameters

    def unset_parameter(self, name, original_parameters):
        '''Set a parameter in self.parameters back to its original value. '''

        if name in original_parameters:
            self.set_parameter(name, original_parameters[name])
            return

        if name in self.parameters:
            if hasattr(self.parameters, 'remove'):
                self.parameters.remove(name)
            else:
                del self.parameters[name]

    def initialize(self):
        '''Initialize the Jacobian and the preconditioner, but make sure the
        nonlinear part is also nonzero so we can replace all values
        later, rather than insert them.'''

        # Backup the original parameters and put model parameters to 1
        parameter_names = ['Reynolds Number', 'Rayleigh Number',
                           'Prandtl Number', 'Rossby Parameter']

        original_parameters = {}
        for i in parameter_names:
            if i in self.parameters:
                original_parameters[i] = self.parameters[i]

            self.set_parameter(i, numpy.random.random())

        # Generate a Jacobian with a random state
        x = Vector(self.map)
        x.Random()
        self.jacobian(x)

        preconditioner_parameters = self.teuchos_parameters.sublist('Preconditioner')
        self.preconditioner = FROSch.IfpackPreconditioner(self.jac, preconditioner_parameters)

        u_map = self.create_dof_map(0, True)
        v_map = self.create_dof_map(1, True)
        # solver_parameters.set('u_map', u_map)
        # solver_parameters.set('v_map', v_map)

        if self.dim == 3:
            w_map = self.create_dof_map(2, True)
            p_map = self.create_dof_map(3, False)
            repeated_velocity_map = self.create_repeated_map([u_map, v_map, w_map])
            # solver_parameters.set('w_map', w_map)
            # solver_parameters.set('p_map', p_map)
            # solver_parameters.set('repeated_velocity_map', repeated_velocity_map)
            self.preconditioner.InitializeNew(repeated_velocity_map,u_map,v_map,w_map,p_map)
        else:
            p_map = self.create_dof_map(2, False)
            repeated_velocity_map = self.create_repeated_map([u_map, v_map])
            # solver_parameters.set('p_map', p_map)
            # solver_parameters.set('repeated_velocity_map', repeated_velocity_map)
            self.preconditioner.InitializeNew(repeated_velocity_map,u_map,v_map,u_map,p_map)

        self.solver = FROSch.Solver(self.jac, self.preconditioner, self.teuchos_parameters)

        # Put back the original parameters
        for i in parameter_names:
            self.unset_parameter(i, original_parameters)

    def partition_domain(self):
        '''Partition the domain into Cartesian subdomains for computing the
        discretization.'''

        rmin = 1e100

        self.npx = 1
        self.npy = 1
        self.npz = 1

        nparts = self.comm.NumProc()
        pid = self.comm.MyPID()

        found = False

        # check all possibilities of splitting the map
        for t1 in range(1, nparts + 1):
            for t2 in range(1, nparts // t1 + 1):
                t3 = nparts // (t1 * t2)
                if t1 * t2 * t3 == nparts:
                    nx_loc = self.nx_global // t1
                    ny_loc = self.ny_global // t2
                    nz_loc = self.nz_global // t3

                    if nx_loc * t1 != self.nx_global or ny_loc * t2 != self.ny_global or nz_loc * t3 != self.nz_global:
                        continue

                    r1 = abs(self.nx_global / t1 - self.ny_global / t2)
                    r2 = abs(self.nx_global / t1 - self.nz_global / t3)
                    r3 = abs(self.ny_global / t2 - self.nz_global / t3)
                    r = r1 + r2 + r3

                    if r < rmin:
                        rmin = r
                        self.npx = t1
                        self.npy = t2
                        self.npz = t3
                        found = True

        if not found:
            raise Exception('Could not split %dx%dx%d domain in %d parts.' % (self.nx_global, self.ny_global,
                                                                              self.nz_global, nparts))

        self.pidx, self.pidy, self.pidz, _ = ind2sub(self.npx, self.npy, self.npz, pid)

        # Compute the local domain size and offset.
        self.nx_local = self.nx_global // self.npx
        self.ny_local = self.ny_global // self.npy
        self.nz_local = self.nz_global // self.npz

        self.nx_offset = self.nx_local * self.pidx
        self.ny_offset = self.ny_local * self.pidy
        self.nz_offset = self.nz_local * self.pidz

        # Add ghost nodes to factor out boundary conditions in the interior.
        if self.pidx > 0:
            self.nx_local += 2
            self.nx_offset -= 2
        if self.pidx < self.npx - 1:
            self.nx_local += 2
        if self.pidy > 0:
            self.ny_local += 2
            self.ny_offset -= 2
        if self.pidy < self.npy - 1:
            self.ny_local += 2
        if self.pidz > 0:
            self.nz_local += 2
            self.nz_offset -= 2
        if self.pidz < self.npz - 1:
            self.nz_local += 2

    def is_ghost(self, i, j=None, k=None):
        '''If a node is a ghost node that is used only for computing the
        discretization and is located outside of an interior boundary.'''

        if j is None:
            i, j, k, _ = ind2sub(self.nx_local, self.ny_local, self.nz_local, i, self.dof)

        ghost = False
        if self.pidx > 0 and i < 2:
            ghost = True
        elif self.pidx < self.npx - 1 and i >= self.nx_local - 2:
            ghost = True
        elif self.pidy > 0 and j < 2:
            ghost = True
        elif self.pidy < self.npy - 1 and j >= self.ny_local - 2:
            ghost = True
        elif self.pidz > 0 and k < 2:
            ghost = True
        elif self.pidz < self.npz - 1 and k >= self.nz_local - 2:
            ghost = True

        return ghost

    def is_interface(self, i, j, k):
        interface = False
        if self.pidx > 0 and i == 1:
            interface = True
        elif self.pidy > 0 and j == 1:
            interface = True
        elif self.pidz > 0 and k == 1:
            interface = True

        return interface

    def create_map(self, overlapping=False):
        '''Create a map on which the local discretization domain is defined.
        The overlapping part is only used for computing the discretization.'''

        local_elements = [0] * self.nx_local * self.ny_local * self.nz_local * self.dof

        pos = 0
        for k in range(self.nz_local):
            for j in range(self.ny_local):
                for i in range(self.nx_local):
                    if not overlapping and self.is_ghost(i, j, k):
                        continue
                    for var in range(self.dof):
                        local_elements[pos] = sub2ind(self.nx_global, self.ny_global, self.nz_global, self.dof,
                                                      i + self.nx_offset, j + self.ny_offset, k + self.nz_offset, var)
                        pos += 1

        return Epetra.Map(-1, local_elements[0:pos], 0, self.comm)

    def create_dof_map(self, var, interface=False):
        '''Create a map on which the local discretization domain is defined for the specified degree of freedom (var).
        E.g., if var=0, your map will contain only the first variable (typically u) in each grid cell.
        The interface part is used for communication between processors.'''

        local_elements = [0] * self.nx_local * self.ny_local * self.nz_local

        pos = 0
        for k in range(self.nz_local):
            for j in range(self.ny_local):
                for i in range(self.nx_local):
                    if self.is_ghost(i, j, k) and (not interface or not self.is_interface(i, j, k)):
                        continue
                    local_elements[pos] = sub2ind(self.nx_global, self.ny_global, self.nz_global, self.dof,
                                                  i + self.nx_offset, j + self.ny_offset, k + self.nz_offset, var)
                    pos += 1

        return Epetra.Map(-1, local_elements[0:pos], 0, self.comm)

    def create_repeated_map(self, maps):
        '''Create a repeated map by merging the given maps.'''

        local_elements = 0
        for m in maps:
            local_elements += m.NumMyElements()

        local_elements = [0] * local_elements

        pos = 0
        for m in maps:
            for i in m.MyGlobalElements():
                local_elements[pos] = i
                pos += 1

        return Epetra.Map(-1, local_elements[0:pos], 0, self.comm)

    def rhs(self, state):
        '''Right-hand side in M * du / dt = F(u) defined on the
        non-overlapping discretization domain map.'''

        state_ass = Vector(self.assembly_map)
        state_ass.Import(state, self.assembly_importer, Epetra.Insert)

        rhs = fvm.Interface.rhs(self, state_ass)
        rhs_ass = Vector(Epetra.Copy, self.assembly_map, rhs)
        rhs = Vector(self.map)
        rhs.Export(rhs_ass, self.assembly_importer, Epetra.Zero)
        return rhs

    def jacobian(self, state):
        '''Jacobian J of F in M * du / dt = F(u) defined on the
        domain map used by FROSch.'''

        state_ass = Vector(self.assembly_map)
        state_ass.Import(state, self.assembly_importer, Epetra.Insert)

        local_jac = fvm.Interface.jacobian(self, state_ass)

        if self.jac is None:
            self.jac = Epetra.FECrsMatrix(Epetra.Copy, self.solve_map, 27)
        else:
            self.jac.PutScalar(0.0)

        for i in range(len(local_jac.begA)-1):
            if self.is_ghost(i):
                continue
            row = self.assembly_map.GID64(i)

            if self.dof > self.dim and row == self.pressure_row:
                self.jac[row, row] = -1.0
                continue

            for j in range(local_jac.begA[i], local_jac.begA[i+1]):
                col = self.assembly_map.GID64(local_jac.jcoA[j])

                if self.dof > self.dim and col == self.pressure_row:
                    continue

                # __setitem__ automatically calls ReplaceGlobalValues if the matrix is filled,
                # InsertGlobalValues otherwise
                self.jac[row, col] = local_jac.coA[j]

        self.jac.GlobalAssemble(True, Epetra.Insert)

        return self.jac

    def mass_matrix(self):
        '''Mass matrix M in M * du / dt = F(u) defined on the
        domain map used by FROSch.'''

        local_mass = fvm.Interface.mass_matrix(self)

        if self.mass is None:
            self.mass = Epetra.FECrsMatrix(Epetra.Copy, self.solve_map, 1)
        else:
            self.mass.PutScalar(0.0)

        for i in range(len(local_mass.begA)-1):
            if self.is_ghost(i):
                continue
            row = self.assembly_map.GID64(i)
            for j in range(local_mass.begA[i], local_mass.begA[i+1]):
                # __setitem__ automatically calls ReplaceGlobalValues if the matrix is filled,
                # InsertGlobalValues otherwise
                self.mass[row, self.assembly_map.GID64(local_mass.jcoA[j])] = local_mass.coA[j]
        self.mass.GlobalAssemble(True, Epetra.Insert)

        return self.mass

    def compute_scaling(self):
        '''Compute scaling for the linear problem'''
        self.left_scaling = Vector(self.solve_map)
        self.left_scaling.PutScalar(1.0)

        self.inv_left_scaling = Vector(self.solve_map)
        self.inv_left_scaling.PutScalar(1.0)

        self.right_scaling = Vector(self.solve_map)
        self.right_scaling.PutScalar(1.0)

        self.inv_right_scaling = Vector(self.solve_map)
        self.inv_right_scaling.PutScalar(1.0)

        dim = self.discretization.dim
        dof = self.discretization.dof

        for lrid in range(self.jac.NumMyRows()):
            grid = self.jac.GRID(lrid)
            var = grid % dof
            values, indices = self.jac.ExtractMyRowCopy(lrid)
            for j in range(len(indices)):
                lcid = indices[j]
                value = values[j]
                gcid = self.jac.GCID(lcid)
                if value < 1e-8:
                    continue

                if var != dim and gcid % dof == dim:
                    # If the row is a velocity and the column a pressure
                    self.left_scaling[lrid] = 1 / value
                    self.inv_left_scaling[lrid] = value
                    break
                elif var == dim and gcid % dof != dim and self.jac.MyGRID(gcid):
                    # If the row is a pressure and the column a velocity
                    lid = self.jac.LRID(gcid)
                    self.right_scaling[lid] = 1 / value
                    self.inv_right_scaling[lid] = value

    def scale_matrix(self, mat):
        assert not hasattr(mat, 'scaled') or not mat.scaled
        mat.scaled = True

        mat.LeftScale(self.left_scaling)
        mat.RightScale(self.right_scaling)

    def scale_jacobian(self):
        self.scale_matrix(self.jac)

    def scale_rhs(self, rhs):
        rhs.Multiply(1.0, self.left_scaling, rhs, 0.0)

    def scale_lhs(self, lhs):
        lhs.Multiply(1.0, self.inv_right_scaling, lhs, 0.0)

    def unscale_matrix(self, mat):
        assert mat.scaled
        mat.scaled = False

        mat.LeftScale(self.inv_left_scaling)
        mat.RightScale(self.inv_right_scaling)

    def unscale_jacobian(self):
        self.unscale_matrix(self.jac)

    def unscale_rhs(self, rhs):
        rhs.Multiply(1.0, self.inv_left_scaling, rhs, 0.0)

    def unscale_lhs(self, lhs):
        lhs.Multiply(1.0, self.right_scaling, lhs, 0.0)

    def solve(self, jac, rhs, rhs2=None, V=None, W=None, C=None, solver=None):
        '''Solve J y = x for y with the possibility of solving a bordered system.'''

        if solver is None:
            solver = self.solver

        rhs_sol = Vector(self.solve_map)
        rhs_sol.Import(rhs, self.solve_importer, Epetra.Insert)

        x_sol = Vector(rhs_sol)

        self.compute_scaling()

        if rhs2 is not None:
            raise Exception('Not implemented')

        self.scale_jacobian()
        self.scale_rhs(rhs_sol)

        self.preconditioner.Compute()
        solver.ApplyInverse(rhs_sol, x_sol)

        self.unscale_jacobian()
        self.unscale_lhs(x_sol)

        x = Vector(rhs)
        x.Export(x_sol, self.solve_importer, Epetra.Insert)

        return x

    def eigs(self, state, return_eigenvectors=False, enable_recycling=False):
        '''Compute the generalized eigenvalues of beta * J(x) * v = alpha * M * v.'''

        # FIXME: Implement this
        pass
