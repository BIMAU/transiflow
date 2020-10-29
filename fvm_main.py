import numpy

from fvm import BoundaryConditions
from fvm import Discretization
from fvm import utils

class CrsMatrix:
    def __init__(self, coA=None, jcoA=None, begA=None):
        self.coA = coA
        self.jcoA = jcoA
        self.begA = begA

def linear_part(nx, ny, nz, dof, Re, Ra=0, Pr=0):
    x = utils.create_uniform_coordinate_vector(nx)
    y = utils.create_uniform_coordinate_vector(ny)
    z = utils.create_uniform_coordinate_vector(nz)

    discretization = Discretization(nx, ny, nz, dof)

    atom = 1 / Re * (discretization.u_xx(x, y, z) + discretization.u_yy(x, y, z) + discretization.u_zz(x, y, z) \
                  +  discretization.v_xx(x, y, z) + discretization.v_yy(x, y, z) + discretization.v_zz(x, y, z) \
                  +  discretization.w_xx(x, y, z) + discretization.w_yy(x, y, z) + discretization.w_zz(x, y, z)) \
        - (discretization.p_x(x, y, z) + discretization.p_y(x, y, z) + discretization.p_z(x, y, z)) \
        + discretization.div(x, y, z)

    if Ra:
        atom += Ra * discretization.forward_average_T_z(x, y, z)

    if Pr:
        atom += 1 / Pr * (discretization.T_xx(x, y, z) + discretization.T_yy(x, y, z) + discretization.T_zz(x, y, z))
        atom += 1 / Pr * discretization.backward_average_w_z(x, y, z)

    return atom

def problem_type_equals(first, second):
    return first.lower() == second.lower()

def boundaries(atom, nx, ny, nz, dof, problem_type='Lid-driven cavity'):
    boundary_conditions = BoundaryConditions(nx, ny, nz, dof)

    x = utils.create_uniform_coordinate_vector(nx)
    y = utils.create_uniform_coordinate_vector(ny)
    z = utils.create_uniform_coordinate_vector(nz)

    frc = numpy.zeros(nx * ny * nz * dof)

    if problem_type_equals(problem_type, 'Rayleigh-Benard'):
        frc += boundary_conditions.heatflux_east(atom, x, y, z, 0)
    elif problem_type_equals(problem_type, 'Differentially heated cavity'):
        frc += boundary_conditions.temperature_east(atom, -1/2)
    else:
        boundary_conditions.dirichlet_east(atom)

    if problem_type_equals(problem_type, 'Rayleigh-Benard'):
        frc += boundary_conditions.heatflux_west(atom, x, y, z, 0)
    elif problem_type_equals(problem_type, 'Differentially heated cavity'):
        frc += boundary_conditions.temperature_west(atom, 1/2)
    else:
        boundary_conditions.dirichlet_west(atom)

    if problem_type_equals(problem_type, 'Lid-driven cavity') and nz <= 1:
        frc += boundary_conditions.moving_lid_north(atom, 1)
    elif problem_type_equals(problem_type, 'Rayleigh-Benard'):
        frc += boundary_conditions.heatflux_north(atom, x, y, z, 0)
    elif problem_type_equals(problem_type, 'Differentially heated cavity'):
        frc += boundary_conditions.heatflux_north(atom, x, y, z, 0)
    else:
        boundary_conditions.dirichlet_north(atom)

    if problem_type_equals(problem_type, 'Rayleigh-Benard'):
        frc += boundary_conditions.heatflux_south(atom, x, y, z, 0)
    elif problem_type_equals(problem_type, 'Differentially heated cavity'):
        frc += boundary_conditions.heatflux_south(atom, x, y, z, 0)
    else:
        boundary_conditions.dirichlet_south(atom)

    if problem_type_equals(problem_type, 'Lid-driven cavity') and nz > 1:
        frc += boundary_conditions.moving_lid_top(atom, 1)
    elif problem_type_equals(problem_type, 'Differentially heated cavity'):
        frc += boundary_conditions.heatflux_top(atom, x, y, z, 0)
    else:
        boundary_conditions.dirichlet_top(atom)

    if problem_type_equals(problem_type, 'Differentially heated cavity'):
        frc += boundary_conditions.heatflux_bottom(atom, x, y, z, 0)
    else:
        boundary_conditions.dirichlet_bottom(atom)

    return frc

def convection(state, nx, ny, nz, dof):
    x = utils.create_uniform_coordinate_vector(nx)
    y = utils.create_uniform_coordinate_vector(ny)
    z = utils.create_uniform_coordinate_vector(nz)

    state_mtx = utils.create_state_mtx(state, nx, ny, nz, dof)

    discretization = Discretization(nx, ny, nz, dof)
    return discretization.convection(state_mtx, x, y, z)

def assemble(atom, nx, ny, nz, dof):
    ''' Assemble the Jacobian. Optimized version of

    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                for d1 in range(dof):
                    for z in range(3):
                        for y in range(3):
                            for x in range(3):
                                for d2 in range(dof):
                                    if abs(atom[i, j, k, d1, d2, x, y, z]) > 1e-14:
                                       jcoA[idx] = row + (x-1) * dof + (y-1) * nx * dof + (z-1) * nx * ny * dof + d2 - d1
                                       coA[idx] = atom[i, j, k, d1, d2, x, y, z]
                                       idx += 1
                    row += 1
                    begA[row] = idx
    '''

    row = 0
    idx = 0
    n = nx * ny * nz * dof
    coA = numpy.zeros(27*n)
    jcoA = numpy.zeros(27*n, dtype=int)
    begA = numpy.zeros(n+1, dtype=int)

    # Check where values are nonzero in the atoms
    configs = []
    for z in range(3):
        for y in range(3):
            for x in range(3):
                for d2 in range(dof):
                    if numpy.any(atom[:, :, :, :, d2, x, y, z]):
                        configs.append([d2, x, y, z])

    # Iterate only over configurations with values in there
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                for d1 in range(dof):
                    for config in configs:
                        if abs(atom[i, j, k, d1, config[0], config[1], config[2], config[3]]) > 1e-14:
                            jcoA[idx] = row + (config[1]-1) * dof + (config[2]-1) * nx * dof + (config[3]-1) * nx * ny * dof + config[0] - d1
                            coA[idx] = atom[i, j, k, d1, config[0], config[1], config[2], config[3]]
                            idx += 1
                    row += 1
                    begA[row] = idx
    return CrsMatrix(coA, jcoA, begA)

def rhs(state, atom, nx, ny, nz, dof):
    ''' Assemble the right-hand side. Optimized version of

    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                for d1 in range(dof):
                    for z in range(3):
                        for y in range(3):
                            for x in range(3):
                                for d2 in range(dof):
                                    if abs(atom[i, j, k, d1, d2, x, y, z]) > 1e-14:
                                        offset = row + (x-1) * dof + (y-1) * nx * dof + (z-1) * nx * ny * dof + d2 - d1
                                        out[row] -= atom[i, j, k, d1, d2, x, y, z] * state[offset]
                    row += 1
    '''

    row = 0
    n = nx * ny * nz * dof

    # Put the state in shifted matrix form
    state_mtx = numpy.zeros([nx+2, ny+2, nz+2, dof])
    state_mtx[1:nx+1, 1:ny+1, 1:nz+1, :] = utils.create_state_mtx(state, nx, ny, nz, dof)

    # Add up all contributions without iterating over the domain
    out_mtx = numpy.zeros([nx, ny, nz, dof])
    for k in range(3):
        for j in range(3):
            for i in range(3):
                for d1 in range(dof):
                    for d2 in range(dof):
                        out_mtx[:, :, :, d1] -= atom[:, :, :, d1, d2, i, j, k] * state_mtx[i:(i+nx), j:(j+ny), k:(k+nz), d2]

    return utils.create_state_vec(out_mtx, nx, ny, nz, dof)
