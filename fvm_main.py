import numpy

from fvm import Discretization
from fvm import utils

class CrsMatrix:
    def __init__(self, coA=None, jcoA=None, begA=None):
        self.coA = coA
        self.jcoA = jcoA
        self.begA = begA

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
