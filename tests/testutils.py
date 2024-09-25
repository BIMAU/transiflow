import numpy
import os

from transiflow import CrsMatrix

def read_matrix(fname):
    A = CrsMatrix([], [], [0])

    dirname = os.path.dirname(__file__)
    with open(os.path.join(dirname, fname), 'r') as f:
        rows = []
        idx = 0
        for i in f.readlines():
            r, c, v = [j.strip() for j in i.strip().split(' ') if j]
            r = int(r) - 1
            c = int(c) - 1
            v = float(v)
            rows.append(r)
            while r >= len(A.begA):
                A.begA.append(idx)
            A.jcoA.append(c)
            A.coA.append(v)
            idx += 1
        A.begA.append(idx)
        assert rows == sorted(rows)

    return A

def write_matrix(A, fname):
    dirname = os.path.dirname(__file__)
    with open(os.path.join(dirname, fname), 'w') as f:
        for i in range(len(A.begA)-1):
            for j in range(A.begA[i], A.begA[i+1]):
                f.write('%12d%12d %.16e\n' % (i+1, A.jcoA[j]+1, A.coA[j]))

def read_vector(fname):
    vec = numpy.array([])

    dirname = os.path.dirname(__file__)
    with open(os.path.join(dirname, fname), 'r') as f:
        for i in f.readlines():
            vec = numpy.append(vec, float(i.strip()))
    return vec

def write_vector(vec, fname):
    dirname = os.path.dirname(__file__)
    with open(os.path.join(dirname, fname), 'w') as f:
        for i in range(len(vec)):
            f.write('%.16e\n' % vec[i])

def assemble_jacobian(atom, nx, ny, nz, dof):
    row = 0
    idx = 0
    n = nx * ny * nz * dof
    coA = numpy.zeros(27*n)
    jcoA = numpy.zeros(27*n, dtype=int)
    begA = numpy.zeros(n+1, dtype=int)

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

    return CrsMatrix(coA, jcoA, begA)
