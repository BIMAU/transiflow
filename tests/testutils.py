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
