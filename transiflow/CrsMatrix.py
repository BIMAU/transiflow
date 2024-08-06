import numpy
import os

import tempfile
import subprocess

class CrsMatrix:
    '''Compressed sparse row matrix used for assembly of the internal
    matrices. Can also be used for debugging purposes'''
    def __init__(self, coA=None, jcoA=None, begA=None, compress=True, m=None, n=None):
        self.coA = coA
        self.jcoA = jcoA
        self.begA = begA

        self._tmp = None

        if compress and coA is not None:
            self.compress()

        self.lu = None
        self.bordered_lu = False

        self._m = m
        self._n = n

    @property
    def m(self):
        '''Row dimension of the matrix'''
        if self._m:
            return self._m

        return len(self.begA) - 1

    @property
    def n(self):
        '''Column dimension of the matrix'''
        if self._n:
            return self._n

        return self.m

    @property
    def shape(self):
        '''Shape of the matrix'''
        return (self.m, self.n)

    @property
    def dtype(self):
        '''Shape of the matrix'''
        return self.coA.dtype

    def compress(self):
        ''' Remove zeros and merge duplicate entries, which may occur in the case of periodic
        boundary conditions.'''
        idx = 0
        beg = self.begA[0]
        for i in range(len(self.begA) - 1):
            unique_indices, inverse_indices = numpy.unique(self.jcoA[beg:self.begA[i+1]], return_inverse=True)

            values = numpy.zeros(len(unique_indices), dtype=self.coA.dtype)
            for orig_idx, inverse_idx in enumerate(inverse_indices):
                values[inverse_idx] += self.coA[beg + orig_idx]

            for j in range(len(unique_indices)):
                if abs(values[j]) > 1e-14:
                    self.jcoA[idx] = unique_indices[j]
                    self.coA[idx] = values[j]
                    idx += 1

            beg = self.begA[i+1]
            self.begA[i+1] = idx

    def solve(self, rhs):
        '''Solve a system using the self.lu property'''
        if self.lu.L.dtype != rhs.dtype and numpy.dtype(rhs.dtype.char.upper()) == rhs.dtype:
            x = rhs.copy()
            x.real = self.solve(rhs.real)
            x.imag = self.solve(rhs.imag)
        else:
            x = self.lu.solve(rhs)

        return x

    def __add__(self, B):
        coA = numpy.zeros(self.begA[-1] + B.begA[-1], dtype=self.dtype)
        jcoA = numpy.zeros(self.begA[-1] + B.begA[-1], dtype=int)
        begA = numpy.zeros(len(self.begA), dtype=int)

        idx = 0
        for i in range(self.n):
            for j in range(self.begA[i], self.begA[i+1]):
                found = False
                for k in range(B.begA[i], B.begA[i+1]):
                    if self.jcoA[j] == B.jcoA[k]:
                        coA[idx] = self.coA[j] + B.coA[k]
                        jcoA[idx] = self.jcoA[j]
                        idx += 1
                        found = True
                        break

                if not found:
                    coA[idx] = self.coA[j]
                    jcoA[idx] = self.jcoA[j]
                    idx += 1

            for j in range(B.begA[i], B.begA[i+1]):
                found = False
                for k in range(self.begA[i], self.begA[i+1]):
                    if B.jcoA[j] == self.jcoA[k]:
                        found = True
                        break

                if not found:
                    coA[idx] = B.coA[j]
                    jcoA[idx] = B.jcoA[j]
                    idx += 1

            begA[i+1] = idx

        return CrsMatrix(coA, jcoA, begA, False)

    def __iadd__(self, B):
        if self.coA is not None:
            A = self + B
            self.coA = A.coA
            self.jcoA = A.jcoA
            self.begA = A.begA
            return self

        self._tmp = B
        return self

    def __neg__(self):
        return CrsMatrix(-self.coA[:self.begA[-1]], self.jcoA[:self.begA[-1]], self.begA, False)

    def __sub__(self, B):
        return self + -B

    def __isub__(self, B):
        self += -B
        return self

    def __mul__(self, x):
        A = CrsMatrix(self.coA[:self.begA[-1]].copy(), self.jcoA[:self.begA[-1]].copy(),
                      self.begA.copy(), False)

        for i in range(self.n):
            for j in range(A.begA[i], A.begA[i+1]):
                A.coA[j] *= x
        return A

    def __truediv__(self, x):
        return self * (1 / x)

    def matvec(self, x):
        '''Return $y = A x$'''
        if len(x.shape) > 1:
            shape = list(x.shape)
            shape[0] = self.m
            b = numpy.zeros(shape, dtype=x.dtype)
            for i in range(self.m):
                for j in range(self.begA[i], self.begA[i+1]):
                    b[i, :] += self.coA[j] * x[self.jcoA[j], :]
            return b

        b = numpy.zeros(self.m, dtype=x.dtype)
        for i in range(self.m):
            for j in range(self.begA[i], self.begA[i+1]):
                b[i] += self.coA[j] * x[self.jcoA[j]]
        return b

    def __matmul__(self, x):
        return self.matvec(x)

    def __str__(self):
        out = ''
        for i in range(self.m):
            for j in range(self.begA[i], self.begA[i+1]):
                out += '%5d %5d %e\n' % (i, self.jcoA[j], self.coA[j])
        return out

    def to_coo(self):
        '''Convert the matrix to coordinate format.

        Returns
        -------
        coA : array_like
            Values
        icoA : array_like
            Row indices
        jcoA : array_like
            Column indices

        '''
        coA = numpy.zeros(self.begA[-1], dtype=self.dtype)
        icoA = numpy.zeros(self.begA[-1], dtype=int)
        jcoA = numpy.zeros(self.begA[-1], dtype=int)

        idx = 0
        for i in range(self.m):
            for j in range(self.begA[i], self.begA[i+1]):
                icoA[idx] = i
                jcoA[idx] = self.jcoA[j]
                coA[idx] = self.coA[j]
                idx += 1

        return coA, icoA, jcoA

    def to_dense(self):
        '''Convert the matrix to a dense matrix.

        Returns
        -------
        A : array_like
            Dense matrix

        '''
        A = numpy.zeros(self.shape, dtype=self.dtype)
        for i in range(self.m):
            for j in range(self.begA[i], self.begA[i+1]):
                A[i, self.jcoA[j]] = self.coA[j]

        return A

    def transpose(self):
        '''Return the transpose of the matrix.

        Returns
        -------
        B : CrsMatrix
            Transpose of the matrix

        '''
        coA, icoA, jcoA = self.to_coo()
        indices = sorted(range(self.begA[-1]), key=lambda i: jcoA[i])

        coB = numpy.zeros(self.begA[-1], dtype=self.dtype)
        jcoB = numpy.zeros(self.begA[-1], dtype=int)
        begB = numpy.zeros(len(self.begA), dtype=int)

        i = 0
        idx = 0
        for j in indices:
            while jcoA[j] > i:
                begB[i+1] = idx
                i += 1

            jcoB[idx] = icoA[j]
            coB[idx] = coA[j]
            idx += 1

        begB[i+1] = idx

        return CrsMatrix(coB, jcoB, begB, False)

    def dump(self, name):
        '''Dump the matrix to a file.

        Parmaters
        ---------
        name : string
            Name of the file

        '''
        with open(name, 'w') as f:
            out = '%%%%MatrixMarket matrix coordinate real general\n%d %d %d\n' % (self.m, self.n, self.begA[self.m])
            for i in range(self.m):
                for j in range(self.begA[i], self.begA[i+1]):
                    out += '%d %d %e\n' % (i+1, self.jcoA[j]+1, self.coA[j])
            f.write(out)

    def _get_index_list(self, idx, n):
        if isinstance(idx, int):
            idx = [idx]
        elif isinstance(idx, slice):
            start = idx.start
            if start is None:
                start = 0

            stop = idx.stop
            if stop is None:
                stop = n

            idx = numpy.arange(start, stop)

        return numpy.array(idx)

    def __getitem__(self, key):
        if not isinstance(key, tuple) or len(key) != 2:
            raise Exception('Key should be a tuple')

        iidx = self._get_index_list(key[0], self.m)
        jidx = self._get_index_list(key[1], self.n)

        if self.coA is None:
            # We must be using +=, which is implemented as
            # self.__setitem__(key, self.__getitem__(key).__iadd__(b))
            # so just return self here
            return CrsMatrix(m=len(iidx), n=len(jidx))

        coA = numpy.zeros(self.begA[-1], dtype=self.dtype)
        jcoA = numpy.zeros(self.begA[-1], dtype=int)
        begA = [0]

        index_list = -numpy.ones(self.n, dtype=int)
        for i, j in enumerate(jidx):
            index_list[j] = i

        idx = 0
        for i in iidx:
            for j in range(self.begA[i], self.begA[i+1]):
                if index_list[self.jcoA[j]] < 0:
                    continue

                coA[idx] = self.coA[j]
                jcoA[idx] = index_list[self.jcoA[j]]
                idx += 1

            begA.append(idx)

        if begA[-1] == 0:
            return 0

        if begA[-1] == 1:
            return coA[0]

        return CrsMatrix(coA[:idx], jcoA[:idx], numpy.array(begA),
                         compress=False, m=len(iidx), n=len(jidx))

    def assemble(self):
        '''Assemble the matrix after using the setter to assign
        values.'''
        assert self._tmp

        iidx = numpy.concatenate([i[0] for i in self._tmp])
        jidx = numpy.concatenate([i[1] for i in self._tmp])
        vals = numpy.concatenate([i[2] for i in self._tmp])

        sorted_idx = numpy.argsort(iidx)

        coA = numpy.zeros(len(iidx), dtype=vals[0].dtype)
        jcoA = numpy.zeros(len(iidx), dtype=int)
        begA = numpy.zeros(self.m + 1, dtype=int)

        idx = 0
        for i in range(self.m):
            for j in sorted_idx[idx:]:
                if iidx[j] > i:
                    break

                coA[idx] = vals[j]
                jcoA[idx] = jidx[j]
                idx += 1

            begA[i+1] = idx

        self._tmp = None

        self.coA = coA
        self.jcoA = jcoA
        self.begA = begA

        self.compress()

    def __setitem__(self, key, A):
        if not self._tmp:
            self._tmp = []

        iidx = self._get_index_list(key[0], self.m)
        jidx = self._get_index_list(key[1], self.n)

        if isinstance(A, CrsMatrix) and A._tmp is not None:
            self[key] = A._tmp
        elif isinstance(A, CrsMatrix):
            tmp = A.to_coo()
            self._tmp.append((iidx[tmp[1]], jidx[tmp[2]], tmp[0]))
        elif hasattr(A, 'shape'):
            tmp = list(numpy.where(abs(A) > 1e-14))
            self._tmp.append((iidx[tmp[0]], jidx[tmp[1]], A[tmp[0], tmp[1]]))
        else:
            self._tmp.append(([iidx[0]], [jidx[0]], [A]))

    def show(self, dof=None):
        '''Use vsm to show the structure of the matrix. vsm can
        currently be found in the mrilu directory of I-EMIC.'''
        def wrtbcsr(beg, jco, co, f):
            n = numpy.int32(len(beg) - 1)

            bc = numpy.int32(n.nbytes)
            os.write(f, bc.tobytes())
            os.write(f, n.tobytes())
            os.write(f, bc.tobytes())

            bc = numpy.int32(beg.nbytes)
            os.write(f, bc.tobytes())
            os.write(f, beg.tobytes())
            os.write(f, bc.tobytes())

            bc = numpy.int32(jco.nbytes)
            os.write(f, bc.tobytes())
            os.write(f, jco.tobytes())
            os.write(f, bc.tobytes())

            bc = numpy.int32(co.nbytes)
            os.write(f, bc.tobytes())
            os.write(f, co.tobytes())
            os.write(f, bc.tobytes())

        nnz = self.begA[-1]
        begA = numpy.ndarray(self.m + 1, numpy.int32)
        jcoA = numpy.ndarray(nnz, numpy.int32)
        coA = numpy.ndarray(nnz, float)

        begA[:] = self.begA
        jcoA[:] = self.jcoA[:nnz]
        coA[:] = self.coA[:nnz]

        if dof:
            idx = 0
            idx_map = [0] * self.m
            for d in range(dof):
                for i in range(d, self.m, dof):
                    idx_map[i] = idx
                    idx += 1

            idx = 0
            row_idx = 0
            for d in range(dof):
                for i in range(d, self.m, dof):
                    for j in range(self.begA[i], self.begA[i+1]):
                        jcoA[idx] = idx_map[self.jcoA[j]]
                        coA[idx] = self.coA[j]
                        idx += 1

                    row_idx += 1
                    begA[row_idx] = idx

        begA += 1
        jcoA += 1

        f, fname = tempfile.mkstemp()
        wrtbcsr(begA, jcoA, coA, f)
        os.close(f)

        subprocess.call(['vsm', fname])

        os.remove(fname)
