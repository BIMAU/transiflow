import numpy

class CrsMatrix:
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

    def _get_m(self):
        if self._m:
            return self._m

        return len(self.begA) - 1

    m = property(_get_m)

    def _get_n(self):
        if self._n:
            return self._n

        return self.m

    n = property(_get_n)

    def _get_shape(self):
        return (self.m, self.n)

    shape = property(_get_shape)

    def _get_dtype(self):
        return self.coA.dtype

    dtype = property(_get_dtype)

    @property
    def data(self):
        return self.coA

    @property
    def indices(self):
        return self.jcoA

    @property
    def indptr(self):
        return self.begA

    @staticmethod
    def _compress(coA, jcoA, begA):
        ''' Remove zeros and merge duplicate entries, which may occur in the case of periodic
        boundary conditions.'''
        idx = 0
        beg = begA[0]
        for i in range(len(begA) - 1):
            unique_indices, inverse_indices = numpy.unique(jcoA[beg:begA[i+1]], return_inverse=True)

            values = numpy.zeros(len(unique_indices), dtype=coA.dtype)
            for orig_idx, inverse_idx in enumerate(inverse_indices):
                values[inverse_idx] += coA[beg + orig_idx]

            for j in range(len(unique_indices)):
                if abs(values[j]) > 1e-14:
                    jcoA[idx] = unique_indices[j]
                    coA[idx] = values[j]
                    idx += 1

            beg = begA[i+1]
            begA[i+1] = idx

    def compress(self):
        self._compress(self.coA, self.jcoA, self.begA)

    @staticmethod
    def _solve(A, rhs):
        if A.lu.L.dtype != rhs.dtype and numpy.dtype(rhs.dtype.char.upper()) == rhs.dtype:
            x = rhs.copy()
            x.real = A.solve(rhs.real)
            x.imag = A.solve(rhs.imag)
        else:
            x = A.lu.solve(rhs)

        return x

    def solve(self, rhs):
        return self._solve(self, rhs)

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
        A = numpy.zeros(self.shape, dtype=self.dtype)
        for i in range(self.m):
            for j in range(self.begA[i], self.begA[i+1]):
                A[i, self.jcoA[j]] = self.coA[j]

        return A

    def transpose(self):
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
