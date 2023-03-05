import numpy

class CrsMatrix:
    def __init__(self, coA=None, jcoA=None, begA=None, compress=True, m=None, n=None):
        self.coA = coA
        self.jcoA = jcoA
        self.begA = begA

        if compress:
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
        if len(rhs.shape) < 2:
            if self.lu.L.dtype != rhs.dtype and numpy.dtype(rhs.dtype.char.upper()) == rhs.dtype:
                x = rhs.copy()
                x.real = self.solve(rhs.real)
                x.imag = self.solve(rhs.imag)
            else:
                x = self.lu.solve(rhs)
        else:
            x = rhs.copy()
            for i in range(x.shape[1]):
                x[:, i] = self.solve(rhs[:, i])
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

    def __sub__(self, B):
        A = CrsMatrix(-B.coA[:B.begA[-1]], B.jcoA[:B.begA[-1]], B.begA, False)
        return self + A

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
        b = numpy.zeros(x.shape, dtype=x.dtype)
        for i in range(self.n):
            for j in range(self.begA[i], self.begA[i+1]):
                b[i] += self.coA[j] * x[self.jcoA[j]]
        return b

    def __matmul__(self, x):
        return self.matvec(x)

    def __str__(self):
        out = ''
        for i in range(self.n):
            for j in range(self.begA[i], self.begA[i+1]):
                out += '%5d %5d %e\n' % (i, self.jcoA[j], self.coA[j])
        return out

    def to_coo(self):
        coA = numpy.zeros(self.begA[-1], dtype=self.dtype)
        icoA = numpy.zeros(self.begA[-1], dtype=int)
        jcoA = numpy.zeros(self.begA[-1], dtype=int)

        idx = 0
        for i in range(self.n):
            for j in range(self.begA[i], self.begA[i+1]):
                icoA[idx] = i
                jcoA[idx] = self.jcoA[j]
                coA[idx] = self.coA[j]
                idx += 1

        return coA, icoA, jcoA

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
            out = '%%%%MatrixMarket matrix coordinate real general\n%d %d %d\n' % (self.n, self.n, self.begA[self.n])
            for i in range(self.n):
                for j in range(self.begA[i], self.begA[i+1]):
                    out += '%d %d %e\n' % (i+1, self.jcoA[j]+1, self.coA[j])
            f.write(out)
