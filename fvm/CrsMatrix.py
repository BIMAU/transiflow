import numpy

class CrsMatrix:
    def __init__(self, coA=None, jcoA=None, begA=None, compress=True):
        self.coA = coA
        self.jcoA = jcoA
        self.begA = begA

        if compress:
            self.compress()

        self.lu = None

    def _get_n(self):
        return len(self.begA) - 1

    n = property(_get_n)

    def _get_shape(self):
        return (self.n, self.n)

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
        A = CrsMatrix(self.coA[:self.begA[-1]].copy(), self.jcoA[:self.begA[-1]].copy(),
                      self.begA.copy(), False)

        for i in range(self.n):
            for j in range(B.begA[i], B.begA[i+1]):
                if abs(B.coA[j]) < 1e-14:
                    continue

                found = False
                for k in range(A.begA[i], A.begA[i+1]):
                    if B.jcoA[j] == A.jcoA[k]:
                        A.coA[k] += B.coA[j]
                        found = True
                        break

                if not found:
                    raise Exception('A does not contain the pattern of B', i,
                                    A.jcoA[A.begA[i]:A.begA[i+1]],
                                    B.jcoA[B.begA[i]:B.begA[i+1]])
        return A

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
        b = numpy.zeros(self.n, dtype=x.dtype)
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
