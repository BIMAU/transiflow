import numpy

class CrsMatrix:
    def __init__(self, coA=None, jcoA=None, begA=None):
        self.coA = coA
        self.jcoA = jcoA
        self.begA = begA

        self.compress()

        self.lu = None

    def _get_n(self):
        return len(self.begA) - 1

    n = property(_get_n)

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

    def __matmul__(self, x):
        b = numpy.zeros(self.n, dtype=x.dtype)
        for i in range(self.n):
            for j in range(self.begA[i], self.begA[i+1]):
                b[i] += self.coA[j] * x[self.jcoA[j]]
        return b

    def __str__(self):
        out = ''
        for i in range(self.n):
            for j in range(self.begA[i], self.begA[i+1]):
                out += '%5d %5d %e\n' % (i, self.jcoA[j], self.coA[j])
        return out
