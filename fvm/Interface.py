from scipy import sparse
from scipy.sparse import linalg

from fvm import Discretization

class Interface:
    def __init__(self, parameters, nx, ny, nz, dim, dof):
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.dim = dim
        self.dof = dof
        self.discretization = Discretization(parameters, nx, ny, nz, dof)

    def set_parameter(self, name, value):
        self.discretization.set_parameter(name, value)

    def get_parameter(self, name):
        return self.discretization.get_parameter(name)

    def rhs(self, state):
        return self.discretization.rhs(state)

    def jacobian(self, state):
        return self.discretization.jacobian(state)

    def solve(self, jac, rhs):
        coA = []
        jcoA = []
        begA = [0]
        for i in range(len(jac.begA)-1):
            if i == 3:
                begA.append(begA[i]+1)
                coA.append(-1)
                jcoA.append(i)
                continue
            for j in range(jac.begA[i], jac.begA[i+1]):
                if jac.jcoA[j] != 3:
                    coA.append(jac.coA[j])
                    jcoA.append(jac.jcoA[j])
            begA.append(len(coA))

        rhs[3] = 0

        A = sparse.csr_matrix((coA, jcoA, begA))
        x = linalg.spsolve(A, rhs)
        return x
