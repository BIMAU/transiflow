from scipy import sparse
from scipy.sparse import linalg

from fvm import Discretization

class Interface:
    def __init__(self, parameters, nx, ny, nz, dof, problem_type='Lid-driven cavity'):
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.dof = dof
        self.problem_type = problem_type
        self.discretization = Discretization(parameters, nx, ny, nz, dof)

    def rhs(self, state, Re_in):
        if Re_in == 0:
            Re = 1
        else:
            Re = Re_in

        self.discretization.set_parameter('Reynolds Number', Re)

        atom = self.discretization.linear_part()
        frc = self.discretization.boundaries(atom, self.problem_type)

        if Re_in != 0:
            atomJ, atomF = self.discretization.nonlinear_part(state)
            atom += atomF

        # FIXME: Check this minus signs
        return -self.discretization.rhs(state, atom) + frc

    def jacobian(self, state, Re_in):
        if Re_in == 0:
            Re = 1
        else:
            Re = Re_in

        self.discretization.set_parameter('Reynolds Number', Re)

        atom = self.discretization.linear_part()
        self.discretization.boundaries(atom)

        if Re_in != 0:
            atomJ, atomF = self.discretization.nonlinear_part(state)
            atom += atomJ

        return self.discretization.jacobian(atom)

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
