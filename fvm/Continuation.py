

from math import sqrt
import numpy as np


def norm(x):
    return sqrt(x.dot(x))

def infinity_norm(x):
    return max(abs(x))


class Continuation:
    def __init__(self, interface, parameters):
        self.interface = interface
        self.parameters = parameters

    def newton(self, x0, tol=1.e-7, maxit=1000):
        x = x0
        for k in range(maxit):
            fval = self.interface.rhs(x)
            jac = self.interface.jacobian(x)
            dx = self.interface.solve(jac, -fval)

            x = x + dx

            dxnorm = norm(dx)
            if dxnorm < tol:
                print('Newton converged in %d steps with norm %e' % (k, dxnorm))
                break

        return x

    def newtoncorrector(self, parameter_name, ds, x, x0, mu, mu0, tol):
        # Set some parameters
        maxit = 10000
        # zeta = 1 / len(x)
        zeta = 1
        delta = 0.1  # changed. It is used to compute F_mu

        # Do the main iteration
        for k in range(maxit):
            # Compute F and F_mu (RHS of 2.2.9)
            self.interface.set_parameter(parameter_name, mu + delta)
            dflval = self.interface.rhs(x)
            self.interface.set_parameter(parameter_name, mu)
            fval = self.interface.rhs(x)
            dflval = (dflval - fval) / delta

            # Compute the jacobian at x
            jac = self.interface.jacobian(x)

            # Compute r (2.2.8)
            diff = x - x0
            rnp1 = zeta * diff.dot(diff) + (1 - zeta) * (mu - mu0) ** 2 - ds ** 2

            r_x = -2 * zeta * (x - x0)
            r_mu = -2 * (1 - zeta) * (mu - mu0)

            if self.parameters.get("Bordered Solver", False):
                # Solve the entire bordered system in one go (2.2.9)
                # dx, dmu = self.interface.solve(jac, -fval, -rnp1, dflval, 2 * zeta * diff, 2 * (1 - zeta) * (mu - mu0))
                #TODO dont use hymls, solve bordered system directly
                res = self.interface.solve_bordered(jac, fval, dflval, r_x, r_mu, rnp1)
                dx = res[:len(res) - 1]
                dmu = res[-1]

            else:
                # Solve twice with F_x (2.2.9)
                z1 = self.interface.solve(jac, -fval)
                z2 = self.interface.solve(jac, dflval)

                # Compute dmu (2.2.13)
                # dmu = (-rnp1 - 2 * zeta * diff.dot(z1)) / (2 * (1 - zeta) * (mu - mu0) - 2 * zeta * diff.dot(z2))
                dmu = (-rnp1 + r_x.dot(z1)) / (-r_mu + r_x.dot(z2))
                # Compute dx (2.2.12)
                dx = z1 - dmu * z2

            # Compute a new x and mu (2.2.10 - 2.2.11)
            x = x + dx
            mu = mu + dmu

            dxnorm = norm(dx)
            # if max(dmu, dxnorm) < tol:
            #     print('Newton corrector converged in %d steps with norm %e' % (k, dxnorm))
            #     return (x, mu)
            if dxnorm < tol:
                print('Newton corrector converged in %d steps with norm %e' % (k, dxnorm))
                num_iterations = k
                return (x, mu, num_iterations)

        print('No convergence achieved by Newton corrector')

    def continuation(self, x0, parameter_name, target, ds, maxit):
        x = x0

        # Get the initial tangent (2.2.5 - 2.2.7).
        delta = 1
        mu = self.interface.get_parameter(parameter_name)
        fval = self.interface.rhs(x)
        self.interface.set_parameter(parameter_name, mu + delta)
        dmu = (self.interface.rhs(x) - fval) / delta
        self.interface.set_parameter(parameter_name, mu)

        # Compute the jacobian at x and solve with it (2.2.5)
        jac = self.interface.jacobian(x)
        dx = -self.interface.solve(jac, dmu)

        # Scaling of the initial tangent (2.2.7)
        dmu = 1
        zeta = 1
        nrm = sqrt(zeta * dx.dot(dx) + dmu ** 2)
        dmu = dmu / nrm
        dx = dx / nrm

        dmu0 = dmu
        dx0 = dx

        # TODO wei
        paras = []
        u = []
        u_norm = []

        # see if it passes the corner
        flag = 0

        C_v = []
        iterations = []

        # Perform the continuation
        for j in range(maxit):
            mu0 = mu
            x0 = x

            # Predictor (2.2.3)
            mu = mu0 + ds * dmu0
            x = x0 + ds * dx0

            # Corrector (2.2.9 and onward)
            (x2, mu2, num_iterations) = self.newtoncorrector(parameter_name, ds, x, x0, mu, mu0, 1e-4)

            print("%s: %f" % (parameter_name, mu2))

            if flag == 0 and mu2 > 3.5:
                flag = 1

            if mu2 > 3.5:
                C_v.append(mu2)
                iterations.append(num_iterations)

            paras.append(mu2)
            u.append(x2)
            u_norm.append(infinity_norm(x2))

            # we need to find an appropriate threshold so that we end up the value in the upper branch that is close to 0
            if flag == 1 and mu2 < 0.1:
                return x2, paras, u, u_norm, C_v, iterations



            # if (mu2 >= target and mu0 < target) or (mu2 <= target and mu0 > target):
            #     # Converge onto the end point (we usually go past it, so we
            #     # use Newton to converge)
            #     mu = target
            #     self.interface.set_parameter(parameter_name, mu)
            #     x = self.newton(x, 1e-4)
            #
            #     return x

            # Set the new values computed by the corrector
            dmu = mu2 - mu0
            mu = mu2
            dx = x2 - x0
            x = x2


            # when ds is small, dmu is also small, in this case, we cannot use this condition.
            # if abs(dmu) < 1e-10:
            if abs(dmu) < ds * 1e-5:
                return x, paras, u, u_norm, C_v, iterations

            # Compute the tangent (2.2.4)
            dx0 = dx / ds
            dmu0 = dmu / ds

        return x, paras, u, u_norm, C_v, iterations
