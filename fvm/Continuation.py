from math import sqrt

def norm(x):
    return sqrt(x.dot(x))

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
        maxit = 100
        zeta = 1 / len(x)
        delta = 1

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
            rnp1 = zeta*diff.dot(diff) + (1 - zeta) * (mu - mu0) ** 2 - ds ** 2

            if self.parameters.get("Bordered Solver", False):
                # Solve the entire bordered system in one go (2.2.9)
                dx, dmu = self.interface.solve(jac, -fval, -rnp1, dflval, 2 * zeta * diff, 2 * (1 - zeta) * (mu - mu0))
            else:
                # Solve twice with F_x (2.2.9)
                z1 = self.interface.solve(jac, -fval)
                z2 = self.interface.solve(jac, dflval)

                # Compute dmu (2.2.13)
                dmu = (-rnp1 - 2 * zeta * diff.dot(z1)) / (2 * (1 - zeta) * (mu - mu0) - 2 * zeta * diff.dot(z2))

                # Compute dx (2.2.12)
                dx = z1 - dmu * z2

            # Compute a new x and mu (2.2.10 - 2.2.11)
            x = x + dx
            mu = mu + dmu

            dxnorm = norm(dx)
            if dxnorm < tol:
                print('Newton corrector converged in %d steps with norm %e' % (k, dxnorm))
                return (x, mu)

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
        zeta = 1 / len(x)
        nrm = sqrt(zeta * dx.dot(dx) + dmu ** 2)
        dmu = dmu / nrm
        dx = dx / nrm

        dmu0 = dmu
        dx0 = dx

        # Perform the continuation
        for j in range(maxit):
            mu0 = mu
            x0 = x

            # Predictor (2.2.3)
            mu = mu0 + ds * dmu0
            x = x0 + ds * dx0

            # Corrector (2.2.9 and onward)
            x2, mu2 = self.newtoncorrector(parameter_name, ds, x, x0, mu, mu0, 1e-4)

            print("%s: %f" % (parameter_name, mu2))

            if (mu2 >= target and mu0 < target) or (mu2 <= target and mu0 > target):
                # Converge onto the end point (we usually go past it, so we
                # use Newton to converge)
                mu = target
                self.interface.set_parameter(parameter_name, mu)
                x = self.newton(x, 1e-4)

                return x

            # Set the new values computed by the corrector
            dmu = mu2 - mu0
            mu = mu2
            dx = x2 - x0
            x = x2

            if abs(dmu) < 1e-10:
                return

            # Compute the tangent (2.2.4)
            dx0 = dx / ds
            dmu0 = dmu / ds

        return x
