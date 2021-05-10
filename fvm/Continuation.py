import sys
import numpy

def norm(x):
    return numpy.sqrt(x.dot(x))

class Data:
    def __init__(self):
        self.mu = []
        self.value = []

class Continuation:

    def __init__(self, interface, parameters):
        self.interface = interface
        self.parameters = parameters

        self.newton_iterations = 0
        self.delta = None
        self.zeta = None

    def newton(self, x0, tol=1.e-7, maxit=1000):
        residual_check = self.parameters.get('Residual Check', 'F')
        verbose = self.parameters.get('Verbose', False)

        x = x0
        for k in range(maxit):
            fval = self.interface.rhs(x)

            if residual_check == 'F' or verbose:
                fnorm = norm(fval)

            if residual_check == 'F' and fnorm < tol:
                print('Newton converged in %d iterations with ||F||=%e' % (k, fnorm))
                sys.stdout.flush()
                break

            jac = self.interface.jacobian(x)
            dx = self.interface.solve(jac, -fval)

            x = x + dx

            if residual_check != 'F' or verbose:
                dxnorm = norm(dx)

            if residual_check != 'F' and dxnorm < tol:
                print('Newton converged in %d iterations with ||dx||=%e' % (k, dxnorm))
                sys.stdout.flush()
                break

            if verbose:
                print('Newton status at iteration %d: ||F||=%e, ||dx||=%e' % (k, fnorm, dxnorm))
                sys.stdout.flush()

        self.newton_iterations = k

        return x

    def newtoncorrector(self, parameter_name, ds, x, x0, mu, mu0):
        residual_check = self.parameters.get('Residual Check', 'F')
        verbose = self.parameters.get('Verbose', False)

        # Set some parameters
        maxit = self.parameters.get('Maximum Newton Iterations', 10)
        tol = self.parameters.get('Newton Tolerance', 1e-4)

        self.newton_iterations = 0

        # Do the main iteration
        for k in range(maxit):

            # Compute F and F_mu (RHS of 2.2.9)
            self.interface.set_parameter(parameter_name, mu + self.delta)
            dflval = self.interface.rhs(x)
            self.interface.set_parameter(parameter_name, mu)
            fval = self.interface.rhs(x)
            dflval = (dflval - fval) / self.delta

            if residual_check == 'F' or verbose:
                fnorm = norm(fval)

            if residual_check == 'F' and fnorm < tol:
                print('Newton corrector converged in %d iterations with ||F||=%e' % (k, fnorm))
                sys.stdout.flush()
                break

            # Compute the jacobian at x
            jac = self.interface.jacobian(x)

            # Compute r (2.2.8)
            diff = x - x0
            rnp1 = self.zeta*diff.dot(diff) + (1 - self.zeta) * (mu - mu0) ** 2 - ds ** 2

            if self.parameters.get("Bordered Solver", False):
                # Solve the entire bordered system in one go (2.2.9)
                dx, dmu = self.interface.solve(jac, -fval, -rnp1, dflval, 2 * self.zeta * diff,
                                               2 * (1 - self.zeta) * (mu - mu0))
            else:
                # Solve twice with F_x (2.2.9)
                z1 = self.interface.solve(jac, -fval)
                z2 = self.interface.solve(jac, dflval)

                # Compute dmu (2.2.13)
                dmu = (-rnp1 - 2 * self.zeta * diff.dot(z1)) / (2 * (1 - self.zeta) * (mu - mu0)
                                                                - 2 * self.zeta * diff.dot(z2))

                # Compute dx (2.2.12)
                dx = z1 - dmu * z2

            # Compute a new x and mu (2.2.10 - 2.2.11)
            x += dx
            mu += dmu

            self.newton_iterations += 1

            if residual_check != 'F' or verbose:
                dxnorm = norm(dx)

            if residual_check != 'F' and dxnorm < tol:
                print('Newton corrector converged in %d iterations with ||dx||=%e' % (k, dxnorm))
                sys.stdout.flush()
                break

            if verbose:
                print('Newton corrector status at iteration %d: ||F||=%e, ||dx||=%e' % (k, fnorm, dxnorm))
                sys.stdout.flush()

        if self.newton_iterations == maxit:
            print('Newton did not converge. Adjusting step size and trying again')
            return x0, mu0

        self.interface.set_parameter(parameter_name, mu)

        return x, mu

    def adjust_step_size(self, ds):
        ''' Step size control, see [Seydel p 188.] '''

        min_step_size = self.parameters.get('Minimum Step Size', 0.01)
        max_step_size = self.parameters.get('Maximum Step Size', 2000)
        optimal_newton_iterations = self.parameters.get('Optimal Newton Iterations', 3)

        factor = optimal_newton_iterations / max(self.newton_iterations, 1)
        factor = min(max(factor, 0.5), 2.0)

        ds *= factor

        return numpy.sign(ds) * min(max(abs(ds), min_step_size), max_step_size)

    def detect_bifurcation(self, parameter_name, x, mu, dx, dmu, eigs, deigs, ds, maxit):
        ''' Converge onto a bifurcation '''

        tol = self.parameters.get('Destination Tolerance', 1e-8)

        for j in range(maxit):
            if abs(eigs[0].real) < tol:
                print("Bifurcation found at %s = %f with eigenvalue %e + %ei" % (
                    parameter_name, mu, eigs[0].real, eigs[0].imag))
                sys.stdout.flush()
                break

            # Secant method
            ds = ds / deigs[0].real * -eigs[0].real
            x, mu, dx, dmu, ds = self.step(parameter_name, x, mu, dx, dmu, ds)

            eigs0 = eigs
            eigs = self.interface.eigs(x)
            deigs = eigs - eigs0

        return x, mu

    def converge(self, parameter_name, x, mu, dx, dmu, target, ds, maxit):
        ''' Converge onto the target value '''

        tol = self.parameters.get('Destination Tolerance', 1e-8)

        for j in range(maxit):
            if abs(target - mu) < tol:
                print("Convergence achieved onto target %s = %f" % (parameter_name, mu))
                sys.stdout.flush()
                break

            # Secant method
            ds = 1 / dmu * (target - mu)
            x, mu, dx, dmu, ds = self.step(parameter_name, x, mu, dx, dmu, ds)

        return x, mu

    def step(self, parameter_name, x, mu, dx, dmu, ds):
        ''' Perform one step of the continuation '''

        mu0 = mu
        x0 = x

        # Predictor (2.2.3)
        mu = mu0 + ds * dmu
        x = x0 + ds * dx

        # Corrector (2.2.9 and onward)
        x, mu = self.newtoncorrector(parameter_name, ds, x, x0, mu, mu0)

        if mu == mu0:
            # No convergence was achieved, adjusting the step size
            prev_ds = ds
            ds = self.adjust_step_size(ds)
            if prev_ds == ds:
                raise Exception('Newton cannot achieve convergence')

            return self.step(parameter_name, x0, mu0, dx, dmu, ds)

        print("%s: %f" % (parameter_name, mu))
        sys.stdout.flush()

        # Set the new values computed by the corrector
        dmu = mu - mu0
        dx = x - x0

        if abs(dmu) < 1e-10:
            raise Exception('dmu too small')

        # Compute the tangent (2.2.4)
        dx /= ds
        dmu /= ds

        return x, mu, dx, dmu, ds

    def store_data(self, data, x, mu):
        data.mu.append(mu)
        if 'Value' in self.parameters:
            data.value.append(self.parameters['Value'](x))
        else:
            data.value.append(x)

    def continuation(self, x0, parameter_name, target, ds, maxit, verbose=False):
        x = x0

        # Set some parameters
        self.delta = 1
        self.zeta = 1 / len(x)

        # Get the initial tangent (2.2.5 - 2.2.7).
        mu = self.interface.get_parameter(parameter_name)
        fval = self.interface.rhs(x)
        self.interface.set_parameter(parameter_name, mu + self.delta)
        dmu = (self.interface.rhs(x) - fval) / self.delta
        self.interface.set_parameter(parameter_name, mu)

        # Compute the jacobian at x and solve with it (2.2.5)
        jac = self.interface.jacobian(x)
        dx = -self.interface.solve(jac, dmu)

        # Scaling of the initial tangent (2.2.7)
        dmu = 1
        nrm = numpy.sqrt(self.zeta * dx.dot(dx) + dmu ** 2)
        dmu /= nrm
        dx /= nrm

        eigs = None
        data = Data()
        self.store_data(data, x, mu)

        # Perform the continuation
        for j in range(maxit):
            mu0 = mu

            x, mu, dx, dmu, ds = self.step(parameter_name, x, mu, dx, dmu, ds)

            self.store_data(data, x, mu)

            if (mu >= target and mu0 < target) or (mu <= target and mu0 > target):
                # Converge onto the end point
                x, mu = self.converge(parameter_name, x, mu, dx, dmu, target, ds, maxit)

                self.store_data(data, x, mu)

                return x, mu, data

            if self.parameters.get('Detect Bifurcation Points', False):
                eigs0 = eigs
                eigs = self.interface.eigs(x)

                if eigs[0].real > 0:
                    deigs = eigs - eigs0
                    x, mu = self.detect_bifurcation(parameter_name, x, mu, dx, dmu, eigs, deigs, ds, maxit)

                    self.store_data(data, x, mu)

                    return x, mu, data

            ds = self.adjust_step_size(ds)

        return x, mu, data
