import math

from math import sqrt
from transiflow.utils import norm

class Continuation:

    def __init__(self, interface, parameters):
        self.interface = interface
        self.parameters = parameters

        self.newton_iterations = 0
        self.delta = None
        self.zeta = None

    def newton(self, x0):
        residual_check = self.parameters.get('Residual Check', 'F')
        verbose = self.parameters.get('Verbose', False)

        # Set Newton some parameters
        maxit = self.parameters.get('Maximum Newton Iterations', 10)
        tol = self.parameters.get('Newton Tolerance', 1e-10)

        x = x0
        for k in range(maxit):
            fval = self.interface.rhs(x)

            if residual_check == 'F' or verbose:
                fnorm = norm(fval)

            if residual_check == 'F' and fnorm < tol:
                print('Newton converged in %d iterations with ||F||=%e' % (k, fnorm), flush=True)
                break

            jac = self.interface.jacobian(x)
            dx = self.interface.solve(jac, -fval)

            x = x + dx

            if residual_check != 'F' or verbose:
                dxnorm = norm(dx)

            if residual_check != 'F' and dxnorm < tol:
                print('Newton converged in %d iterations with ||dx||=%e' % (k, dxnorm), flush=True)
                break

            if verbose:
                print('Newton status at iteration %d: ||F||=%e, ||dx||=%e' % (k, fnorm, dxnorm), flush=True)

        self.newton_iterations = k

        return x

    def newtoncorrector(self, parameter_name, ds, x, x0, mu, mu0):
        residual_check = self.parameters.get('Residual Check', 'F')
        verbose = self.parameters.get('Verbose', False)

        # Set Newton some parameters
        maxit = self.parameters.get('Maximum Newton Iterations', 10)
        tol = self.parameters.get('Newton Tolerance', 1e-4)

        self.newton_iterations = 0

        fnorm = None
        dxnorm = None

        # Do the main iteration
        for k in range(maxit):
            # Compute F (RHS of 2.2.9)
            self.interface.set_parameter(parameter_name, mu)
            fval = self.interface.rhs(x)

            if residual_check == 'F' or verbose:
                prev_norm = fnorm
                fnorm = norm(fval)

            if residual_check == 'F' and fnorm < tol:
                print('Newton corrector converged in %d iterations with ||F||=%e' % (k, fnorm), flush=True)
                break

            if residual_check == 'F' and prev_norm is not None and prev_norm < fnorm:
                self.newton_iterations = maxit
                break

            # Compute r (2.2.8)
            diff = x - x0
            rnp1 = self.zeta * diff.dot(diff) + (1 - self.zeta) * (mu - mu0) ** 2 - ds ** 2

            # Compute F_mu (LHS of 2.2.9)
            self.interface.set_parameter(parameter_name, mu + self.delta)
            dflval = (self.interface.rhs(x) - fval) / self.delta
            self.interface.set_parameter(parameter_name, mu)

            # Compute the jacobian F_x at x (LHS of 2.2.9)
            jac = self.interface.jacobian(x)

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
                prev_norm = dxnorm
                dxnorm = norm(dx)

            if residual_check != 'F' and dxnorm < tol:
                print('Newton corrector converged in %d iterations with ||dx||=%e' % (k, dxnorm), flush=True)
                break

            if verbose:
                print('Newton corrector status at iteration %d: ||F||=%e, ||dx||=%e' % (k, fnorm, dxnorm), flush=True)

            if residual_check != 'F' and prev_norm is not None and prev_norm < dxnorm:
                self.newton_iterations = maxit
                break

        if self.newton_iterations == maxit:
            print('Newton did not converge. Adjusting step size and trying again', flush=True)
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

        ds = math.copysign(min(max(abs(ds), min_step_size), max_step_size), ds)

        if self.parameters.get('Verbose', False):
            print('New stepsize: ds=%e, factor=%e' % (ds, factor), flush=True)

        return ds

    def detect_bifurcation(self, parameter_name, x, mu, dx, dmu, eig, deig, v, ds, maxit):
        ''' Converge onto a bifurcation '''

        tol = self.parameters.get('Destination Tolerance', 1e-4)

        for j in range(maxit):
            if abs(eig.real) < tol:
                print("Bifurcation found at %s = %f with eigenvalue %e + %ei" % (
                    parameter_name, mu, eig.real, eig.imag), flush=True)
                break

            # Secant method
            ds = ds / deig.real * -eig.real
            x, mu, dx, dmu, ds = self.step(parameter_name, x, mu, dx, dmu, ds)

            eig_prev = eig
            eigs, v = self.interface.eigs(x, return_eigenvectors=True, enable_recycling=True)
            eig = eigs[0]
            deig = eig - eig_prev

        return x, mu, v

    def converge(self, parameter_name, x, mu, dx, dmu, target, ds, maxit):
        ''' Converge onto the target value '''

        tol = self.parameters.get('Destination Tolerance', 1e-4)

        for j in range(maxit):
            if abs(target - mu) < tol:
                self.interface.set_parameter(parameter_name, target)
                print("Convergence achieved onto target %s = %f" % (parameter_name, mu), flush=True)
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

        print("%s: %f" % (parameter_name, mu), flush=True)

        if 'Postprocess' in self.parameters and self.parameters['Postprocess']:
            self.parameters['Postprocess'](self.interface, x, mu)

        # Set the new values computed by the corrector
        dmu = mu - mu0
        dx = x - x0

        if abs(dmu) < 1e-12:
            raise Exception('dmu too small')

        # Compute the tangent (2.2.4)
        dx /= ds
        dmu /= ds

        return x, mu, dx, dmu, ds

    def switch_branches_tangent(self, parameter_name, x, mu, dx, dmu, v, ds):
        ''' Switch branches according to (5.16) '''

        dmu0 = dmu

        # Compute the F(x) and F_x(x)
        fval = self.interface.rhs(x)
        jac = self.interface.jacobian(x)

        # Compute F_mu(x)
        self.interface.set_parameter(parameter_name, mu + self.delta)
        dflval = (self.interface.rhs(x) - fval) / self.delta
        self.interface.set_parameter(parameter_name, mu)

        if self.parameters.get("Bordered Solver", False):
            # Solve the entire bordered system in one go (5.16)
            dx, dmu = self.interface.solve(jac, 0 * x, 0, dflval, dx, dmu)
        else:
            # Solve with F_x (equation below 5.16)
            z = self.interface.solve(jac, dflval)
            dmu = -dx.dot(v) / (dmu - dx.dot(z))
            dx = v - dmu * z

        if abs(dmu) < 1e-12:
            dmu = dmu0
            ds = self.parameters.get('Minimum Step Size', 0.01)

        return x, mu, dx, dmu, ds

    def switch_branches_asymmetry(self, parameter_name, x, mu, ds):
        continuation = Continuation(self.interface, self.parameters)
        x, a = continuation.continuation(x, 'Asymmetry Parameter', 0, 1000, 10, 1, switched_branches=True)
        x, mu = continuation.continuation(x, parameter_name, mu, mu + 1, ds, switched_branches=True)
        x, a = continuation.continuation(x, 'Asymmetry Parameter', a, 0, -a, switched_branches=True)

        dx, dmu = self.initial_tangent(x, parameter_name, mu)

        return x, mu, dx, dmu, ds

    def switch_branches(self, parameter_name, x, mu, dx, dmu, v, ds):
        branch_switching_method = self.parameters.get('Branch Switching Method', 'Tangent')
        if branch_switching_method == 'Asymmetry':
            return self.switch_branches_asymmetry(parameter_name, x, mu, ds)

        return self.switch_branches_tangent(parameter_name, x, mu, dx, dmu, v, ds)

    def initial_tangent(self, x, parameter_name, mu):
        ''' Compute the initial tangent '''

        # Get the initial tangent (2.2.5 - 2.2.7).
        self.interface.set_parameter(parameter_name, mu + self.delta)
        dflval = self.interface.rhs(x)
        self.interface.set_parameter(parameter_name, mu)
        fval = self.interface.rhs(x)
        dflval = (dflval - fval) / self.delta

        # Compute the jacobian at x and solve with it (2.2.5)
        jac = self.interface.jacobian(x)
        dx = self.interface.solve(jac, -dflval)

        # Scaling of the initial tangent (2.2.7)
        dmu = 1
        nrm = sqrt(self.zeta * dx.dot(dx) + dmu ** 2)
        dmu /= nrm
        dx /= nrm

        return dx, dmu

    def continuation(self, x0, parameter_name, start, target, ds,
                     dx=None, dmu=None,
                     maxit=None, switched_branches=False):
        '''Perform a pseudo-arclength continuation in parameter_name from
        parameter value start to target with arclength step size ds,
        and starting from an initial state x0.

        Returns the final state x and the final parameter value mu.

        Postprocessing can be done by setting the 'Postprocess'
        parameter to a lambda x, mu: ... function, which gets called
        after every continuation step.

        A bifurcation can be detected automatically when the 'Detect
        Bifurcation Points' parameter is set to True.

        '''

        x = x0
        mu = start

        # Set some parameters
        self.delta = self.parameters.get('Delta', 1)
        self.zeta = 1 / x.size

        if not dx or not dmu:
            # Get the initial tangent (2.2.5 - 2.2.7).
            dx, dmu = self.initial_tangent(x, parameter_name, mu)
        else:
            dx /= ds
            dmu /= ds

        eig = None

        if not maxit:
            maxit = self.parameters.get('Maximum Continuation Steps', 1000)

        # Some configuration for the detection of bifurcations
        detect_bifurcations = self.parameters.get('Detect Bifurcation Points', False)
        enable_branch_switching = self.parameters.get('Enable Branch Switching', False)
        enable_recycling = False

        # Perform the continuation
        for j in range(maxit):
            mu0 = mu

            x, mu, dx, dmu, ds = self.step(parameter_name, x, mu, dx, dmu, ds)

            if detect_bifurcations or (enable_branch_switching and not switched_branches):
                eig_prev = eig
                eigs, v = self.interface.eigs(x, return_eigenvectors=True, enable_recycling=enable_recycling)
                eig = eigs[0]
                enable_recycling = True

                if eig_prev is not None and eig.real * eig_prev.real < 0:
                    deig = eig - eig_prev
                    x, mu, v = self.detect_bifurcation(parameter_name, x, mu, dx, dmu, eig, deig, v, ds, maxit - j)

                    if enable_branch_switching and not switched_branches:
                        switched_branches = True
                        x, mu, dx, dmu, ds = self.switch_branches(parameter_name, x, mu, dx, dmu, v[:, 0].real, ds)
                        continue

                    return x, mu

                if eig_prev is None and eig.real > 0:
                    # We're past the bifurcation already, so go backwards
                    ds = -ds

            if (mu >= target and mu0 < target) or (mu <= target and mu0 > target):
                # Converge onto the end point
                x, mu = self.converge(parameter_name, x, mu, dx, dmu, target, ds, maxit - j)

                return x, mu

            ds = self.adjust_step_size(ds)

        return x, mu
