import sys

from fvm.utils import norm

class TimeIntegration:
    def __init__(self, interface, parameters):
        self.interface = interface
        self.parameters = parameters

    def newton(self, x0, dt):
        residual_check = self.parameters.get('Residual Check', 'F')
        verbose = self.parameters.get('Verbose', False)
        theta = self.parameters.get('Theta', 1)

        # Set Newton some parameters
        maxit = self.parameters.get('Maximum Newton Iterations', 10)
        tol = self.parameters.get('Newton Tolerance', 1e-10)

        x = x0
        b0 = self.interface.rhs(x0)
        mass = self.interface.mass_matrix()

        for k in range(maxit):
            # M * u_n + dt * theta * F(u_(n+1)) + dt * (1 - theta) * F(u_n) - M * u_(n+1) = 0
            fval = mass @ (x0 - x) + dt * theta * self.interface.rhs(x) + dt * (1 - theta) * b0
            fval /= theta * dt

            if residual_check == 'F' or verbose:
                fnorm = norm(fval)

            if residual_check == 'F' and fnorm < tol:
                print('Newton converged in %d iterations with ||F||=%e' % (k, fnorm))
                sys.stdout.flush()
                break

            # J - 1 / (theta * dt) * M
            jac = self.interface.jacobian(x) - mass / (theta * dt)
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

        return x

    def postprocess(self, x, t):
        if 'Postprocess' in self.parameters and self.parameters['Postprocess']:
            self.parameters['Postprocess'](self.interface, x, t)

    def integration(self, x0, dt, tmax):
        x = x0
        t = 0

        self.postprocess(x, t)

        while t < tmax:
            x = self.newton(x, dt)
            t += dt

            print("t = %f" % t)
            sys.stdout.flush()

            self.postprocess(x, t)

        return x, t
