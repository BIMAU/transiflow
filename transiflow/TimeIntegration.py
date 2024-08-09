from transiflow.utils import norm

class TimeIntegration:
    '''Time integration using the theta method.

    Parameters
    ----------
    interface
        Interface object that implements the following functions:
        ``rhs(x)``, ``jacobian(x)``, ``mass_matrix()`` and
        ``solve(jac, rhs)``.
    residual_check: str, optional
        Method for checking the residual in the Newton method
        (default: 'F').

        - ``F``: Use the norm of the ``rhs(x)`` function.
        - ``dx``: Use the norm of the step ``dx``.
    verbose : bool, optional
        Give extra information about convergence. Since we have to
        compute this information, this may be slower.

    '''

    def __init__(self, interface, parameters, residual_check='F', verbose=False):
        self.interface = interface
        self.parameters = parameters
        self.verbose = verbose

        self.residual_check = residual_check

    def newton(self, x0, dt):
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

            if self.residual_check == 'F' or self.verbose:
                fnorm = norm(fval)

            if self.residual_check == 'F' and fnorm < tol:
                print('Newton converged in %d iterations with ||F||=%e' % (k, fnorm), flush=True)
                break

            # J - 1 / (theta * dt) * M
            jac = self.interface.jacobian(x) - mass / (theta * dt)
            dx = self.interface.solve(jac, -fval)

            x = x + dx

            if self.residual_check != 'F' or self.verbose:
                dxnorm = norm(dx)

            if self.residual_check != 'F' and dxnorm < tol:
                print('Newton converged in %d iterations with ||dx||=%e' % (k, dxnorm), flush=True)
                break

            if self.verbose:
                print('Newton status at iteration %d: ||F||=%e, ||dx||=%e' % (k, fnorm, dxnorm), flush=True)

        return x

    def postprocess(self, x, t):
        if 'Postprocess' in self.parameters and self.parameters['Postprocess']:
            self.parameters['Postprocess'](self.interface, x, t)

    def integration(self, x0, dt, tmax):
        '''
        Perform the time integration.

        Parameters
        ----------
        x0 : array_like
            Initial solution.
        dt : scalar
            Time step.
        tmax : scalar
            End time

        Returns
        -------
        x : array_like
            Solution at the end time
        t : scalar
            End time

        '''

        x = x0
        t = 0

        self.postprocess(x, t)

        while t < tmax:
            x = self.newton(x, dt)
            t += dt

            print("t = %f" % t, flush=True)

            self.postprocess(x, t)

        return x, t
