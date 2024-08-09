from transiflow.utils import norm

class TimeIntegration:
    '''Time integration using the theta method.

    Parameters
    ----------
    interface
        Interface object that implements the following functions:
        ``rhs(x)``, ``jacobian(x)``, ``mass_matrix()`` and
        ``solve(jac, rhs)``.
    newton_tolerance : scalar, optional
        Tolerance used in the Newton corrector to determine
        convergence.
    maximum_newton_iterations : int, optional
        Maximum number of Newton iterations.
    residual_check: str, optional
        Method for checking the residual in the Newton method
        (default: 'F').

        - ``F``: Use the norm of the ``rhs(x)`` function.
        - ``dx``: Use the norm of the step ``dx``.
    verbose : bool, optional
        Give extra information about convergence. Since we have to
        compute this information, this may be slower.

    '''

    def __init__(self, interface, theta=1.0,
                 newton_tolerance=1e-10, maximum_newton_iterations=10,
                 residual_check='F', verbose=False):
        self.interface = interface
        self.verbose = verbose
        self.theta = theta

        self.maximum_newton_iterations = maximum_newton_iterations
        self.newton_tolerance = newton_tolerance
        self.residual_check = residual_check

    def _newton(self, x0, dt):
        x = x0
        b0 = self.interface.rhs(x0)
        mass = self.interface.mass_matrix()

        for k in range(self.maximum_newton_iterations):
            # M * u_n + dt * theta * F(u_(n+1)) + dt * (1 - theta) * F(u_n) - M * u_(n+1) = 0
            fval = mass @ (x0 - x) + dt * self.theta * self.interface.rhs(x) + dt * (1 - self.theta) * b0
            fval /= self.theta * dt

            if self.residual_check == 'F' or self.verbose:
                fnorm = norm(fval)

            if self.residual_check == 'F' and fnorm < self.newton_tolerance:
                print('Newton converged i %d iterations with ||F||=%e' % (k, fnorm), flush=True)
                break

            # J - 1 / (theta * dt) * M
            jac = self.interface.jacobian(x) - mass / (self.theta * dt)
            dx = self.interface.solve(jac, -fval)

            x = x + dx

            if self.residual_check != 'F' or self.verbose:
                dxnorm = norm(dx)

            if self.residual_check != 'F' and dxnorm < self.newton_tolerance:
                print('Newton converged in %d iterations with ||dx||=%e' % (k, dxnorm), flush=True)
                break

            if self.verbose:
                print('Newton status at iteration %d: ||F||=%e, ||dx||=%e' % (k, fnorm, dxnorm), flush=True)

        return x

    def integration(self, x0, dt, tmax, callback=None):
        '''
        Perform the time integration.

        Parameters
        ----------
        x0 : array_like
            Initial solution.
        dt : scalar
            Time step.
        tmax : scalar
            End time.
        callback : function, optional
            User-supplied function to call after each continuation
            step. It is called as ``callback(interface, x, t)``.

        Returns
        -------
        x : array_like
            Solution at the end time
        t : scalar
            End time

        '''

        x = x0
        t = 0

        if callback:
            callback(self.interface, x, t)

        while t < tmax:
            x = self._newton(x, dt)
            t += dt

            print("t = %f" % t, flush=True)

            if callback:
                callback(self.interface, x, t)

        return x, t
