import math
import numpy

from transiflow.utils import norm

class Continuation:
    '''Pseudo-arclength continuation of an abstract interface that can
    return a Jacobian matrix and a right-hand side.

    Parameters
    ----------
    interface
        Interface object that implements the following functions:
        ``rhs(x)``, ``jacobian(x)``, ``solve(jac, rhs)``, ``eigs(x)``
        and ``set_parameter(name, value)``.
    delta : scalar, optional
        Parameter value difference used when computing the approximate
        derivative.
    destination_tolerance : scalar, optional
        Tolerance used for the continuation parameter when converging
        onto a bifurcation point.
    newton_tolerance : scalar, optional
        Tolerance used in the Newton corrector to determine
        convergence.
    maximum_newton_iterations : int, optional
        Maximum number of Newton iterations.
    optimal_newton_iterations : int, optional
        Number of Newton iterations in which we try to converge by
        adjusting ds.
    residual_check: str, optional
        Method for checking the residual in the Newton method
        (default: 'F').

        - ``F``: Use the norm of the ``rhs(x)`` function.
        - ``dx``: Use the norm of the step ``dx``.
    bordered_solver : bool, optional
        Use a bordered solver the entire linear system in the
        correction equation in a single operation, instead of using
        two separate solves. This has to be implemented by the
        interface.
    verbose : bool, optional
        Give extra information about convergence. Since we have to
        compute this information, this may be slower.

    '''

    def __init__(self, interface,
                 delta=1.0, destination_tolerance=1e-4,
                 newton_tolerance=1e-4, maximum_newton_iterations=10,
                 optimal_newton_iterations=3, residual_check='F',
                 bordered_solver=False, verbose=False):
        self.interface = interface
        self.verbose = verbose

        self.destination_tolerance = destination_tolerance
        self.delta = delta
        self.zeta = None

        self.maximum_newton_iterations = maximum_newton_iterations
        self.optimal_newton_iterations = optimal_newton_iterations
        self.newton_tolerance = newton_tolerance
        self.newton_iterations = 0
        self.residual_check = residual_check

        self.bordered_solver = bordered_solver

    def newton(self, x0, tol=1e-10):
        '''Newton method that can be used to converge to a steady
        state that is very nearby. This can for instance be used to
        obtain an initial guess.

        Parameters
        ----------
        x0 : array_like
            Initial solution.
        tol : scalar, optional
            Convergence tolerance

        Returns
        -------
        x : array_like
            Root at the current parameter value.

        '''

        x = x0
        for k in range(self.maximum_newton_iterations):
            fval = self.interface.rhs(x)

            if self.residual_check == 'F' or self.verbose:
                fnorm = norm(fval)

            if self.residual_check == 'F' and fnorm < tol:
                print('Newton converged in %d iterations with ||F||=%e' % (k, fnorm), flush=True)
                break

            jac = self.interface.jacobian(x)
            dx = self.interface.solve(jac, -fval)

            x = x + dx

            if self.residual_check != 'F' or self.verbose:
                dxnorm = norm(dx)

            if self.residual_check != 'F' and dxnorm < tol:
                print('Newton converged in %d iterations with ||dx||=%e' % (k, dxnorm), flush=True)
                break

            if self.verbose:
                print('Newton status at iteration %d: ||F||=%e, ||dx||=%e' % (k, fnorm, dxnorm), flush=True)

        self.newton_iterations = k

        return x

    def _newton_corrector(self, parameter_name, ds, x, x0, mu, mu0):
        self.newton_iterations = 0

        fnorm = None
        dxnorm = None

        # Do the main iteration
        for k in range(self.maximum_newton_iterations):
            # Compute F (RHS of 2.2.9)
            self.interface.set_parameter(parameter_name, mu)
            fval = self.interface.rhs(x)

            if self.residual_check == 'F' or self.verbose:
                prev_norm = fnorm
                fnorm = norm(fval)

            if self.residual_check == 'F' and fnorm < self.newton_tolerance:
                print('Newton corrector converged in %d iterations with ||F||=%e' % (k, fnorm), flush=True)
                break

            if self.residual_check == 'F' and prev_norm is not None and prev_norm < fnorm:
                self.newton_iterations = self.maximum_newton_iterations
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

            if self.bordered_solver:
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

            if self.residual_check != 'F' or self.verbose:
                prev_norm = dxnorm
                dxnorm = norm(dx)

            if self.residual_check != 'F' and dxnorm < self.newton_tolerance:
                print('Newton corrector converged in %d iterations with ||dx||=%e' % (k, dxnorm), flush=True)
                break

            if self.verbose:
                print('Newton corrector status at iteration %d: ||F||=%e, ||dx||=%e' % (k, fnorm, dxnorm), flush=True)

            if self.residual_check != 'F' and prev_norm is not None and prev_norm < dxnorm:
                self.newton_iterations = self.maximum_newton_iterations
                break

        if self.newton_iterations == self.maximum_newton_iterations:
            print('Newton did not converge. Adjusting step size and trying again', flush=True)
            return x0, mu0

        self.interface.set_parameter(parameter_name, mu)

        return x, mu

    def _adjust_step_size(self, ds, ds_min, ds_max):
        ''' Step size control, see [Seydel p 188.] '''
        factor = self.optimal_newton_iterations / max(self.newton_iterations, 1)
        factor = min(max(factor, 0.5), 2.0)

        ds *= factor

        ds = math.copysign(min(max(abs(ds), ds_min), ds_max), ds)

        if self.verbose:
            print('New stepsize: ds=%e, factor=%e' % (ds, factor), flush=True)

        return ds

    def _detect_bifurcation(self, parameter_name, x, mu, dx, dmu, eigs, deig, v,
                            ds, ds_min, ds_max, maxit):
        ''' Converge onto a bifurcation '''

        for j in range(maxit):
            i = numpy.argmin(numpy.abs(eigs.real))
            if abs(eigs[i].real) < self.destination_tolerance:
                print("Bifurcation found at %s = %f with eigenvalue %e + %ei" % (
                    parameter_name, mu, eigs[i].real, eigs[i].imag), flush=True)
                break

            # Secant method
            ds = ds / deig.real * -eigs.real[i]
            x, mu, dx, dmu, ds = self._step(parameter_name, x, mu, dx, dmu,
                                            ds, ds_min, ds_max)

            prev_eigs = eigs
            eigs, v = self.interface.eigs(x, return_eigenvectors=True, enable_recycling=True)
            i = numpy.argmin(numpy.abs(eigs.real))
            deig = eigs[i] - prev_eigs[i]

        return x, mu, v

    def _converge(self, parameter_name, x, mu, dx, dmu, target,
                  ds, ds_min, ds_max, maxit):
        ''' Converge onto the target value '''

        for j in range(maxit):
            if abs(target - mu) < self.destination_tolerance:
                self.interface.set_parameter(parameter_name, target)
                print("Convergence achieved onto target %s = %f" % (parameter_name, mu), flush=True)
                break

            # Secant method
            ds = 1 / dmu * (target - mu)
            x, mu, dx, dmu, ds = self._step(parameter_name, x, mu, dx, dmu,
                                            ds, ds_min, ds_max)

        return x, mu

    def _step(self, parameter_name, x, mu, dx, dmu,
              ds, ds_min, ds_max):
        ''' Perform one step of the continuation '''

        mu0 = mu
        x0 = x

        # Predictor (2.2.3)
        mu = mu0 + ds * dmu
        x = x0 + ds * dx

        # Corrector (2.2.9 and onward)
        x, mu = self._newton_corrector(parameter_name, ds, x, x0, mu, mu0)

        if mu == mu0:
            # No convergence was achieved, adjusting the step size
            prev_ds = ds
            ds = self._adjust_step_size(ds, ds_min, ds_max)
            if prev_ds == ds:
                raise Exception('Newton cannot achieve convergence')

            return self._step(parameter_name, x0, mu0, dx, dmu,
                              ds, ds_min, ds_max)

        print("%s: %f" % (parameter_name, mu), flush=True)

        # Set the new values computed by the corrector
        dmu = mu - mu0
        dx = x - x0

        if abs(dmu) < 1e-12:
            raise Exception('dmu too small')

        # Compute the tangent (2.2.4)
        dx /= ds
        dmu /= ds

        return x, mu, dx, dmu, ds

    def _switch_branches_tangent(self, parameter_name, x, mu, dx, dmu, v,
                                 ds, ds_min, ds_max):
        ''' Switch branches according to (5.16) '''

        dmu0 = dmu

        # Compute the F(x) and F_x(x)
        fval = self.interface.rhs(x)
        jac = self.interface.jacobian(x)

        # Compute F_mu(x)
        self.interface.set_parameter(parameter_name, mu + self.delta)
        dflval = (self.interface.rhs(x) - fval) / self.delta
        self.interface.set_parameter(parameter_name, mu)

        if self.bordered_solver:
            # Solve the entire bordered system in one go (5.16)
            dx, dmu = self.interface.solve(jac, 0 * x, 0, dflval, dx, dmu)
        else:
            # Solve with F_x (equation below 5.16)
            z = self.interface.solve(jac, dflval)
            dmu = -dx.dot(v) / (dmu - dx.dot(z))
            dx = v - dmu * z

        if abs(dmu) < 1e-12:
            dmu = dmu0
            ds = ds_min

        return x, mu, dx, dmu, ds

    def _switch_branches(self, parameter_name, x, mu, dx, dmu, v,
                         ds, ds_min, ds_max):
        return self._switch_branches_tangent(parameter_name, x, mu, dx, dmu, v,
                                             ds, ds_min, ds_max)

    def _num_positive_eigs(self, eigs):
        # Include the range of the destination tolerance here to make sure
        # we don't converge onto the same target twice
        return sum([eig.real > -self.destination_tolerance for eig in eigs])

    def _initial_tangent(self, x, parameter_name, mu):
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
        nrm = math.sqrt(self.zeta * dx.dot(dx) + dmu ** 2)
        dmu /= nrm
        dx /= nrm

        return dx, dmu

    def _return(self, x, mu, dx, dmu, ds, callback, return_step):
        if callback is not None:
            callback(self.interface, x, mu)

        if return_step:
            return x, mu, dx * ds, dmu * ds

        return x, mu

    def continuation(self, x0, parameter_name, start, target,
                     ds, ds_min=0.01, ds_max=1000,
                     dx=None, dmu=None, maxit=1000,
                     detect_bifurcations=False, switch_branches=False,
                     return_step=False, callback=None):
        '''Perform a pseudo-arclength continuation in
        ``parameter_name`` from parameter value start to ``target``
        with arclength step size ``ds``, and starting from an initial
        state ``x0``.

        Returns the final state x and the final parameter value mu.

        Parameters
        ----------
        x0 : array_like
            Initial solution.
        parameter_name : string
            Name of the parameter we want to perform the continuation
            in.
        start : scalar
            Starting value of the continuation parameter.
        target : scalar
            Target value of the continuation parameter.
        ds : scalar
            Arclength step size.
        ds_min : scalar, optional
            Minimum arclength step size.
        ds_max : scalar, optional
            Maximum arclength step size.
        dx : array_like, optional
            Vector difference defining the initial tangent.
        dmu : scalar, optional
            Parameter difference defining the initial tangent.
        maxit : int, optional
            Maximum number of continuation iterations.
        detect_bifurcations : bool, optional
            Detect bifurcations by detecting a switch in eigenvalue
            signs. Note that this is very expensive.
        switch_branches : bool, optional
            Switch branches using the tangent. This usually doesn't
            work and is currently untested.
        return_step : bool, optional
            Return ``dx`` and ``dmu`` when set to True. These can be
            used in the next ``continuation()`` call.
        callback : function, optional
            User-supplied function to call after each continuation
            step. It is called as ``callback(interface, x, mu)``.

        Returns
        -------
        x : array_like
            Value at the target or bifurcation point if were are
            detecting bifurcation points.
        mu : scalar
            Value of the bifurcation parameter at the end of the
            continuation.
        dx : array_like, optional
            Array to pass back to ``continuation()`` which is returned
            if ``return_step`` is set to True.
        dmu : scalar, optional
            Value to pass back to ``continuation()`` which is returned
            if ``return_step`` is set to True.

        '''

        x = x0
        mu = start

        self.zeta = 1 / x.size

        if dx is None or dmu is None:
            # Get the initial tangent (2.2.5 - 2.2.7).
            dx, dmu = self._initial_tangent(x, parameter_name, mu)
        else:
            dx /= ds
            dmu /= ds

        switched_branches = False
        eigs = None

        # Some configuration for the detection of bifurcations
        enable_recycling = False

        # Perform the continuation
        for j in range(maxit):
            mu0 = mu

            if detect_bifurcations or (switch_branches and not switched_branches):
                prev_eigs = eigs
                eigs, v = self.interface.eigs(x, return_eigenvectors=True, enable_recycling=enable_recycling)
                enable_recycling = True

                if prev_eigs is not None and self._num_positive_eigs(eigs) != self._num_positive_eigs(prev_eigs):
                    i = numpy.argmin(numpy.abs(eigs.real))
                    deig = eigs[i] - prev_eigs[i]
                    x, mu, v = self._detect_bifurcation(parameter_name, x, mu, dx, dmu, eigs, deig, v,
                                                        ds, ds_min, ds_max, maxit - j)

                    if switch_branches and not switched_branches:
                        switched_branches = True
                        x, mu, dx, dmu, ds = self._switch_branches(parameter_name, x, mu, dx, dmu, v[:, 0].real,
                                                                   ds, ds_min, ds_max)
                        continue

                    return self._return(x, mu, dx, dmu, ds, callback, return_step)

            if callback is not None:
                callback(self.interface, x, mu)

            x, mu, dx, dmu, ds = self._step(parameter_name, x, mu, dx, dmu,
                                            ds, ds_min, ds_max)

            if (mu >= target and mu0 < target) or (mu <= target and mu0 > target):
                # Converge onto the end point
                x, mu = self._converge(parameter_name, x, mu, dx, dmu, target,
                                       ds, ds_min, ds_max, maxit - j)

                return self._return(x, mu, dx, dmu, ds, callback, return_step)

            ds = self._adjust_step_size(ds, ds_min, ds_max)

        return self._return(x, mu, dx, dmu, ds, callback, return_step)
