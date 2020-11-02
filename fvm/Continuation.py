from math import sqrt

def norm(x):
    return sqrt(x.dot(x))

def newton(interface, x0, tol=1.e-7, maxit=1000):
    x = x0
    for k in range(maxit):
        fval = interface.rhs(x)
        jac = interface.jacobian(x)
        dx = interface.solve(jac, -fval)

        x = x + dx

        dxnorm = norm(dx)
        if dxnorm < tol:
            print('Newton converged in %d steps with norm %e' % (k, dxnorm))
            break

    return x

def newtoncorrector(interface, parameter_name, ds, x, x0, l, l0, tol):
    # Set some parameters
    maxit = 100
    zeta = 1 / len(x)
    delta = 1

    # Do the main iteration
    for k in range(maxit):
        # Compute F amd F_mu (RHS of 2.2.9)
        interface.set_parameter(parameter_name, l + delta)
        dflval = interface.rhs(x)
        interface.set_parameter(parameter_name, l)
        fval = interface.rhs(x)
        dflval = (dflval - fval) / delta

        # Compute the jacobian at x
        jac = interface.jacobian(x)

        # Solve twice with F_x (2.2.9)
        z1 = interface.solve(jac, -fval)
        z2 = interface.solve(jac, dflval)

        # Compute r (2.2.8)
        diff = x - x0
        rnp1 = zeta*diff.dot(diff) + (1-zeta)*(l-l0)**2 - ds**2

        # Compute dl (2.2.13)
        dl = (-rnp1 - 2*zeta*diff.dot(z1)) / (2*(1-zeta)*(l-l0) - 2*zeta*diff.dot(z2))

        # Compute dx (2.2.12)
        dx = z1 - dl*z2

        # Compute a new x and l (2.2.10 - 2.2.11)
        x = x + dx
        l = l + dl

        dxnorm = norm(dx)
        if dxnorm < tol:
            print('Newton corrector converged in %d steps with norm %e' % (k, dxnorm))
            return (x, l)

    print('No convergence achieved by Newton corrector')

def continuation(interface, x0, parameter_name, target, ds, maxit):
    x = x0

    # Get the initial tangent (2.2.5 - 2.2.7). 'l' is called mu in Erik's thesis.
    delta = 1
    l = interface.get_parameter(parameter_name)
    fval = interface.rhs(x)
    interface.set_parameter(parameter_name, l + delta)
    dl = (interface.rhs(x) - fval) / delta
    interface.set_parameter(parameter_name, l)

    # Compute the jacobian at x and solve with it (2.2.5)
    jac = interface.jacobian(x)
    dx = -interface.solve(jac, dl)

    # Scaling of the initial tangent (2.2.7)
    dl = 1
    zeta = 1 / len(x)
    nrm = sqrt(zeta * dx.dot(dx) + dl**2)
    dl = dl / nrm
    dx = dx / nrm

    dl0 = dl
    dx0 = dx

    # Perform the continuation
    for j in range(maxit):
        l0 = l
        x0 = x

        # Predictor (2.2.3)
        l = l0 + ds * dl0
        x = x0 + ds * dx0

        # Corrector (2.2.9 and onward)
        x2, l2 = newtoncorrector(interface, parameter_name, ds, x, x0, l, l0, 1e-4)

        print("%s: %f" % (parameter_name, l2))

        if (l2 >= target and l0 < target) or (l2 <= target and l0 > target):
            # Converge onto the end point (we usually go past it, so we
            # use Newton to converge)
            l = target
            interface.set_parameter(parameter_name, l)
            x = newton(interface, x, 1e-4)

            return x

        # Set the new values computed by the corrector
        dl = l2 - l0
        l = l2
        dx = x2 - x0
        x = x2

        if abs(dl) < 1e-10:
            return

        # Compute the tangent (2.2.4)
        dx0 = dx / ds
        dl0 = dl / ds

    return x
