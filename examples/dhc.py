import matplotlib.pyplot as plt

from transiflow import Continuation
from transiflow import Interface
from transiflow import plot_utils
from transiflow import utils


def postprocess(data, interface, x, mu):
    data['Rayleigh Number'].append(mu)
    data['Volume Averaged Kinetic Energy'].append(
        utils.compute_volume_averaged_kinetic_energy(x, interface))


def main(nx=32):
    ''' An example of performing a continuation for a 2D differentially heated cavity and detecting a bifurcation point'''
    ny = nx
    nz = 1

    # Define the problem
    parameters = {'Problem Type': 'Differentially Heated Cavity',
                  # Problem parameters
                  'Rayleigh Number': 1,
                  'Prandtl Number': 1000,
                  'Reynolds Number': 1,
                  # Problem size
                  'X-max': 4.08 / 80,
                  'Y-max': 1,
                  # Give back extra output (this is also more expensive)
                  'Verbose': False}

    interface = Interface(parameters, nx, ny, nz)
    continuation = Continuation(interface, newton_tolerance=1e-7)

    # Compute an initial guess
    x0 = interface.vector()

    # Store data for computing the bifurcation diagram using postprocessing
    data = {'Rayleigh Number': [], 'Volume Averaged Kinetic Energy': []}
    callback = lambda interface, x, mu: postprocess(data, interface, x, mu)

    # Perform an initial continuation to Rayleigh number 1e8 without
    # detecting bifurcation points
    ds = 100
    target = 1e9
    x, mu = continuation.continuation(x0, 'Rayleigh Number', 0, target,
                                      ds, ds_max=1e8, callback=callback)

    parameters['Eigenvalue Solver'] = {}
    parameters['Eigenvalue Solver']['Target'] = 0
    parameters['Eigenvalue Solver']['Tolerance'] = 1e-9
    parameters['Eigenvalue Solver']['Number of Eigenvalues'] = 5

    # Now detect the bifurcation point
    ds = 1e8
    target = 3.5e9
    x2, mu2 = continuation.continuation(x, 'Rayleigh Number', mu, target, ds, ds_max=1e8,
                                        detect_bifurcations=True, callback=callback)

    ke = utils.compute_volume_averaged_kinetic_energy(x2, interface)

    # Compute the unstable branch after the bifurcation
    continuation = Continuation(interface, newton_tolerance=1e-4)
    x3, mu3 = continuation.continuation(x2, 'Rayleigh Number', mu2, target,
                                        ds, ds_max=1e8, callback=callback)

    # Plot a bifurcation diagram
    bif = plt.scatter(mu2, ke, marker='^')
    plt.plot(data['Rayleigh Number'], data['Volume Averaged Kinetic Energy'])

    plt.title('Bifurcation diagram for the differentially heated cavity with $n_x=n_z={}$'.format(nx))
    plt.xlabel('Rayleigh number')
    plt.ylabel('Volume averaged kinetic energy')
    plt.legend([bif], ['First Hopf bifurcation'])
    plt.show()

    # Add a perturbation based on the eigenvector
    interface.set_parameter('Rayleigh Number', mu2)
    _, v = interface.eigs(x2, True)
    v = v[:, 0].real

    # Plot the velocity magnitude
    plot_utils.plot_velocity_magnitude(v, interface, title='Velocity magnitude of the bifurcating eigenvector')

    # Plot the pressure
    v = plot_utils.create_state_mtx(v, nx, ny, nz, interface.discretization.dof)
    plot_utils.plot_value(v[:, :, 0, 2], interface, title='Pressure component of the bifurcating eigenvector')


if __name__ == '__main__':
    main()
