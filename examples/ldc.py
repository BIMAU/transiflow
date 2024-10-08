import matplotlib.pyplot as plt

from transiflow import Continuation
from transiflow import Interface
from transiflow import plot_utils
from transiflow import utils


def postprocess(data, interface, x, mu):
    data['Reynolds Number'].append(mu)
    data['Volume Averaged Kinetic Energy'].append(
        utils.compute_volume_averaged_kinetic_energy(x, interface))


def main(nx=32):
    ''' An example of performing a continuation for a 2D lid-driven cavity and detecting a bifurcation point'''
    ny = nx
    nz = 1

    # Define the problem
    parameters = {'Problem Type': 'Lid-driven Cavity',
                  # Problem parameters
                  'Reynolds Number': 1,
                  'Lid Velocity': 0,
                  # Use a stretched grid
                  'Grid Stretching Factor': 1.5,
                  # Give back extra output (this is also more expensive)
                  'Verbose': False}

    interface = Interface(parameters, nx, ny, nz)
    continuation = Continuation(interface)

    # Compute an initial guess
    x0 = interface.vector()
    x0 = continuation.continuation(x0, 'Lid Velocity', 0, 1, 1)[0]

    # Store data for computing the bifurcation diagram using postprocessing
    data = {'Reynolds Number': [], 'Volume Averaged Kinetic Energy': []}
    callback = lambda interface, x, mu: postprocess(data, interface, x, mu)

    # Perform an initial continuation to Reynolds number 7000 without detecting bifurcation points
    ds = 100
    target = 6000
    x, mu = continuation.continuation(x0, 'Reynolds Number', 1, target, ds,
                                      callback=callback)

    # Now detect the bifurcation point
    parameters['Eigenvalue Solver'] = {}
    parameters['Eigenvalue Solver']['Target'] = 3j
    parameters['Eigenvalue Solver']['Tolerance'] = 1e-9
    parameters['Eigenvalue Solver']['Number of Eigenvalues'] = 5

    bifurcation_continuation = Continuation(interface, newton_tolerance=1e-12)

    target = 10000
    x2, mu2 = bifurcation_continuation.continuation(x, 'Reynolds Number', mu, target,
                                                    ds, ds_max=100, detect_bifurcations=True,
                                                    callback=callback)

    ke = utils.compute_volume_averaged_kinetic_energy(x2, interface)

    # Compute the unstable branch after the bifurcation
    target = 10000
    x3, mu3 = continuation.continuation(x2, 'Reynolds Number', mu2, target, ds,
                                        callback=callback)

    # Plot a bifurcation diagram
    bif = plt.scatter(mu2, ke, marker='^')
    plt.plot(data['Reynolds Number'], data['Volume Averaged Kinetic Energy'])

    plt.title('Bifurcation diagram for the lid-driven cavity with $n_x=n_z={}$'.format(nx))
    plt.xlabel('Reynolds number')
    plt.ylabel('Volume averaged kinetic energy')
    plt.legend([bif], ['First Hopf bifurcation'])
    plt.show()

    # Add a perturbation based on the eigenvector
    interface.set_parameter('Reynolds Number', mu2)
    _, v = interface.eigs(x2, True)
    v = v[:, 0].real

    # Plot the velocity magnitude
    plot_utils.plot_velocity_magnitude(v, interface, title='Velocity magnitude of the bifurcating eigenvector')

    # Plot the pressure
    v = plot_utils.create_state_mtx(v, nx, ny, nz, interface.discretization.dof)
    plot_utils.plot_value(v[:, :, 0, 2], interface, title='Pressure component of the bifurcating eigenvector')


if __name__ == '__main__':
    main()
