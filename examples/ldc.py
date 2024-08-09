import matplotlib.pyplot as plt

from transiflow import Continuation
from transiflow import Interface
from transiflow import plot_utils
from transiflow import utils


class Data:
    def __init__(self):
        self.mu = []
        self.value = []

    def append(self, mu, value):
        self.mu.append(mu)
        self.value.append(value)

    def filter(self):
        '''Filter out values obtained while converging onto a target'''
        idx = []
        for i, mu in enumerate(self.mu):
            if idx:
                idx = [j for j in idx if self.mu[j] < mu]

            idx.append(i)

        self.mu = [self.mu[i] for i in idx]
        self.value = [self.value[i] for i in idx]

def main():
    ''' An example of performing a continuation for a 2D lid-driven cavity and detecting a bifurcation point'''
    dim = 2
    dof = 3
    nx = 32
    ny = nx
    nz = 1

    # Define the problem
    parameters = {'Problem Type': 'Lid-driven Cavity',
                  # Problem parameters
                  'Reynolds Number': 1,
                  'Lid Velocity': 0,
                  # Use a stretched grid
                  'Grid Stretching Factor': 1.5,
                  # Set a maximum step size ds
                  'Maximum Step Size': 500,
                  # Give back extra output (this is also more expensive)
                  'Verbose': False}

    interface = Interface(parameters, nx, ny, nz, dim, dof)
    continuation = Continuation(interface, parameters)

    # Compute an initial guess
    x0 = interface.vector()
    x0 = continuation.continuation(x0, 'Lid Velocity', 0, 1, 1)[0]

    # Store data for computing the bifurcation diagram using postprocessing
    data = Data()
    parameters['Postprocess'] = lambda interface, x, mu: data.append(
        mu, utils.compute_volume_averaged_kinetic_energy(x, interface))

    # Perform an initial continuation to Reynolds number 7000 without detecting bifurcation points
    ds = 100
    target = 6000
    x, mu = continuation.continuation(x0, 'Reynolds Number', 0, target, ds)

    # Now detect the bifurcation point
    parameters['Destination Tolerance'] = 1e-4
    parameters['Detect Bifurcation Points'] = True
    parameters['Maximum Step Size'] = 100

    parameters['Eigenvalue Solver'] = {}
    parameters['Eigenvalue Solver']['Target'] = 3j
    parameters['Eigenvalue Solver']['Tolerance'] = 1e-9
    parameters['Eigenvalue Solver']['Number of Eigenvalues'] = 5

    bifurcation_continuation = Continuation(interface, parameters, newton_tolerance=1e-12)

    target = 10000
    x2, mu2 = bifurcation_continuation.continuation(x, 'Reynolds Number', mu, target, ds)

    ke = utils.compute_volume_averaged_kinetic_energy(x2, interface)

    # Compute the unstable branch after the bifurcation
    parameters['Detect Bifurcation Points'] = False
    parameters['Maximum Step Size'] = 2000

    target = 10000
    x3, mu3 = continuation.continuation(x2, 'Reynolds Number', mu2, target, ds)

    # Plot a bifurcation diagram. Filter out the part where we
    # have to go back an forth when converging onto a target
    data.filter()

    bif = plt.scatter(mu2, ke, marker='^')
    plt.plot(data.mu, data.value)

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
    v = plot_utils.create_state_mtx(v, nx, ny, nz, dof)
    plot_utils.plot_value(v[:, :, 0, 2], interface, title='Pressure component of the bifurcating eigenvector')


if __name__ == '__main__':
    main()
