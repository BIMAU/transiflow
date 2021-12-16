# FVM

FVM is a Python package that implements several standard test cases in computational fluid dynamics using the finite volume method.
We provide an interface to compute a right-hand side, Jacobian matrix, and mass matrix for these problems, which allows us to perform time integration, or a continuation to compute a bifurcation diagram for the problem at hand.

## Continuation

We provide a pseudo-arclength continuation method with adaptive arclength step size.
Given a continuation parameters and a target value, the continuation can be called as follows

```Python
    # Define the problem
    parameters = {'Reynolds Number': 0, 'Problem Type': 'Lid-driven cavity'}
    interface = Interface(parameters, nx, ny, nz, dim, dof)

    # Instantiate the continuation
    continuation = Continuation(interface, parameters)

    # Compute an initial guess
    x0 = numpy.zeros(dof * nx * ny * nz)
    x0 = continuation.newton(x0)

    # Perform the continuation. x will be the state at the target Reynolds number.
    x, mu = continuation.continuation(x0, 'Reynolds Number', start, target, ds)
```

## Eigenvalue computation

For the computation of eigenvalues, which can be used for the detection of bifurcation points, we provide an interface to [JaDaPy](https://github.com/BIMAU/jadapy).
JaDaPy has to be installed or included in the `PYTHONPATH` to use it.
An example of how to perform a continuation and compute eigenvalues can be found in `examples/ldc.py`.

## Installation

FVM is best installed in a [virtual environment](https://docs.python.org/3/library/venv.html).
We state the most common steps for creating and using a virtual environment here.
Refer to the documentation for more details.

To create a virtual environment run
```
python3 -m venv /path/to/new/virtual/environment
```

and to activate the virtual environment, run
```
source /path/to/new/virtual/environment/bin/activate
```

After this, we can install fvm from the fvm source directory.
```
pip install .
```

This will also install all of the requirements.
Now one should be able to run an example.
```
python examples/ldc.py
```

If one does not want to install fvm, but instead just wants to run it from the source directory, one can install the requirements by running
```
pip install -r requirements.txt
```

And one can run the example with
```
PYTHONPATH=. python examples/ldc.py
```

If the example fails with
```
ldc.py:64: UserWarning: Matplotlib is currently using agg, which is a non-GUI backend, so cannot show the figure.
  plt.show()
```
this means that [tkinter](https://docs.python.org/3/library/tkinter.html) is not available.
You can either just save the image to the disk, or install e.g. `python3-tk` or `python3-matplotlib` on Debian-based Linux distributions.
