# TransiFlow

TransiFlow is a Python package that implements several standard test cases in computational fluid dynamics using the finite volume method.
We provide an interface to compute a right-hand side, Jacobian matrix, and mass matrix for these problems, which allows us to perform time integration, or a continuation to compute a bifurcation diagram for the problem at hand.

## Continuation

We provide a pseudo-arclength continuation method with adaptive arclength step size.
Given a continuation parameters and a target value, the continuation can be called as follows

```Python
    # Define the problem
    parameters = {'Reynolds Number': 0, 'Problem Type': 'Lid-driven cavity'}
    interface = Interface(parameters, nx, ny, nz)

    # Instantiate the continuation
    continuation = Continuation(interface)

    # Compute an initial guess
    x0 = interface.vector()
    x0 = continuation.newton(x0)

    # Perform the continuation. x will be the state at the target Reynolds number.
    x, mu = continuation.continuation(x0, 'Reynolds Number', start, target, ds)
```

## Eigenvalue computation

For the computation of eigenvalues, which can be used for the detection of bifurcation points, we provide an interface to [JaDaPy](https://github.com/BIMAU/jadapy).
JaDaPy has to be installed or included in the `PYTHONPATH` to use it.
An example of how to perform a continuation and compute eigenvalues can be found in `examples/ldc.py`.

## Installation

TransiFlow is best installed in a [virtual environment](https://docs.python.org/3/library/venv.html).
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

After this, we can upgrade pip and install TransiFlow in editable mode from the transiflow source directory.
```
pip install --upgrade pip
pip install -e .
```
This will also install all of the dependencies.
The same can be done for JaDaPy in the same virtual environment.

Now one should be able to run an example.
```
python examples/ldc.py
```

If the example fails with
```
ldc.py:64: UserWarning: Matplotlib is currently using agg, which is a non-GUI backend, so cannot show the figure.
  plt.show()
```
this means that [tkinter](https://docs.python.org/3/library/tkinter.html) is not available.
You can either just save the image to the disk, or install e.g. `python3-tk` or `python3-matplotlib` on Debian-based Linux distributions.
