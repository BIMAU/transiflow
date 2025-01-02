Performing a continuation
=========================
In this section, we will briefly guide you through the :ref:`Double-gyre wind-driven circulation` example as a means of explaining what a typical workflow looks like.
The full example can be found in ``examples/qg.py``.

First we create an :func:`Interface <.create>`.
This is the interface to the computational backend and to the discretization of the problem.
The problem is defined by the ``parameters`` dictionary, the problem size by ``nx`` and ``ny``, and optionally ``nz`` for 3D problems.
By specifying the ``backend`` parameter, we could use a :ref:`different computational backend <choosing a backend>` (the default is SciPy) and everything would work exactly the same.

.. code-block:: Python

    parameters = {'Problem Type': 'Double Gyre',
                  'Reynolds Number': 16,
                  'Rossby Parameter': 1000,
                  'Wind Stress Parameter': 0}

    interface = Interface(parameters, nx, ny)

We also create an instance of the :class:`.Continuation` class, which contains all the parameters that are used to control the continuation process.
It is not aware of the problem we are solving, and it does not have a state, so we are free to reuse, or create new instances whenever we like

.. code-block:: Python

    continuation = Continuation(interface)

We now perform the initial continuation in the wind stress parameter.
We use an arc-length step size of 100, and increase the wind stress parameter to 1000 by means of the continuation process.

.. code-block:: Python

    x0 = interface.vector()

    ds = 100
    target = 1000
    x1 = continuation.continuation(x0, 'Wind Stress Parameter', 0, target, ds)[0]

We can now plot the solutions at the current parameter values.

.. code-block:: Python

    plot_utils.plot_streamfunction(x1, interface, title='Streamfunction at Re=16')
    plot_utils.plot_vorticity(x1, interface, title='Vorticity at Re=16')

Now that we have our initial solution ``x1`` at Reynolds number 16, we can start increasing the Reynolds number.
In the end, we want to plot a bifurcation diagram, so we want to keep track of the maximum of the stream function as function of the Reynolds number.
For this purpose we create some ``data`` dictionaries containing these values.

.. code-block:: Python

    data2 = {'Reynolds Number': [], 'Stream Function Maximum': []}
    callback = lambda interface, x, mu: postprocess(data2, interface, x, mu)

    ds = 5
    target = 40
    x2, mu2 = continuation.continuation(x1, 'Reynolds Number', 16, target, ds,
                                        callback=callback)

We now have the central branch of the supercritical pitchfork bifurcation, both before and after the bifurcation point.
We will now perform a series of continuations to get on the stable branch after the bifurcation by first adding asymmetry to the problem, and then, once we are close to the branch, remove the asymmetry once again.

.. code-block:: Python

    # Add asymmetry to the problem
    ds = 10
    target = 1
    interface.set_parameter('Reynolds Number', 16)
    x3, mu3 = continuation.continuation(x1, 'Asymmetry Parameter', 0, target, ds, maxit=1)

    ds = 5
    target = 40
    x4, mu4 = continuation.continuation(x3, 'Reynolds Number', 16, target, ds)

    # Go back to the symmetric problem
    ds = -1
    target = 0
    x5, mu5 = continuation.continuation(x4, 'Asymmetry Parameter', mu3, target, ds)

We are now on the stable branch at Reynolds number 40.
We will now go backwards and around the bifurcation point to compute both stable branches of the pitchfork.
We again store data so we are able to plot the bifurcation diagram.

.. code-block:: Python

    data6 = {'Reynolds Number': [], 'Stream Function Maximum': []}
    callback = lambda interface, x, mu: postprocess(data6, interface, x, mu)

    ds = -5
    target = 40
    x6, mu6 = continuation.continuation(x5, 'Reynolds Number', mu4, target, ds,
                                        callback=callback)

Now we can finally plot the bifurcation diagram.

.. code-block:: Python

    plt.title('Bifurcation diagram for the QG model with $n_x=n_y={}$'.format(nx))
    plt.xlabel('Reynolds number')
    plt.ylabel('Maximum value of the streamfunction')
    plt.plot(data2['Reynolds Number'], data2['Stream Function Maximum'])
    plt.plot(data6['Reynolds Number'], data6['Stream Function Maximum'])
    plt.show()
