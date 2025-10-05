---
title: 'TransiFlow: A Python package for performing bifurcation analysis on fluid flow problems'
tags:
  - Python
  - bifurcation analysis
  - fluid dynamics
  - parallel computing
  - finite volume method
authors:
  - name: Sven Baars
    orcid: 0000-0002-4228-6479
    affiliation: 1
  - name: David Nolte
    orcid: 0000-0002-3535-8075
    affiliation: 2
  - name: Fred W. Wubs
    orcid: 0000-0002-1263-1067
    affiliation: 2
  - name: Henk A. Dijkstra
    orcid: 0000-0001-5817-7675
    affiliation: 1
affiliations:
  - name: Institute for Marine and Atmospheric Research (IMAU), Utrecht University, The Netherlands
    index: 1
  - name: Bernoulli Institute for Mathematics, Computer Science and Artificial Intelligence, University of Groningen, The Netherlands
    index: 2
date: 7 May 2024
bibliography: paper.bib
---

# Summary

Dynamical systems derived from models of fluid flow show transition behaviors associated with fluid flow instabilities [@wubs:23].
For example, dynamical systems from ocean models in which transition behavior is caused by slow changes in parameters representing the surface forcing [@westen:24].
A flow transition is a qualitative change in the flow when a specific parameter is varied, e.g. a transition from a no-flow heat-conducting fluid to a heat-transporting flow (as in Rayleigh-Bénard convection), or a steady flow which has a steady forcing and steady boundary conditions that turns into a transient flow which may even produce sound (as in the von Kármán vortex street).
In both these examples the qualitatively different solutions do not appear out of the blue.

If one perturbs a steady solution before the transition point then the perturbation will die out and recover to the steady solution.
This means that the steady solution is stable.
However, this perturbed flow already reveals the shape that will occur after the transition point.
For forcing beyond this point that shape will grow into a new steady flow (the stable solution becomes unstable) for Rayleigh-Bénard convection and into a transient flow for the von Kármán vortex street (the stationary flow becomes unstable).
The parameter value for which the transition occurs is called a bifurcation point, sometimes also referred to as a transition or tipping point.

Studying these phenomena (bifurcation analysis) can be done by performing numerical simulations and observing transient behavior after a certain time.
This is, however, computationally expensive and in many cases infeasible.
Instead, so-called continuation methods are able to trace stable and unstable steady states in parameter space, obviating expensive transient simulations [@dijkstra:05].
The `TransiFlow` Python package implements a continuation framework in which fluid flow problems can be studied with the help of several computational back-ends that can, based on the needs of the user, be easily switched between.

One motivation behind `TransiFlow` is that writing research software that works efficiently on a parallel computer is a challenging task.
Numerical models are often developed as a sequential code with parallelization as an afterthought, making them difficult to parallelize, or as a parallel code from the start, making them complicated to work with.
This is especially problematic since people that work with these codes generally only use with them for the duration of a project.
If there is insufficient documentation and continuity between the projects then knowledge of how to use or develop the codes may get lost, rendering the software useless.

In climate modelling, this is a prominent issue, since the models are complex, are usually intercoupled with other models (e.g., ocean, atmosphere, ice), take a very long time to run (i.e., multiple months) and require large amounts of parallelism to reach a sufficient resolution (i.e., using thousands of cores for a single run) [@thies:09; @mulder:21].
Therefore, ease of developing and using the parallel software is crucial.

By abstracting away the computational back-end the user can adjust a model to their own needs on their own machine (e.g., a laptop) in Python using the SciPy back-end, and once the model works, run a large scale simulation on a supercomputer using, e.g., the `Trilinos` back-end, which can use a combination of OpenMP, MPI, and potentially GPUs, without requiring any changes to the code.
The computationally expensive parts of the program are implemented by these libraries so one does not have to worry about the efficiency of the Python implementation of the model.
Initial tests indicate that the overhead of using Python is less than 1% of the total computational cost.

![Bifurcation diagram of the double-gyre wind-driven circulation configuration that is included in `TransiFlow`.
The markers indicate pitchfork, Hopf and saddle-node bifurcations that were automatically detected by the software.
Solid lines indicate stable steady states of the system; dashed lines indicate unstable steady states.
A more extensive description of the bifurcation diagram and steps to reproduce it can be found in [@sapsis:13].
](qg-bif.pdf){height=250pt}

# Statement of need

`TransiFlow` aims to be an easy-to-use tool for performing bifurcation analysis on fluid flow problems that can be used in combination with fast parallel solvers with minimal additional effort.
`TransiFlow` implements pseudo-arclength continuation and implicit time integration methods, as well as finite-volume discretizations for the incompressible Navier-Stokes equations with optional heat and salinity transport.
We also provide implementations of various canonical fluid flow problems such as lid-driven and differentially heated cavities, Rayleigh-Bénard convection and Taylor-Couette flow, a feature none of its competitors provide.

The main competitors are [`AUTO`](http://indy.cs.concordia.ca/auto/) [@doedel:07], [`MatCont`](https://sourceforge.net/projects/matcont/) [@dhooge:08] [BifurcationKit.jl](https://bifurcationkit.github.io/BifurcationKitDocs.jl/stable/) [@veltz:20] and [`pde2path`](https://www.staff.uni-oldenburg.de/hannes.uecker/pde2path/) [@uecker:14].
These packages are widely used, and they are much more feature complete in terms of bifurcation analysis.
However, they lack the discretized fluid flow models and parallel solver interfaces that make `TransiFlow` easy to use.

A continuation package that allows for easy coupling with parallel solvers is the [`LOCA`](https://trilinos.github.io/nox_and_loca.html) package in `Trilinos` [@trilinos].
Using `LOCA`, however, requires a vast knowledge of C++ and any change to the continuation algorithm requires a full stack of C++ templates to be reimplemented.
Moreover, `LOCA` has not seen any active development in over 10 years.

An alternative Python package for performing bifurcation analysis is [`PyNCT`](https://pypi.org/project/PyNCT/) [@draelants:15].
It is, however, difficult to extend, the latest version of the software is not freely available, and it can only be used for systems that are symmetric positive-definite.

# Past and ongoing research

`TransiFlow` has been used to generate results in [@wubs:23]  and [@bernuzzi:24] and has been used in courses at Utrecht University and the University of Groningen (NL).
Earlier versions have also been used to generate results in [@song:19] and [@baars:20].
The code is currently being used in various projects at Utrecht University, such as a project for coupling climate models and a project for studying transitions in the Atlantic Meridional Overturning Circulation.
There is also ongoing work on adding 3D ocean and atmosphere discretizations to `TransiFlow`.
`TransiFlow` is also used by several external researchers.

# Acknowledgements

H.A.D. and S.B. are funded by the European Research Council through the ERC-AdG project TAOC (project 101055096).
S.B. and F.W.W. were supported by funding from the SMCM project of the Netherlands eScience Center (NLeSC) with project number 027.017.G02.
We would like to thank Lourens Veen for his contributions in providing guidance in setting up the documentation and continuous deployment.

# References
