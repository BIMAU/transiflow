# How to contribute

Thanks for considering contributing to TransiFlow! Any contribution is welcome, but let me list some examples:

- Add a new problem type. In this case you'd need to add a discretization for the problem, an example in the examples directory and a problem description to the documentation. This should be done in a single pull request.
- Add a new computational backend. An example can be a PETSc backend that provides a specialized preconditioner. Since this can be a large undertaking, this can (and probably should) be split into many smaller pull request.
- Add a new feature. For instance a new way to track bifurcations or a new time stepper.
- Fix a bug. Hopefully you don't encounter any bugs, but in case you do, feel free to fix them or report an issue.
- Fix the documentation. If you're in the process of reading the documentation, you're the one most likely to find any issues, that no one would notice otherwise. Fixing these issues is extremely useful!

## Coding conventions

In general, just follow the style of the code around the code you're editing. Other than that:

- Make sure every commit makes sense as a stand-alone commit, and that the code still works after every commit. This makes the code easy to review and makes it easier to find bugs in the future.
- Make sure every commit has a readable commit message that follows the style of the existing commits.