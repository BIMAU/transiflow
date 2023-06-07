from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np


# TODO: nothing is being validated yet.
@dataclass
class Parameters:
    """This would be a great place to document the meaning of these parameters

    args:
        residual_check: Residual Check
        verbose: Verbose
        maximum_newton_iterations: Maximum Newton Iterations
        newton_tolerance: Newton Tolerance
        bordered_solver: Bordered Solver
        minimum_step_size: Minimum Step Size
        maximum_step_size: Maximum Step Size
        optimal_newton_iterations: Optimal Newton Iterations
        destination_tolerance: Destination Tolerance
        branch_switching_method: Branch Switching Method
        delta: Delta
        maximum_continuation_steps: Maximum Continuation Steps
        detect_bifurcation_points: Detect Bifurcation Points
        enable_branch_switching: Enable Branch Switching
        r_min: R-min
        r_max: R-max
        theta_min: Theta-min
        theta_max: Theta-max
        x_min: X-min
        x_max: X-max
        y_min: Y-min
        y_max: Y-max
        z_min: Z-min
        z_max: Z-max
        z_periodic: Z-periodic
        grid_stretching: Grid Stretching
        grid_stretching_method: Grid Stretching Method
        grid_stretching_factor: Grid Stretching Factor
        theta: Theta
        eigenvalue_solver: Eigenvalue Solver
        arithmetic: Arithmetic
        target: Target
        initial_subspace_dimension: Initial Subspace Dimension
        minimum_subspace_dimension: Minimum Subspace Dimension
        maximum_subspace_dimension: Maximum Subspace Dimension
        recycle_subspaces: Recycle Subspaces
        tolerance: Tolerance
        number_of_eigenvalues: Number of Eigenvalues
        preconditioner: Preconditioner
        drop_tolerance: Drop Tolerance
        fill_factor: Fill Factor
        drop_rule: Drop Rule
        use_iterative_solver: Use Iterative Solver
        iterative_solver: Iterative Solver
        restart: Restart
        maximum_iterations: Maximum Iterations
        convergence_tolerance: Convergence Tolerance
    """

    residual_check: Literal["F", "T"] = "F"
    verbose: bool = False
    maximum_newton_iterations: int = 10
    newton_tolerance: float = 1e-10
    bordered_solver: bool = False
    minimum_step_size: float = 0.01
    maximum_step_size: float = 2000.0
    optimal_newton_iterations: int = 3
    destination_tolerance: float = 1e-4
    branch_switching_method: str = "Tangent"
    delta: float = 1.0
    maximum_continuation_steps: int = 1000
    detect_bifurcation_points: bool = False
    enable_branch_switching: bool = False
    r_min: float = 1.0
    r_max: float = 2.0
    theta_min: float = 0.0
    theta_max: float = 2 * np.pi
    x_min: float = 0.0
    x_max: float = 1.0
    y_min: float = 0.0
    y_max: float = 1.0
    z_min: float = 0.0
    z_max: float = 1.0
    z_periodic: bool = False
    grid_stretching: bool = False
    grid_stretching_method: str = "tanh"
    grid_stretching_factor: float = 0.1
    theta: float = 1.0
    eigenvalue_solver: dict = field(default_factory=dict)
    arithmetic: str = "complex"
    target: float = 0.0
    initial_subspace_dimension: int = 0
    minimum_subspace_dimension: int = 30
    maximum_subspace_dimension: int = 60
    recycle_subspaces: bool = True
    tolerance: float = 1e-7
    number_of_eigenvalues: int = 5
    preconditioner: dict = field(default_factory=dict)
    drop_tolerance: Any = None
    fill_factor: Any = None
    drop_rule: Any = None
    use_iterative_solver: bool = False
    iterative_solver: dict = field(default_factory=dict)
    restart: int = 100
    maximum_iterations: int = 1000
    convergence_tolerance: float = 1e-6
