"""Unified kinematics public API.

Exposes high-level convenience functions:
  - forward_kinematics (auto backend)
  - inverse_kinematics (auto backend)
  - jacobian (auto backend)

Lower-level solver classes (NumPy / Torch) are kept for advanced usage.
"""

from .fk import forward_kinematics  # unified FK
from .ik import inverse_kinematics  # unified IK
from .jacobian import jacobian  # unified Jacobian
from .ik_solver_numpy import IKSolverNumPy  # retained for advanced usage (not exported)
from .ik_solver_torch import IKSolverTorch  # type: ignore  # retained (not exported)

from .jacobian_analytic_numpy import analytic_jacobian_numpy  # legacy analytic (kept for direct np use)
from .jacobian_numpy import numeric_jacobian_numpy, _orientation_error_numpy  # numeric np
from .jacobian_analytic_torch import analytic_jacobian_torch  # torch
from .jacobian_numeric_torch import numeric_jacobian_torch
from .jacobian_autograd_torch import autograd_geometric_jacobian_torch

__all__ = [
    # unified API
    "forward_kinematics",
    "inverse_kinematics",
    "jacobian",
    # (solvers intentionally not exported to keep surface minimal)
    # numpy utilities
    "analytic_jacobian_numpy",
    "numeric_jacobian_numpy",
    "_orientation_error_numpy",
    # torch
    "analytic_jacobian_torch",
    "numeric_jacobian_torch",
    "autograd_geometric_jacobian_torch",
]

