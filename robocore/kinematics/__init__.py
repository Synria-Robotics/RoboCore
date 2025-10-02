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

# Solver classes (not exported, for advanced usage)
from .fk_utils.fk_solver_numpy import FKSolverNumPy
from .fk_utils.fk_solver_torch import FKSolverTorch  # type: ignore
from .ik_utils.ik_solver_numpy import IKSolverNumPy
from .ik_utils.ik_solver_torch import IKSolverTorch  # type: ignore
from .jacobian_utils.jacobian_solver_numpy import JacobianSolverNumPy
from .jacobian_utils.jacobian_solver_torch import JacobianSolverTorch  # type: ignore

__all__ = [
    # unified API (main public interface)
    "forward_kinematics",
    "inverse_kinematics",
    "jacobian",
]

