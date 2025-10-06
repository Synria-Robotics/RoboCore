"""Unified kinematics public API.

Copyright (c) 2025 Synria Robotics Co., Ltd.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see <https://www.gnu.org/licenses/>.

Author: Synria Robotics Team
Website: https://synriarobotics.ai
"""

from .fk import forward_kinematics  # unified FK
from .ik import inverse_kinematics  # unified IK
from .jacobian import jacobian  # unified Jacobian
from .bimanual import bimanual_forward_kinematics, bimanual_inverse_kinematics, bimanual_jacobian

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
    # bimanual
    "bimanual_forward_kinematics",
    "bimanual_inverse_kinematics",
    "bimanual_jacobian",
]

