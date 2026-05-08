"""Unified kinematics public API.

Copyright (c) 2025 Synria Robotics Co., Ltd.

Licensed under the MIT License.

Author: Synria Robotics Team
Website: https://synriarobotics.ai
"""

from importlib import import_module

from .fk import forward_kinematics  # unified FK
from .ik import inverse_kinematics  # unified IK
from .jacobian import jacobian  # unified Jacobian
from .bimanual import bimanual_forward_kinematics, bimanual_inverse_kinematics, bimanual_jacobian

_LAZY_EXPORTS = {
    'FKSolverNumPy': ('.fk_utils.fk_solver_numpy', 'FKSolverNumPy'),
    'FKSolverTorch': ('.fk_utils.fk_solver_torch', 'FKSolverTorch'),
    'IKSolverNumPy': ('.ik_utils.ik_solver_numpy', 'IKSolverNumPy'),
    'IKSolverTorch': ('.ik_utils.ik_solver_torch', 'IKSolverTorch'),
    'JacobianSolverNumPy': ('.jacobian_utils.jacobian_solver_numpy', 'JacobianSolverNumPy'),
    'JacobianSolverTorch': ('.jacobian_utils.jacobian_solver_torch', 'JacobianSolverTorch'),
}


def __getattr__(name):
    if name in _LAZY_EXPORTS:
        module_name, attr_name = _LAZY_EXPORTS[name]
        value = getattr(import_module(module_name, __name__), attr_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    # unified API (main public interface)
    "forward_kinematics",
    "inverse_kinematics",
    "jacobian",
    # bimanual
    "bimanual_forward_kinematics",
    "bimanual_inverse_kinematics",
    "bimanual_jacobian",
    # advanced solvers
    "FKSolverNumPy",
    "FKSolverTorch",
    "IKSolverNumPy",
    "IKSolverTorch",
    "JacobianSolverNumPy",
    "JacobianSolverTorch",
]

