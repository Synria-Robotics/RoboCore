"""Inverse kinematics utility modules."""

from importlib import import_module

_LAZY_EXPORTS = {
    "IKSolverNumPy": ("robocore.kinematics.ik_utils.ik_solver_numpy", "IKSolverNumPy"),
    "IKSolverTorch": ("robocore.kinematics.ik_utils.ik_solver_torch", "IKSolverTorch"),
    "IKSolverCpp": ("robocore.kinematics.ik_utils.ik_solver_cpp", "IKSolverCpp"),
}


def __getattr__(name):
    if name in _LAZY_EXPORTS:
        module_name, attr_name = _LAZY_EXPORTS[name]
        value = getattr(import_module(module_name), attr_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = list(_LAZY_EXPORTS)
