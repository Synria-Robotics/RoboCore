"""Jacobian utility modules."""

from importlib import import_module

_LAZY_EXPORTS = {
    "JacobianSolverNumPy": (
        "robocore.kinematics.jacobian_utils.jacobian_solver_numpy",
        "JacobianSolverNumPy",
    ),
    "JacobianSolverTorch": (
        "robocore.kinematics.jacobian_utils.jacobian_solver_torch",
        "JacobianSolverTorch",
    ),
    "JacobianSolverCpp": (
        "robocore.kinematics.jacobian_utils.jacobian_solver_cpp",
        "JacobianSolverCpp",
    ),
    "stop_chain_index": (
        "robocore.kinematics.jacobian_utils.jacobian_solver_cpp",
        "stop_chain_index",
    ),
}


def __getattr__(name):
    if name in _LAZY_EXPORTS:
        module_name, attr_name = _LAZY_EXPORTS[name]
        value = getattr(import_module(module_name), attr_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = list(_LAZY_EXPORTS)
