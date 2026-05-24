"""Forward kinematics utility modules."""

from importlib import import_module

_LAZY_EXPORTS = {
    "FKSolverNumPy": ("robocore.kinematics.fk_utils.fk_solver_numpy", "FKSolverNumPy"),
    "FKSolverTorch": ("robocore.kinematics.fk_utils.fk_solver_torch", "FKSolverTorch"),
    "FKSolverCpp": ("robocore.kinematics.fk_utils.fk_solver_cpp", "FKSolverCpp"),
    "chain_arrays_from_robot_model": (
        "robocore.kinematics.fk_utils.fk_solver_cpp",
        "chain_arrays_from_robot_model",
    ),
    "chain_arrays_global_config": (
        "robocore.kinematics.fk_utils.fk_solver_cpp",
        "chain_arrays_global_config",
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
