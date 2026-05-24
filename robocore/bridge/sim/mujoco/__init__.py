"""MuJoCo physics simulation modules."""

from importlib import import_module

_LAZY_EXPORTS = {
    'PhysicsSimulator': ('.physics_simulator', 'PhysicsSimulator'),
    'TrajectoryExecutor': ('.trajectory_executor', 'TrajectoryExecutor'),
    'TrajectoryEvaluator': ('.trajectory_evaluator', 'TrajectoryEvaluator'),
    'ComparisonAnalyzer': ('.comparison_analyzer', 'ComparisonAnalyzer'),
    'interpolate_trajectory': ('.utils', 'interpolate_trajectory'),
    'interpolate_trajectory_with_derivatives': ('.utils', 'interpolate_trajectory_with_derivatives'),
    'compute_jerk': ('.utils', 'compute_jerk'),
    'fft_analysis': ('.utils', 'fft_analysis'),
    'detect_vibration': ('.utils', 'detect_vibration'),
    'compute_rms_error': ('.utils', 'compute_rms_error'),
    'compute_max_error': ('.utils', 'compute_max_error'),
}


def __getattr__(name):
    if name in _LAZY_EXPORTS:
        module_name, attr_name = _LAZY_EXPORTS[name]
        value = getattr(import_module(module_name, __name__), attr_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = list(_LAZY_EXPORTS)
