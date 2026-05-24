"""Trajectory Planning Module."""

from importlib import import_module

_LAZY_EXPORTS = {
    'BaseTrajectoryPlanner': ('.base', 'BaseTrajectoryPlanner'),
    'CubicPolynomialPlanner': ('.joint_space.polynomial', 'CubicPolynomialPlanner'),
    'QuinticPolynomialPlanner': ('.joint_space.polynomial', 'QuinticPolynomialPlanner'),
    'SepticPolynomialPlanner': ('.joint_space.polynomial', 'SepticPolynomialPlanner'),
    'BSplinePlanner': ('.joint_space.spline', 'BSplinePlanner'),
    'MultiSegmentPlanner': ('.joint_space.multi_segment', 'MultiSegmentPlanner'),
    'LinearPositionPlanner': ('.cartesian_space.position', 'LinearPositionPlanner'),
    'SLERPPlanner': ('.cartesian_space.orientation', 'SLERPPlanner'),
    'CircularArcPlanner': ('.cartesian_space.circular', 'CircularArcPlanner'),
    'SplineCurvePlanner': ('.cartesian_space.spline', 'SplineCurvePlanner'),
    'TrapezoidalVelocityProfile': ('.velocity_profile.trapezoidal', 'TrapezoidalVelocityProfile'),
    'SCurveVelocityProfile': ('.velocity_profile.s_curve', 'SCurveVelocityProfile'),
    'draw_axis': ('.utils', 'draw_axis'),
    'plot_cartesian_trajectory': ('.utils', 'plot_cartesian_trajectory'),
    'plot_joint_trajectory': ('.utils', 'plot_joint_trajectory'),
    'plot_cartesian_with_ik': ('.utils', 'plot_cartesian_with_ik'),
}


def __getattr__(name):
    if name in _LAZY_EXPORTS:
        module_name, attr_name = _LAZY_EXPORTS[name]
        value = getattr(import_module(module_name, __name__), attr_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = list(_LAZY_EXPORTS)
