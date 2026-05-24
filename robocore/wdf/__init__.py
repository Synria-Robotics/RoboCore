"""Workspace distance field helpers."""

from importlib import import_module

_LAZY_EXPORTS = {
    'plot_2D_sdf': ('robocore.wdf.vis', 'plot_2D_sdf'),
    'plot_3D_sdf_with_gradient': ('robocore.wdf.vis', 'plot_3D_sdf_with_gradient'),
    'plot_3D_sdf_with_gradient_surface_points': (
        'robocore.wdf.vis',
        'plot_3D_sdf_with_gradient_surface_points',
    ),
    'plot_3D_sdf_envelope': ('robocore.wdf.vis', 'plot_3D_sdf_envelope'),
}


def __getattr__(name):
    if name in _LAZY_EXPORTS:
        module_name, attr_name = _LAZY_EXPORTS[name]
        value = getattr(import_module(module_name), attr_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    'plot_2D_sdf',
    'plot_3D_sdf_with_gradient',
    'plot_3D_sdf_with_gradient_surface_points',
    'plot_3D_sdf_envelope',
]
