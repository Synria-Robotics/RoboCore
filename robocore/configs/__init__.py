"""Configuration management for RoboCore."""

from importlib import import_module

from .description_paths import resolve_description_path

_LAZY_EXPORTS = {
    'ConfigManager': ('robocore.configs.config_manager', 'ConfigManager'),
    'load_config': ('robocore.configs.config_manager', 'load_config'),
    'get_default_config': ('robocore.configs.config_manager', 'get_default_config'),
    'RobotConfig': ('robocore.configs.schemas', 'RobotConfig'),
    'KinematicsConfig': ('robocore.configs.schemas', 'KinematicsConfig'),
    'IKConfig': ('robocore.configs.schemas', 'IKConfig'),
    'SolverConfig': ('robocore.configs.schemas', 'SolverConfig'),
    'ComputeConfig': ('robocore.configs.schemas', 'ComputeConfig'),
}


def __getattr__(name):
    if name in _LAZY_EXPORTS:
        module_name, attr_name = _LAZY_EXPORTS[name]
        value = getattr(import_module(module_name), attr_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    'ConfigManager',
    'load_config',
    'get_default_config',
    'resolve_description_path',
    'RobotConfig',
    'KinematicsConfig',
    'IKConfig',
    'SolverConfig',
    'ComputeConfig'
]
