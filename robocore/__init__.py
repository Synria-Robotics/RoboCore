"""High-Performance Robotics Kinematics Library

Copyright (c) 2025 Synria Robotics Co., Ltd.

Licensed under the MIT License.

Author: Synria Robotics Team
Website: https://synriarobotics.ai
"""

__version__ = "2.5.0rc3"
__author__ = "Synria Robotics Team"
__copyright__ = "Copyright (c) 2025 Synria Robotics Co., Ltd."
__license__ = "MIT"

from importlib import import_module

# Export backend management functions
from .utils.backend import set_backend, get_backend

_LAZY_SUBMODULES = {
    'modeling',
    'kinematics',
    'dynamics',
    'transform',
    'planning',
    'analysis',
    'configs',
    'utils',
    'control',
}


def __getattr__(name):
    if name in _LAZY_SUBMODULES:
        module = import_module(f'.{name}', __name__)
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    'set_backend',
    'get_backend',
    'modeling',
    'kinematics',
    'dynamics',
    'transform',
    'planning',
    'analysis',
    'configs',
    'utils',
    'control',
]

