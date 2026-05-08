"""Utility modules for RoboCore.

Copyright (c) 2025 Synria Robotics Co., Ltd.

Licensed under the MIT License.

Author: Synria Robotics Team
Website: https://synriarobotics.ai
"""

from importlib import import_module

# Export commonly used functions
from .backend import set_backend, get_backend, BackendManager
from .path import (
    get_robocore_path,
    get_robocore_root,
    create_dir,
    list_absl_path,
    get_resource,
)
from .beauty_logger import beauty_print, beauty_print_array, beauty_print_matrix

_LAZY_SUBMODULES = {
    'backend',
    'path',
    'beauty_logger',
    'torch_utils',
}


def __getattr__(name):
    if name == 'select_device':
        select_device = import_module('.torch_utils', __name__).select_device
        globals()['select_device'] = select_device
        return select_device
    if name in _LAZY_SUBMODULES:
        module = import_module(f'.{name}', __name__)
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    # Modules
    'backend',
    'path',
    'beauty_logger',
    'torch_utils',
    # Backend functions
    'set_backend',
    'get_backend',
    'BackendManager',
    # Path functions
    'get_robocore_path',
    'get_robocore_root',
    'create_dir',
    'list_absl_path',
    'get_resource',
    # Logger functions
    'beauty_print',
    'beauty_print_array',
    'beauty_print_matrix',
    # Torch utils
    'select_device',
]

