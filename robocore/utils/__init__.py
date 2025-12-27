"""Utility modules for RoboCore.

Copyright (c) 2025 Synria Robotics Co., Ltd.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see <https://www.gnu.org/licenses/>.

Author: Synria Robotics Team
Website: https://synriarobotics.ai
"""

from . import backend
from . import path
from . import beauty_logger
from . import torch_utils

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
from .torch_utils import select_device

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

