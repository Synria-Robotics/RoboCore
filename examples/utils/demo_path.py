"""Module

Copyright (c) 2025 Synria Robotics Co., Ltd.

Licensed under the MIT License.

Author: Synria Robotics Team
Website: https://synriarobotics.ai
"""

from robocore.utils.path import get_robocore_path
from robocore.utils.beauty_logger import beauty_print
from robocore.configs import resolve_description_path

beauty_print(get_robocore_path())
beauty_print(get_robocore_path("configs"))
beauty_print(resolve_description_path("synriard://Alicia_D/v5_6/gripper_100mm/urdf"))
