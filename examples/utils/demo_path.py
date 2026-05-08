"""Module

Copyright (c) 2025 Synria Robotics Co., Ltd.

Licensed under the MIT License.

Author: Synria Robotics Team
Website: https://synriarobotics.ai
"""

from robocore.utils.path import get_robocore_path
from robocore.utils.beauty_logger import beauty_print

beauty_print(get_robocore_path())
beauty_print(get_robocore_path("assets/robot_descriptions/urdf"))
beauty_print(get_robocore_path("assets/robot_descriptions/urdf/122"))  # should raise FileNotFoundError
