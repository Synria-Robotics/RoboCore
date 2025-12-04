"""Module

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

import os
from pathlib import Path
import robocore

def get_robocore_path(relative_path = ""):
    """
    Get the installed robocore package path.
    
    :return: Absolute path to the installed robocore package directory
    """
    path = os.path.join(Path(robocore.__file__).parent.absolute(), relative_path)
    if os.path.exists(path):
        return path
    else:
        raise FileNotFoundError(f"Path does not exist: {path}")

def get_robocore_root():
    """
    Get the root directory of the RoboCore project from installed package.
    
    :return: Absolute path to the RoboCore project root directory
    """
    robocore_pkg_path = Path(get_robocore_path())
    return str(robocore_pkg_path.parent.absolute())