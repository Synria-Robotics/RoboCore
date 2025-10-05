"""MuJoCo bridge (stub minimal path).

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

from __future__ import annotations

from pathlib import Path
from typing import Dict, Sequence

from robocore.bridge.bridge import Bridge
from robocore.modeling.robot_model import RobotModel


class MuJoCoBridge(Bridge):
    """MuJoCo simulation bridge (placeholder).

    :param urdf_path: path to URDF file.
    """

    def __init__(self, urdf_path: str | Path):
        self.model = RobotModel(urdf_path)
        self._q = [0.0] * self.model.num_dof()

    def reset(self):
        """Reset state."""
        self._q = [0.0] * self.model.num_dof()

    def step(self, action: Sequence[float]):
        """Apply target joint values (instant set).

        :param action: new joint positions length dof.
        """
        if len(action) != self.model.num_dof():
            raise ValueError("Action length mismatch")
        self._q = list(action)

    def get_observation(self) -> Dict[str, object]:
        """Return current state and end-effector pose.

        :return: observation dict.
        """
        fk = self.model.forward_kinematics(self._q)
        return {"q": self._q, "end_pose": fk["end"]}


__all__ = ["MuJoCoBridge"]
