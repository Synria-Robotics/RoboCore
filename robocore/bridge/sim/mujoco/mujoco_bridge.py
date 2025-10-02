"""MuJoCo bridge (stub minimal path).

This provides a very small façade; real physics integration can be added later.
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
        self._q = [0.0] * self.model.dof()

    def reset(self):
        """Reset state."""
        self._q = [0.0] * self.model.dof()

    def step(self, action: Sequence[float]):
        """Apply target joint values (instant set).

        :param action: new joint positions length dof.
        """
        if len(action) != self.model.dof():
            raise ValueError("Action length mismatch")
        self._q = list(action)

    def get_observation(self) -> Dict[str, object]:
        """Return current state and end-effector pose.

        :return: observation dict.
        """
        fk = self.model.forward_kinematics(self._q)
        return {"q": self._q, "end_pose": fk["end"]}


__all__ = ["MuJoCoBridge"]
