"""Bridge base classes.

Copyright (c) 2025 Synria Robotics Co., Ltd.

Licensed under the MIT License.

Author: Synria Robotics Team
Website: https://synriarobotics.ai
"""

from __future__ import annotations

from typing import Any, Sequence


class Bridge:
    """Abstract bridge.

    :param model: robot model instance.
    """

    def __init__(self, model: Any):
        self.model = model

    def reset(self):  # pragma: no cover - simple stub
        """Reset system."""

    def step(self, action: Sequence[float]):  # pragma: no cover - stub
        """Apply action.

        :param action: joint targets.
        """

    def get_observation(self):  # pragma: no cover - stub
        """Get observation.

        :return: observation dict.
        """
        return {}


__all__ = ["Bridge"]
