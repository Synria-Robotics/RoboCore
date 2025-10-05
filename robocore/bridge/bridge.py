"""Bridge base classes.

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
