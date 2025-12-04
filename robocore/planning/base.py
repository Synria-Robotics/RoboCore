"""Base Trajectory Planner Interface

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

from abc import ABC, abstractmethod
from typing import Any, Optional, Dict, Tuple

from robocore.utils.backend import get_backend_manager


class BaseTrajectoryPlanner(ABC):
    """Base class for all trajectory planners.
    
    All trajectory planners inherit from this class and implement
    the plan method to generate trajectories.
    """
    
    def __init__(self):
        """Initialize base trajectory planner."""
        self._backend_manager = get_backend_manager()
    
    @abstractmethod
    def plan(
        self,
        start: Any,
        end: Any,
        duration: float,
        num_points: int,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate trajectory from start to end.
        
        :param start: Start state (joint positions, pose, etc.)
        :param end: End state (joint positions, pose, etc.)
        :param duration: Trajectory duration in seconds
        :param num_points: Number of points in trajectory
        :param kwargs: Additional parameters specific to planner
        :return: Dictionary with keys:
            - 't': Time array [num_points]
            - 'q' or 'poses': Position array [num_points, n] or [num_points, 4, 4]
            - 'qd' or 'velocities': Velocity array [num_points, n] or [num_points, 6]
            - 'qdd' or 'accelerations': Acceleration array [num_points, n] or [num_points, 6]
        """
        raise NotImplementedError("Subclasses must implement plan method")
    
    def _ensure_array(self, data: Any) -> Any:
        """Convert input to appropriate array type based on backend.
        
        :param data: Input data
        :return: Array in current backend format
        """
        return self._backend_manager.ensure_array(data)
    
    def _array(self, data: Any, dtype: Optional[Any] = None) -> Any:
        """Create array with explicit dtype.
        
        :param data: Input data
        :param dtype: Override dtype (optional)
        :return: Array in current backend format
        """
        return self._backend_manager.array(data, dtype=dtype)
    
    def _zeros(self, shape: Tuple[int, ...], dtype: Optional[Any] = None) -> Any:
        """Create zeros array.
        
        :param shape: Array shape
        :param dtype: Override dtype (optional)
        :return: Zeros array
        """
        return self._backend_manager.zeros(shape, dtype=dtype)
    
    def _linspace(self, start: float, stop: float, num: int) -> Any:
        """Create linearly spaced array.
        
        :param start: Start value
        :param stop: Stop value
        :param num: Number of points
        :return: Linearly spaced array
        """
        if self._backend_manager.is_numpy:
            import numpy as np
            return np.linspace(start, stop, num)
        else:
            import torch
            return torch.linspace(start, stop, num, device=self._backend_manager.get_device(), dtype=self._backend_manager.get_dtype())

