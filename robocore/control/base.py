"""Base Controller Interface

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
from typing import Any, Optional

import numpy as np

from robocore.utils.backend import get_backend_manager


class BaseController(ABC):
    """Base class for all controllers.
    
    All controllers inherit from this class and implement the compute method.
    Controllers compute control torques based on current and desired states.
    """
    
    def __init__(self):
        """Initialize base controller."""
        self._backend_manager = get_backend_manager()
        self._reset_state()
    
    def _reset_state(self):
        """Reset controller internal state."""
        pass
    
    @abstractmethod
    def compute(self, *args, **kwargs) -> Any:
        """Compute control output (torque).
        
        :return: Control torque array [n×1]
        """
        raise NotImplementedError("Subclasses must implement compute method")
    
    def reset(self):
        """Reset controller state.
        
        Resets any internal state variables (e.g., integral terms in PID).
        """
        self._reset_state()
    
    def update_params(self, **params):
        """Update controller parameters.
        
        :param params: Parameter name-value pairs to update
        """
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Unknown parameter: {key}")
    
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
    
    def _zeros(self, shape: tuple, dtype: Optional[Any] = None) -> Any:
        """Create zeros array.
        
        :param shape: Array shape
        :param dtype: Override dtype (optional)
        :return: Zeros array
        """
        return self._backend_manager.zeros(shape, dtype=dtype)
    
    def _ones(self, shape: tuple, dtype: Optional[Any] = None) -> Any:
        """Create ones array.
        
        :param shape: Array shape
        :param dtype: Override dtype (optional)
        :return: Ones array
        """
        return self._backend_manager.ones(shape, dtype=dtype)

