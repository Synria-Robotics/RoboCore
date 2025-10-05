"""Backend management for numpy/torch switching with GPU support.

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

import threading
from typing import Optional, Literal, Any
import numpy as np

# Thread-safe singleton
_lock = threading.Lock()
_backend_manager = None


class BackendManager:
    """
    Global backend manager for numpy/torch switching.
    
    Supports:
    - Backend selection: 'numpy' or 'torch'
    - Device selection: 'cpu', 'cuda', 'cuda:0', etc.
    - Dtype management: float32, float64
    - Automatic device placement for torch tensors
    """
    
    def __init__(self):
        self._backend: Literal['numpy', 'torch'] = 'numpy'
        self._device: str = 'cpu'
        self._dtype = np.float64
        self._torch_dtype = None
        self._torch_available = False
        
        # Try to import torch
        try:
            import torch as _torch
            self._torch = _torch
            self._torch_available = True
            self._torch_dtype = _torch.float64
        except ImportError:
            self._torch = None
    
    def set_backend(
        self, 
        backend: Literal['numpy', 'torch'] = 'numpy',
        device: str = 'cpu',
        dtype: Optional[Any] = None
    ):
        """
        Set the global backend.
        
        :param backend: 'numpy' or 'torch'
        :param device: 'cpu', 'cuda', 'cuda:0', etc. (only for torch)
        :param dtype: numpy.float32/float64 or torch.float32/float64
        """
        if backend == 'torch' and not self._torch_available:
            raise RuntimeError("Torch is not available. Install pytorch first.")
        
        self._backend = backend
        
        if backend == 'torch':
            # Validate device
            if device.startswith('cuda'):
                if not self._torch.cuda.is_available():
                    raise RuntimeError("CUDA is not available")
            self._device = device
            
            # Set dtype
            if dtype is None:
                self._torch_dtype = self._torch.float64
                self._dtype = np.float64
            else:
                if hasattr(dtype, '__module__') and 'torch' in dtype.__module__:
                    self._torch_dtype = dtype
                    # Map torch dtype to numpy
                    if dtype == self._torch.float32:
                        self._dtype = np.float32
                    else:
                        self._dtype = np.float64
                else:
                    self._dtype = dtype
                    # Map numpy dtype to torch
                    if dtype == np.float32:
                        self._torch_dtype = self._torch.float32
                    else:
                        self._torch_dtype = self._torch.float64
        else:
            self._device = 'cpu'
            if dtype is not None:
                self._dtype = dtype
            else:
                self._dtype = np.float64
    
    def get_backend(self) -> str:
        """Get current backend name."""
        return self._backend
    
    def get_device(self) -> str:
        """Get current device."""
        return self._device
    
    def get_dtype(self):
        """Get current dtype."""
        if self._backend == 'torch':
            return self._torch_dtype
        return self._dtype
    
    def ensure_array(self, data):
        """
        Convert input to appropriate array type based on current backend.
        
        :param data: Input data (list, numpy array, or torch tensor)
        :return: Array in the current backend format
        """
        if self._backend == 'numpy':
            if isinstance(data, np.ndarray):
                return data.astype(self._dtype)
            elif self._torch_available and isinstance(data, self._torch.Tensor):
                return data.cpu().numpy().astype(self._dtype)
            else:
                return np.array(data, dtype=self._dtype)
        else:  # torch
            target_dtype = self._torch_dtype
            if self._torch_available and isinstance(data, self._torch.Tensor):
                data = data.to(device=self._device, dtype=target_dtype)
                return data
            elif isinstance(data, np.ndarray):
                return self._torch.from_numpy(data).to(
                    device=self._device, dtype=target_dtype
                )
            else:
                return self._torch.tensor(
                    data, device=self._device, dtype=target_dtype
                )
    
    def array(self, data, dtype=None):
        """
        Create array with explicit dtype.
        
        :param data: Input data
        :param dtype: Override dtype (optional)
        :return: Array in current backend format
        """
        if self._backend == 'numpy':
            return np.array(data, dtype=dtype if dtype is not None else self._dtype)
        else:
            if dtype is None:
                dtype = self._torch_dtype
            return self._torch.tensor(data, device=self._device, dtype=dtype)
    
    def zeros(self, shape, dtype=None):
        """
        Create zeros array.
        
        :param shape: Array shape
        :param dtype: Override dtype (optional)
        :return: Zeros array
        """
        if self._backend == 'numpy':
            return np.zeros(shape, dtype=dtype if dtype is not None else self._dtype)
        else:
            if dtype is None:
                dtype = self._torch_dtype
            return self._torch.zeros(shape, device=self._device, dtype=dtype)
    
    def ones(self, shape, dtype=None):
        """
        Create ones array.
        
        :param shape: Array shape
        :param dtype: Override dtype (optional)
        :return: Ones array
        """
        if self._backend == 'numpy':
            return np.ones(shape, dtype=dtype if dtype is not None else self._dtype)
        else:
            if dtype is None:
                dtype = self._torch_dtype
            return self._torch.ones(shape, device=self._device, dtype=dtype)
    
    def eye(self, n, dtype=None):
        """
        Create identity matrix.
        
        :param n: Matrix size
        :param dtype: Override dtype (optional)
        :return: Identity matrix
        """
        if self._backend == 'numpy':
            return np.eye(n, dtype=dtype if dtype is not None else self._dtype)
        else:
            if dtype is None:
                dtype = self._torch_dtype
            return self._torch.eye(n, device=self._device, dtype=dtype)
    
    @property
    def module(self):
        """Get the underlying module (numpy or torch)."""
        if self._backend == 'torch':
            return self._torch
        return np
    
    @property
    def is_torch(self) -> bool:
        """Check if current backend is torch."""
        return self._backend == 'torch'
    
    @property
    def is_numpy(self) -> bool:
        """Check if current backend is numpy."""
        return self._backend == 'numpy'


def get_backend_manager() -> BackendManager:
    """
    Get the global backend manager instance (thread-safe singleton).
    
    :return: Global BackendManager instance
    """
    global _backend_manager
    if _backend_manager is None:
        with _lock:
            if _backend_manager is None:
                _backend_manager = BackendManager()
    return _backend_manager


# Convenience functions
def set_backend(
    backend: Literal['numpy', 'torch'] = 'numpy',
    device: str = 'cpu',
    dtype: Optional[Any] = None
):
    """
    Set the global backend.
    
    :param backend: 'numpy' or 'torch'
    :param device: 'cpu', 'cuda', 'cuda:0', etc.
    :param dtype: Data type for arrays
    """
    get_backend_manager().set_backend(backend, device, dtype)


def get_backend() -> str:
    """
    Get current backend name.
    
    :return: 'numpy' or 'torch'
    """
    return get_backend_manager().get_backend()


def ensure_array(data):
    """
    Convert data to current backend format.
    
    :param data: Input data
    :return: Array in current backend format
    """
    return get_backend_manager().ensure_array(data)
