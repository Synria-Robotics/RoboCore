"""Backend management for numpy/torch switching with GPU support.

Copyright (c) 2025 Synria Robotics Co., Ltd.

Licensed under the MIT License.

Author: Synria Robotics Team
Website: https://synriarobotics.ai
"""

import threading
from typing import Optional, Literal, Any
import numpy as np

from robocore.utils.beauty_logger import beauty_print

# Thread-safe singleton
_lock = threading.Lock()
_backend_manager = None


class BackendManager:
    """
    Global backend manager for numpy/torch/cpp switching.

    Supports:
    - Backend selection: 'numpy', 'torch', or 'cpp' (native FK/IK/Jacobian extensions; array API like NumPy)
    - Device selection: 'cpu', 'cuda', 'cuda:0', etc. (torch only)
    - Dtype management: float32, float64
    - Automatic device placement for torch tensors
    """

    def __init__(self):
        self._backend: Literal['numpy', 'torch', 'cpp'] = 'numpy'
        self._device: str = 'cpu'
        self._dtype = np.float64
        self._torch_available = False

        # Try to import torch
        try:
            import torch as _torch
            self._torch = _torch
            self._torch_available = True
        except ImportError:
            self._torch = None

    def set_backend(
        self,
        backend: Literal['numpy', 'torch', 'cpp'] = 'numpy',
        device: Any = None,
        dtype: Optional[Any] = None
    ):
        """
        Set the global backend.

        :param backend: 'numpy', 'torch', or 'cpp'
        :param device: For torch, None picks CUDA if available else CPU; or 'cpu', 'cuda', 'cuda:0', torch.device. Ignored for numpy/cpp (CPU).
        :param dtype: numpy.float32/float64 or torch.float32/float64
        """
        if backend not in ('numpy', 'torch', 'cpp'):
            raise ValueError(f"backend must be 'numpy', 'torch', or 'cpp', got {backend!r}")
        if backend == 'torch' and not self._torch_available:
            raise RuntimeError("Torch is not available. Install pytorch first.")

        self._backend = backend

        if backend == 'torch':
            # Convert torch.device object to string if needed
            if self._torch_available and isinstance(device, self._torch.device):
                device = str(device)

            if device is None:
                device = (
                    'cuda'
                    if self._torch.cuda.is_available()
                    else 'cpu'
                )

            # Validate device
            if device.startswith('cuda'):
                if not self._torch.cuda.is_available():
                    raise RuntimeError("CUDA is not available")
            self._device = device

            # Set dtype - convert to torch dtype if needed
            if dtype is None:
                self._dtype = self._torch.float64
            else:
                if hasattr(dtype, '__module__') and 'torch' in dtype.__module__:
                    # Already a torch dtype
                    self._dtype = dtype
                else:
                    # Convert numpy dtype to torch
                    if dtype == np.float32:
                        self._dtype = self._torch.float32
                    else:
                        self._dtype = self._torch.float64
        else:
            self._device = 'cpu'
            # Set dtype - convert to numpy dtype if needed
            if dtype is None:
                self._dtype = np.float64
            else:
                if hasattr(dtype, '__module__') and 'torch' in dtype.__module__:
                    # Convert torch dtype to numpy
                    if dtype == self._torch.float32:
                        self._dtype = np.float32
                    else:
                        self._dtype = np.float64
                else:
                    # Already a numpy dtype
                    self._dtype = dtype

        beauty_print(f"Backend set to {backend} on device {self._device} with dtype {self._dtype}")

    def get_backend(self) -> str:
        """Get current backend name."""
        return self._backend

    def get_device(self) -> str:
        """Get current device."""
        return self._device

    def get_dtype(self):
        """Get current dtype."""
        return self._dtype

    def ensure_array(self, data):
        """
        Convert input to appropriate array type based on current backend.

        :param data: Input data (list, numpy array, or torch tensor)
        :return: Array in the current backend format
        """
        if self._backend in ('numpy', 'cpp'):
            if isinstance(data, np.ndarray):
                return data.astype(self._dtype)
            elif self._torch_available and isinstance(data, self._torch.Tensor):
                return data.cpu().numpy().astype(self._dtype)
            else:
                return np.array(data, dtype=self._dtype)
        else:  # torch
            if self._torch_available and isinstance(data, self._torch.Tensor):
                data = data.to(device=self._device, dtype=self._dtype)
                return data
            elif isinstance(data, np.ndarray):
                return self._torch.from_numpy(data).to(
                    device=self._device, dtype=self._dtype
                )
            else:
                return self._torch.tensor(
                    data, device=self._device, dtype=self._dtype
                )

    def array(self, data, dtype=None):
        """
        Create array with explicit dtype.

        :param data: Input data
        :param dtype: Override dtype (optional)
        :return: Array in current backend format
        """
        if self._backend in ('numpy', 'cpp'):
            return np.array(data, dtype=dtype if dtype is not None else self._dtype)
        else:
            return self._torch.tensor(data, device=self._device, dtype=dtype if dtype is not None else self._dtype)

    def zeros(self, shape, dtype=None):
        """
        Create zeros array.

        :param shape: Array shape
        :param dtype: Override dtype (optional)
        :return: Zeros array
        """
        if self._backend in ('numpy', 'cpp'):
            return np.zeros(shape, dtype=dtype if dtype is not None else self._dtype)
        else:
            return self._torch.zeros(shape, device=self._device, dtype=dtype if dtype is not None else self._dtype)

    def ones(self, shape, dtype=None):
        """
        Create ones array.

        :param shape: Array shape
        :param dtype: Override dtype (optional)
        :return: Ones array
        """
        if self._backend in ('numpy', 'cpp'):
            return np.ones(shape, dtype=dtype if dtype is not None else self._dtype)
        else:
            return self._torch.ones(shape, device=self._device, dtype=dtype if dtype is not None else self._dtype)

    def eye(self, n, dtype=None):
        """
        Create identity matrix.

        :param n: Matrix size
        :param dtype: Override dtype (optional)
        :return: Identity matrix
        """
        if self._backend in ('numpy', 'cpp'):
            return np.eye(n, dtype=dtype if dtype is not None else self._dtype)
        else:
            return self._torch.eye(n, device=self._device, dtype=dtype if dtype is not None else self._dtype)

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
        """Check if current backend uses NumPy arrays (numpy or cpp)."""
        return self._backend in ('numpy', 'cpp')


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
    backend: Literal['numpy', 'torch', 'cpp'] = 'numpy',
    device: Any = None,
    dtype: Optional[Any] = None
):
    """
    Set the global backend.

    :param backend: 'numpy', 'torch', or 'cpp'
    :param device: For torch: None auto-selects CUDA when available, else CPU. Explicit 'cpu' keeps CPU.
    :param dtype: Data type for arrays
    """
    get_backend_manager().set_backend(backend, device, dtype)


def get_backend() -> str:
    """
    Get current backend name.

    :return: 'numpy', 'torch', or 'cpp'
    """
    return get_backend_manager().get_backend()


def ensure_array(data):
    """
    Convert data to current backend format.

    :param data: Input data
    :return: Array in current backend format
    """
    return get_backend_manager().ensure_array(data)


def to_numpy(x: Any) -> np.ndarray:
    """
    Convert torch tensor to numpy array.

    :param x: Input data
    :return: Numpy array
    """
    import torch
    if isinstance(x, torch.Tensor):
        return x.cpu().numpy()
    elif isinstance(x, np.ndarray):
        return x
    elif isinstance(x, list):
        return np.array(x)
    elif isinstance(x, tuple):
        return np.array(x)
    elif isinstance(x, dict):
        return np.array(x)
    elif isinstance(x, str):
        return x
    return np.array(x)
