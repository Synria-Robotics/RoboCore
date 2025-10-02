"""Backend selection utilities.

Provide a tiny abstraction layer so code can switch between numpy and torch.
"""

from __future__ import annotations

import importlib
from typing import Any, Callable, Sequence

_BACKEND_NAME = "numpy"  # default
_np = importlib.import_module("numpy")
_torch = None


def set_backend(name: str) -> None:
    """Select math backend.

    :param name: 'numpy' or 'torch'.
    """
    global _BACKEND_NAME, _torch
    name = name.lower()
    if name not in ("numpy", "torch"):
        raise ValueError("Unsupported backend: %s" % name)
    if name == "torch" and _torch is None:
        _torch = importlib.import_module("torch")
    _BACKEND_NAME = name


def backend_name() -> str:
    """Return current backend name.

    :return: backend name.
    """
    return _BACKEND_NAME


def _lib():  # internal helper
    return _torch if _BACKEND_NAME == "torch" else _np


def array(data: Any) -> Any:
    """Create array/tensor from data.

    :param data: input (list / sequence / existing tensor).
    :return: backend array.
    """
    lib = _lib()
    if _BACKEND_NAME == "torch":
        return data if isinstance(data, lib.Tensor) else lib.tensor(data, dtype=lib.get_default_dtype())
    return lib.array(data)


def zeros(shape: Sequence[int]) -> Any:
    """Zero array.

    :param shape: shape sequence.
    :return: zeros array.
    """
    lib = _lib()
    return lib.zeros(shape)


def eye(n: int) -> Any:
    """Identity matrix.

    :param n: dimension.
    :return: identity matrix.
    """
    lib = _lib()
    return lib.eye(n)


def sin(x: Any) -> Any:  # convenience wrappers
    return _lib().sin(x)


def cos(x: Any) -> Any:
    return _lib().cos(x)


def stack(xs: Sequence[Any], axis: int = 0) -> Any:
    lib = _lib()
    return lib.stack(xs, axis=axis) if _BACKEND_NAME == "torch" else lib.stack(xs, axis=axis)


def matmul(a: Any, b: Any) -> Any:
    lib = _lib()
    return a @ b if _BACKEND_NAME == "torch" else lib.matmul(a, b)


def as_numpy(x: Any):  # for interop
    """Convert to numpy array if backend is torch.

    :param x: tensor or array.
    :return: numpy array.
    """
    if _BACKEND_NAME == "torch":
        return x.detach().cpu().numpy()
    return x


def norm(x: Any, axis: int | None = None) -> Any:
    """Compute L2 norm.

    :param x: array.
    :param axis: axis along which to compute norm.
    :return: norm value(s).
    """
    lib = _lib()
    if _BACKEND_NAME == "torch":
        return lib.norm(x, dim=axis)
    return lib.linalg.norm(x, axis=axis)


def sqrt(x: Any) -> Any:
    """Square root.

    :param x: array.
    :return: sqrt(x).
    """
    return _lib().sqrt(x)


def dot(a: Any, b: Any) -> Any:
    """Dot product / matrix multiplication.

    :param a: first array.
    :param b: second array.
    :return: dot product.
    """
    lib = _lib()
    if _BACKEND_NAME == "torch":
        return a @ b
    return lib.dot(a, b) if a.ndim == 1 and b.ndim == 1 else lib.matmul(a, b)


def transpose(x: Any) -> Any:
    """Transpose matrix.

    :param x: array.
    :return: transposed array.
    """
    return x.T


def inv(x: Any) -> Any:
    """Matrix inverse.

    :param x: square matrix.
    :return: inverse.
    """
    lib = _lib()
    if _BACKEND_NAME == "torch":
        return lib.linalg.inv(x)
    return lib.linalg.inv(x)


def solve(A: Any, b: Any) -> Any:
    """Solve linear system Ax = b.

    :param A: coefficient matrix.
    :param b: right-hand side.
    :return: solution x.
    """
    lib = _lib()
    if _BACKEND_NAME == "torch":
        return lib.linalg.solve(A, b)
    return lib.linalg.solve(A, b)


def acos(x: Any) -> Any:
    """Inverse cosine.

    :param x: array.
    :return: acos(x).
    """
    return _lib().arccos(x) if _BACKEND_NAME == "numpy" else _lib().acos(x)


def atan2(y: Any, x: Any) -> Any:
    """Inverse tangent with two arguments.

    :param y: y coordinate.
    :param x: x coordinate.
    :return: atan2(y, x).
    """
    return _lib().arctan2(y, x) if _BACKEND_NAME == "numpy" else _lib().atan2(y, x)


def clip(x: Any, min_val: float, max_val: float) -> Any:
    """Clip values to range.

    :param x: array.
    :param min_val: minimum value.
    :param max_val: maximum value.
    :return: clipped array.
    """
    lib = _lib()
    if _BACKEND_NAME == "torch":
        return lib.clamp(x, min_val, max_val)
    return lib.clip(x, min_val, max_val)


def to_list(x: Any) -> list:
    """Convert array/tensor to Python list.

    :param x: array or tensor.
    :return: nested Python list.
    """
    if _BACKEND_NAME == "torch":
        return x.detach().cpu().numpy().tolist()
    return x.tolist() if hasattr(x, 'tolist') else list(x)


__all__ = [
    "set_backend",
    "backend_name",
    "array",
    "zeros",
    "eye",
    "sin",
    "cos",
    "stack",
    "matmul",
    "as_numpy",
    "norm",
    "sqrt",
    "dot",
    "transpose",
    "inv",
    "solve",
    "acos",
    "atan2",
    "clip",
    "to_list",
]
