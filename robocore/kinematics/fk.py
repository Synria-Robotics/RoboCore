"""Unified forward kinematics public interface.

Usage:
    poses = forward_kinematics(model, q)                 # auto select backend
    end_pose = forward_kinematics(model, q, return_end=True)
    poses_t = forward_kinematics(model, q, backend='torch', device='cpu')

Returns a dict mapping link_name -> 4x4 pose matrix (NumPy array or torch.Tensor)
unless ``return_end=True`` in which case only the end-effector 4x4 pose is
returned.
"""

from __future__ import annotations

from typing import Sequence, Dict, Any, Union

from .fk_numpy import forward_kinematics_numpy as _fk_np

_HAS_TORCH = False
try:  # pragma: no cover
    from .fk_torch import forward_kinematics_torch as _fk_torch  # type: ignore
    import torch  # type: ignore
    _HAS_TORCH = True
except Exception:  # noqa: E722
    torch = None  # type: ignore


def _select_backend(backend: str) -> str:
    if backend == 'auto':
        return 'torch' if _HAS_TORCH else 'numpy'
    if backend not in ('numpy', 'torch'):
        raise ValueError(f"Unsupported backend '{backend}', expected 'auto'|'numpy'|'torch'")
    if backend == 'torch' and not _HAS_TORCH:
        raise RuntimeError("Torch backend requested but PyTorch is not available")
    return backend


def forward_kinematics(
    model: Any,
    q: Sequence[float] | Any,
    *,
    backend: str = 'auto',
    return_end: bool = False,
    device: Any | None = None,
    dtype: Any | None = None,
) -> Union[Dict[str, Any], Any]:
    """Compute forward kinematics for a robot model.

    Parameters
    ----------
    model : RobotModel
        Parsed robot model instance exposing attributes used internally
        (``_chain_joints``, ``_actuated``, ``base_link``, ``end_link``).
    q : sequence | tensor
        Joint configuration of length = dof.
    backend : str, default 'auto'
        'auto' picks torch if available else numpy; or explicitly 'numpy'|'torch'.
    return_end : bool, default False
        If True, return only end-effector 4x4 pose instead of full dict.
    device : torch device (torch backend only)
    dtype : torch dtype (torch backend only). If omitted and device is 'mps',
        defaults to float32 (due to limited float64 support); else float64.

    Returns
    -------
    dict | tensor
        Mapping link->pose or single 4x4 pose if return_end=True.
    """
    b = _select_backend(backend)
    if b == 'numpy':
        poses = _fk_np(model._chain_joints, model._actuated, model.base_link, model.end_link, q)  # type: ignore[attr-defined]
        return poses['end'] if return_end else poses
    # torch path
    # Decide dtype default
    if dtype is None and _HAS_TORCH:  # pragma: no branch
        if device is not None and str(device).startswith('mps'):
            # MPS float64 不稳定，默认使用 float32
            dtype = torch.float32  # type: ignore[attr-defined]
        else:
            dtype = torch.float64  # type: ignore[attr-defined]
    poses = _fk_torch(
        model._chain_joints,  # type: ignore[attr-defined]
        model._actuated,      # type: ignore[attr-defined]
        model.base_link,      # type: ignore[attr-defined]
        model.end_link,       # type: ignore[attr-defined]
        q,
        device=device,
        dtype=dtype,
    )
    return poses['end'] if return_end else poses


__all__ = ["forward_kinematics"]
