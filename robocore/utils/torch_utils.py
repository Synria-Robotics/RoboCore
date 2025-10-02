"""Torch device selection helper."""
from __future__ import annotations

try:
    import torch
except ImportError as e:  # pragma: no cover
    raise ImportError("torch_utils 需要 PyTorch") from e


def select_device(device=None):
    if device is not None:
        return torch.device(device) if isinstance(device, str) else device
    if torch.cuda.is_available():
        return torch.device('cuda')
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():  # type: ignore[attr-defined]
        return torch.device('mps')
    return torch.device('cpu')