"""Torch device selection helper.

Copyright (c) 2025 Synria Robotics Co., Ltd.

Licensed under the MIT License.

Author: Synria Robotics Team
Website: https://synriarobotics.ai
"""

from __future__ import annotations

try:
    import torch
except ImportError as e:  # pragma: no cover
    raise ImportError("torch_utils 需要 PyTorch") from e


def select_device(device=None):
    """Select torch device (cpu/cuda). MPS support removed.

    If user explicitly passes an mps device string, raise to avoid silent misuse.
    """
    if device is not None:
        d = torch.device(device) if isinstance(device, str) else device
        if str(d).startswith('mps'):
            raise ValueError("MPS support has been removed. Please use 'cpu' or 'cuda'.")
        return d
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')