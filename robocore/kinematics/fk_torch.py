"""PyTorch forward kinematics implementation.

与 `forward_kinematics_numpy` 功能等价，但使用 torch.Tensor，方便后续在 GPU / 自动求导场景中复用。

注意：当前实现不依赖 autograd，仅使用张量算子做矩阵乘法；若需要对末端姿态关于 q 自动求导，可在上层定义
一个函数包装 `forward_kinematics_torch_end(model, q)` 并调用 `torch.autograd.functional.jacobian`。
"""
from __future__ import annotations

from typing import Dict, Sequence
import math

try:  # 延迟 import，避免未安装 torch 时影响其它模块
    import torch
except ImportError as e:  # pragma: no cover
    raise ImportError("fk_torch 需要 PyTorch，请先安装: pip install torch") from e
try:
    from .torch_utils import select_device  # type: ignore
except Exception:  # pragma: no cover
    def select_device(d=None):  # fallback
        return torch.device('cpu')

Tensor = torch.Tensor

def _rpy_matrix_torch(r, p, y):
    sr, cr = torch.sin(r), torch.cos(r)
    sp, cp = torch.sin(p), torch.cos(p)
    sy, cy = torch.sin(y), torch.cos(y)
    return torch.stack([
        torch.stack([cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr]),
        torch.stack([sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr]),
        torch.stack([-sp,      cp * sr,               cp * cr]),
    ])


def _axis_rotation_torch(axis, theta):
    norm = torch.linalg.norm(axis)
    if norm < 1e-12:
        return torch.eye(3, dtype=axis.dtype, device=axis.device)
    a = axis / norm
    ax, ay, az = a
    ct, st = torch.cos(theta), torch.sin(theta)
    vt = 1 - ct
    return torch.stack([
        torch.stack([ct + ax * ax * vt, ax * ay * vt - az * st, ax * az * vt + ay * st]),
        torch.stack([ay * ax * vt + az * st, ct + ay * ay * vt, ay * az * vt - ax * st]),
        torch.stack([az * ax * vt - ay * st, az * ay * vt + ax * st, ct + az * az * vt]),
    ])


def _axis_translation_torch(axis, d):
    norm = torch.linalg.norm(axis)
    if norm < 1e-12:
        return torch.zeros(3, dtype=axis.dtype, device=axis.device)
    return axis / norm * d


def _make_transform(R, t):
    T = torch.eye(4, dtype=R.dtype, device=R.device)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def forward_kinematics_torch(joint_chain, actuated_joints, base_link: str, end_link: str, q: Sequence[float] | Tensor, *, device=None, dtype=torch.float64):
    """执行串联链路的 FK，返回每个 link 的 4x4 位姿 (torch.Tensor)。

    参数与 NumPy 版本基本一致；q 可为 list / numpy / torch (长度 = dof)。
    """
    device = select_device(device)
    if not torch.is_tensor(q):
        q = torch.tensor(q, dtype=dtype, device=device)
    else:
        q = q.to(dtype=dtype, device=device)

    q_map = {j.name: q[j.index] for j in actuated_joints}
    poses: Dict[str, torch.Tensor] = {base_link: torch.eye(4, dtype=dtype, device=q.device)}

    for joint in joint_chain:
        parent_pose = poses[joint.parent]
        # origin 变换
        R_o = _rpy_matrix_torch(
            torch.tensor(joint.origin_rpy[0], dtype=dtype, device=q.device),
            torch.tensor(joint.origin_rpy[1], dtype=dtype, device=q.device),
            torch.tensor(joint.origin_rpy[2], dtype=dtype, device=q.device),
        )
        t_o = torch.tensor(joint.origin_xyz, dtype=dtype, device=q.device)
        T_origin = _make_transform(R_o, t_o)

        if joint.joint_type == "revolute":
            R_m = _axis_rotation_torch(torch.tensor(joint.axis, dtype=dtype, device=q.device), q_map.get(joint.name, torch.tensor(0.0, dtype=dtype, device=q.device)))
            t_m = torch.zeros(3, dtype=dtype, device=q.device)
        elif joint.joint_type == "prismatic":
            R_m = torch.eye(3, dtype=dtype, device=q.device)
            t_m = _axis_translation_torch(torch.tensor(joint.axis, dtype=dtype, device=q.device), q_map.get(joint.name, torch.tensor(0.0, dtype=dtype, device=q.device)))
        else:
            R_m = torch.eye(3, dtype=dtype, device=q.device)
            t_m = torch.zeros(3, dtype=dtype, device=q.device)
        T_motion = _make_transform(R_m, t_m)
        child_pose = parent_pose @ T_origin @ T_motion
        poses[joint.child] = child_pose

    poses["end"] = poses.get(end_link, list(poses.values())[-1])
    return poses


def forward_kinematics_torch_end(model, q: Sequence[float] | Tensor, *, device=None, dtype=torch.float64):
    """仅返回末端 link 的 4x4 位姿 (torch)。"""
    device = select_device(device)
    poses = forward_kinematics_torch(model._chain_joints, model._actuated, model.base_link, model.end_link, q, device=device, dtype=dtype)  # type: ignore[attr-defined]
    return poses["end"]


__all__ = ["forward_kinematics_torch", "forward_kinematics_torch_end"]
