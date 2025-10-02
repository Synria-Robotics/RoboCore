"""PyTorch analytic (geometric) Jacobian.

与 NumPy 版 `analytic_jacobian_numpy` 同步，实现 6×n 几何雅可比，并对旋转块变换到末端坐标系。
返回 torch.Tensor，dtype 默认 float64。
"""
from __future__ import annotations

import math
from typing import List

try:
    import torch
except ImportError as e:  # pragma: no cover
    raise ImportError("jacobian_analytic_torch 需要 PyTorch，请先安装: pip install torch") from e

Tensor = torch.Tensor

def analytic_jacobian_torch(model, q, *, device=None, dtype=torch.float64):
    if not torch.is_tensor(q):
        q = torch.tensor(q, dtype=dtype, device=device)
    else:
        q = q.to(dtype=dtype, device=device)
    n = model.dof()
    if q.numel() != n:
        raise ValueError(f"Configuration length {q.numel()} != dof {n}")

    q_map = {js.name: q[js.index] for js in model._actuated}  # type: ignore[attr-defined]
    T_parent = torch.eye(4, dtype=dtype, device=q.device)
    p_list: List[Tensor] = [None] * n  # type: ignore
    z_list: List[Tensor] = [None] * n  # type: ignore
    end_T = T_parent

    def rpy_matrix(r, p, y):
        sr, cr = torch.sin(r), torch.cos(r)
        sp, cp = torch.sin(p), torch.cos(p)
        sy, cy = torch.sin(y), torch.cos(y)
        return torch.stack([
            torch.stack([cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr]),
            torch.stack([sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr]),
            torch.stack([-sp, cp * sr, cp * cr]),
        ])

    def axis_rot(axis, theta):
        axis = axis / (torch.linalg.norm(axis) + 1e-15)
        ax, ay, az = axis
        ct, st = torch.cos(theta), torch.sin(theta)
        vt = 1 - ct
        return torch.stack([
            torch.stack([ct + ax * ax * vt, ax * ay * vt - az * st, ax * az * vt + ay * st]),
            torch.stack([ay * ax * vt + az * st, ct + ay * ay * vt, ay * az * vt - ax * st]),
            torch.stack([az * ax * vt - ay * st, az * ay * vt + ax * st, ct + az * az * vt]),
        ])

    def axis_trans(axis, d):
        axis = axis / (torch.linalg.norm(axis) + 1e-15)
        return axis * d

    for urdf_joint in model._chain_joints:  # type: ignore[attr-defined]
        R_o = rpy_matrix(
            torch.tensor(urdf_joint.origin_rpy[0], dtype=dtype, device=q.device),
            torch.tensor(urdf_joint.origin_rpy[1], dtype=dtype, device=q.device),
            torch.tensor(urdf_joint.origin_rpy[2], dtype=dtype, device=q.device),
        )
        t_o = torch.tensor(urdf_joint.origin_xyz, dtype=dtype, device=q.device)
        T_origin = torch.eye(4, dtype=dtype, device=q.device)
        T_origin[:3, :3] = R_o
        T_origin[:3, 3] = t_o
        T_joint_origin = T_parent @ T_origin

        if urdf_joint.joint_type in ("revolute", "prismatic"):
            js = next(js for js in model._actuated if js.name == urdf_joint.name)  # type: ignore[attr-defined]
            axis_local = torch.tensor(urdf_joint.axis, dtype=dtype, device=q.device)
            z_i = T_joint_origin[:3, :3] @ (axis_local / (torch.linalg.norm(axis_local) + 1e-15))
            p_i = T_joint_origin[:3, 3].clone()
            p_list[js.index] = p_i
            z_list[js.index] = z_i

        R_m = torch.eye(3, dtype=dtype, device=q.device)
        t_m = torch.zeros(3, dtype=dtype, device=q.device)
        if urdf_joint.joint_type == "revolute":
            theta = q_map.get(urdf_joint.name, torch.tensor(0.0, dtype=dtype, device=q.device))
            R_m = axis_rot(torch.tensor(urdf_joint.axis, dtype=dtype, device=q.device), theta)
        elif urdf_joint.joint_type == "prismatic":
            dval = q_map.get(urdf_joint.name, torch.tensor(0.0, dtype=dtype, device=q.device))
            t_m = axis_trans(torch.tensor(urdf_joint.axis, dtype=dtype, device=q.device), dval)

        T_motion = torch.eye(4, dtype=dtype, device=q.device)
        T_motion[:3, :3] = R_m
        T_motion[:3, 3] = t_m
        T_child = T_joint_origin @ T_motion
        T_parent = T_child
        end_T = T_child

    p_end = end_T[:3, 3]
    J_geo = torch.zeros((6, n), dtype=dtype, device=q.device)
    for i in range(n):
        z_i = z_list[i]; p_i = p_list[i]
        if z_i is None or p_i is None:
            raise RuntimeError("缺少关节轴或位置，Jacobian 构造失败")
        js = model._actuated[i]
        if js.joint_type == "revolute":
            J_geo[:3, i] = torch.cross(z_i, (p_end - p_i), dim=0)
            J_geo[3:6, i] = z_i
        else:  # prismatic
            J_geo[:3, i] = z_i
    R_end = end_T[:3, :3]
    J = J_geo.clone()
    J[3:6, :] = R_end.T @ J_geo[3:6, :]
    return J

__all__ = ["analytic_jacobian_torch"]
