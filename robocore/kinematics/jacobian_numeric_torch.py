"""Torch 数值雅可比 (与 numpy numeric_jacobian_numpy 逻辑一致)。

提供 central difference 与 forward difference 两种模式；返回 6×n 张量：
  前 3 行：末端位置对关节的偏导
  后 3 行：轴角误差（参考姿态为原始 R_ref）关于关节的偏导
"""
from __future__ import annotations

try:
    import torch
except ImportError as e:  # pragma: no cover
    raise ImportError("需要 PyTorch 来使用 jacobian_numeric_torch") from e

from .fk_torch import forward_kinematics_torch_end
from .torch_utils import select_device

def _orientation_error_torch(Ra, Rb):
    R_err = Ra.transpose(0,1) @ Rb
    trace = torch.trace(R_err)
    angle = torch.acos(torch.clamp((trace - 1.0) * 0.5, -1.0, 1.0))
    if angle < 1e-10:
        return torch.zeros(3, dtype=Ra.dtype, device=Ra.device)
    denom = 2.0 * torch.sin(angle)
    wx = (R_err[2,1] - R_err[1,2]) / denom
    wy = (R_err[0,2] - R_err[2,0]) / denom
    wz = (R_err[1,0] - R_err[0,1]) / denom
    axis = torch.stack([wx, wy, wz])
    return axis * angle

def numeric_jacobian_torch(model, q, epsilon: float = 5e-5, use_central_diff: bool = True, *, device=None, dtype=torch.float64):
    device = select_device(device)
    if device.type == 'mps' and dtype == torch.float64:  # MPS 不支持 float64
        dtype = torch.float32
    if not torch.is_tensor(q):
        q = torch.tensor(q, dtype=dtype, device=device)
    else:
        q = q.to(dtype=dtype, device=device)
    n = model.dof()
    J = torch.zeros((6, n), dtype=dtype, device=device)
    # 参考姿态
    T_ref = forward_kinematics_torch_end(model, q, device=device, dtype=dtype)
    R_ref = T_ref[:3, :3].clone()
    p_cache = {}
    if use_central_diff:
        for i in range(n):
            qp = q.clone(); qp[i] += epsilon
            T_pos = forward_kinematics_torch_end(model, qp, device=device, dtype=dtype)
            p_pos = T_pos[:3, 3]; R_pos = T_pos[:3, :3]
            qn = q.clone(); qn[i] -= epsilon
            T_neg = forward_kinematics_torch_end(model, qn, device=device, dtype=dtype)
            p_neg = T_neg[:3, 3]; R_neg = T_neg[:3, :3]
            J[:3, i] = (p_pos - p_neg) / (2 * epsilon)
            err_pos = _orientation_error_torch(R_ref, R_pos)
            err_neg = _orientation_error_torch(R_ref, R_neg)
            J[3:6, i] = (err_pos - err_neg) / (2 * epsilon)
    else:
        for i in range(n):
            qp = q.clone(); qp[i] += epsilon
            T_pos = forward_kinematics_torch_end(model, qp, device=device, dtype=dtype)
            p_pos = T_pos[:3, 3]; R_pos = T_pos[:3, :3]
            p_ref = T_ref[:3, 3]
            J[:3, i] = (p_pos - p_ref) / epsilon
            err = _orientation_error_torch(R_ref, R_pos)
            J[3:6, i] = err / epsilon
    return J

__all__ = ["numeric_jacobian_torch"]
