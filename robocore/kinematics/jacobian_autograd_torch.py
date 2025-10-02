"""使用 autograd 验证解析几何 Jacobian。

实现思路：
  - 函数 pose_vec(q) -> [p(3), R(9)]
  - 通过 torch.autograd.functional.jacobian 得到 d[p,R]/dq (12×n)
  - 对每列 j，取 dR_j (3×3) 计算 skew = dR_j @ R^T => skew(ω_world)
  - ω_world = vee(skew)
  - 旋转到末端坐标系：ω_end = R^T @ ω_world
  - 组合线速度块和 ω_end 得到 6×n Jacobian，与 analytic_jacobian_torch 输出对齐
"""
from __future__ import annotations

try:
    import torch
except ImportError as e:  # pragma: no cover
    raise ImportError("需要 PyTorch 以使用 autograd Jacobian") from e

from .fk_torch import forward_kinematics_torch_end
from .torch_utils import select_device

def autograd_geometric_jacobian_torch(model, q, *, device=None, dtype=torch.float64):
    device = select_device(device)
    if device.type == 'mps' and dtype == torch.float64:
        dtype = torch.float32
    if not torch.is_tensor(q):
        q = torch.tensor(q, dtype=dtype, device=device, requires_grad=True)
    else:
        q = q.to(dtype=dtype, device=device)
        q.requires_grad_(True)
    n = model.dof()
    def pose_vec(q_):
        T = forward_kinematics_torch_end(model, q_, device=device, dtype=dtype)
        p = T[:3, 3]
        R = T[:3, :3].reshape(-1)
        return torch.cat([p, R])  # (12,)
    J_big = torch.autograd.functional.jacobian(pose_vec, q, create_graph=False, vectorize=False)  # (12,n)
    pJ = J_big[:3, :]  # (3,n)
    R_flat_J = J_big[3:, :]  # (9,n)
    # 末端当前姿态
    T_ref = forward_kinematics_torch_end(model, q, device=device, dtype=dtype)
    R = T_ref[:3, :3]
    J = torch.zeros((6, n), dtype=dtype, device=device)
    J[:3, :] = pJ
    R_T = R.transpose(0,1)
    for j in range(n):
        dR_flat = R_flat_J[:, j]
        dR = dR_flat.view(3, 3)
        skew = dR @ R_T
        wx = (skew[2,1] - skew[1,2]) * 0.5
        wy = (skew[0,2] - skew[2,0]) * 0.5
        wz = (skew[1,0] - skew[0,1]) * 0.5
        w_world = torch.stack([wx, wy, wz])
        w_end = R_T @ w_world
        J[3:6, j] = w_end
    return J.detach()

__all__ = ["autograd_geometric_jacobian_torch"]
