"""PyTorch forward kinematics solver.

与 `FKSolverNumPy` 功能等价，但使用 torch.Tensor，方便后续在 GPU / 自动求导场景中复用。

注意：当前实现不依赖 autograd，仅使用张量算子做矩阵乘法；若需要对末端姿态关于 q 自动求导，可在上层定义
一个函数包装 `solver.solve(q)` 并调用 `torch.autograd.functional.jacobian`。
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Sequence

try:  # 延迟 import，避免未安装 torch 时影响其它模块
    import torch
except ImportError as e:  # pragma: no cover
    raise ImportError("fk_solver_torch 需要 PyTorch，请先安装: pip install torch") from e

try:
    from robocore.utils.torch_utils import select_device  # type: ignore
except Exception:  # pragma: no cover
    def select_device(d=None):  # fallback
        return torch.device('cpu')

from robocore.transform.transform_core import (
    rpy_to_rotation_matrix_torch,
    axis_angle_to_rotation_matrix_torch,
    axis_translation_torch,
    make_transform_torch,
)

if TYPE_CHECKING:
    from robocore.modeling.robot_model import RobotModel

Tensor = torch.Tensor


class FKSolverTorch:
    """PyTorch-accelerated forward kinematics solver.
    
    Features:
    - Fast matrix operations using PyTorch
    - GPU acceleration support
    - Automatic differentiation compatible
    - Support for revolute, prismatic, and fixed joints
    """
    
    def __init__(self, model: "RobotModel"):
        """Initialize FK solver.
        
        :param model: robot model.
        """
        self.model = model
        self.n = model.dof()
        self.joint_chain = model._chain_joints
        self.actuated_joints = model._actuated
        self.base_link = model.base_link
        self.end_link = model.end_link
    
    def solve(
        self,
        q: Sequence[float] | Tensor,
        return_end_only: bool = False,
        device=None,
        dtype=torch.float64
    ) -> Dict[str, torch.Tensor]:
        """Compute forward kinematics.
        
        :param q: joint configuration (n,).
        :param return_end_only: if True, only return end-effector pose.
        :param device: torch device ('cpu', 'cuda', 'mps', or None for auto).
        :param dtype: torch dtype (default: torch.float64).
        :return: dict of link names to 4x4 pose matrices as torch tensors.
        """
        device = select_device(device)
        
        if not torch.is_tensor(q):
            q = torch.tensor(q, dtype=dtype, device=device)
        else:
            q = q.to(dtype=dtype, device=device)
        
        if q.shape[0] != self.n:
            raise ValueError(f"Expected q with {self.n} elements, got {q.shape[0]}")
        
        q_map = {j.name: q[j.index] for j in self.actuated_joints}
        poses: Dict[str, torch.Tensor] = {
            self.base_link: torch.eye(4, dtype=dtype, device=q.device)
        }
        
        for joint in self.joint_chain:
            parent_pose = poses[joint.parent]
            
            # origin 变换
            R_o = rpy_to_rotation_matrix_torch(
                torch.tensor(joint.origin_rpy[0], dtype=dtype, device=q.device),
                torch.tensor(joint.origin_rpy[1], dtype=dtype, device=q.device),
                torch.tensor(joint.origin_rpy[2], dtype=dtype, device=q.device),
            )
            t_o = torch.tensor(joint.origin_xyz, dtype=dtype, device=q.device)
            T_origin = make_transform_torch(R_o, t_o)
            
            if joint.joint_type == "revolute":
                R_m = axis_angle_to_rotation_matrix_torch(
                    torch.tensor(joint.axis, dtype=dtype, device=q.device),
                    q_map.get(joint.name, torch.tensor(0.0, dtype=dtype, device=q.device))
                )
                t_m = torch.zeros(3, dtype=dtype, device=q.device)
            elif joint.joint_type == "prismatic":
                R_m = torch.eye(3, dtype=dtype, device=q.device)
                t_m = axis_translation_torch(
                    torch.tensor(joint.axis, dtype=dtype, device=q.device),
                    q_map.get(joint.name, torch.tensor(0.0, dtype=dtype, device=q.device))
                )
            else:
                R_m = torch.eye(3, dtype=dtype, device=q.device)
                t_m = torch.zeros(3, dtype=dtype, device=q.device)
            
            T_motion = make_transform_torch(R_m, t_m)
            child_pose = parent_pose @ T_origin @ T_motion
            poses[joint.child] = child_pose
        
        poses["end"] = poses.get(self.end_link, list(poses.values())[-1])
        
        if return_end_only:
            return {"end": poses["end"]}
        
        return poses


__all__ = ["FKSolverTorch"]
