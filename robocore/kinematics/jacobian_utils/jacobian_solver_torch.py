"""PyTorch Jacobian solver.

Provides analytic, numeric (finite-difference), and autograd Jacobian computation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Optional
import math

try:
    import torch
except ImportError as e:
    raise ImportError("jacobian_solver_torch 需要 PyTorch, 请先: pip install torch") from e

try:
    from ...utils.torch_utils import select_device
except Exception:
    def select_device(device=None):
        if device is not None:
            return torch.device(device)
        return torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))

from robocore.transform.transform_core import orientation_error_torch

if TYPE_CHECKING:
    from robocore.modeling.robot_model import RobotModel

Tensor = torch.Tensor

__all__ = ["JacobianSolverTorch"]


class JacobianSolverTorch:
    """PyTorch-accelerated Jacobian solver.
    
    Features:
    - Analytic (geometric) Jacobian computation
    - Numeric (finite-difference) Jacobian computation  
    - Autograd-based Jacobian computation
    - GPU acceleration support
    - Automatic differentiation compatible
    """
    
    def __init__(self, model: "RobotModel"):
        """Initialize Jacobian solver.
        
        :param model: robot model.
        """
        self.model = model
        self.n = model.dof()
    
    def solve(
        self,
        q: Tensor | list,
        method: Literal["analytic", "numeric", "autograd"] = "analytic",
        epsilon: float = 5e-5,
        use_central_diff: bool = True,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> Tensor:
        """Compute 6×n Jacobian matrix.
        
        :param q: joint configuration (n,).
        :param method: 'analytic', 'numeric', or 'autograd'.
        :param epsilon: finite difference step size (numeric only).
        :param use_central_diff: use central difference if True (numeric only).
        :param device: torch device.
        :param dtype: torch dtype.
        :return: 6×n Jacobian matrix as torch tensor.
        """
        device = select_device(device)
        
        # Default dtype handling (MPS doesn't support float64 well)
        if dtype is None:
            dtype = torch.float32 if str(device).startswith('mps') else torch.float64
        
        if method == "analytic":
            return self._solve_analytic(q, device, dtype)
        elif method == "numeric":
            return self._solve_numeric(q, epsilon, use_central_diff, device, dtype)
        elif method == "autograd":
            return self._solve_autograd(q, device, dtype)
        else:
            raise ValueError(f"Unknown method '{method}', expected 'analytic', 'numeric', or 'autograd'")
    
    def _solve_analytic(self, q, device, dtype) -> Tensor:
        """Compute analytic (geometric) Jacobian."""
        if not torch.is_tensor(q):
            q = torch.tensor(q, dtype=dtype, device=device)
        else:
            q = q.to(dtype=dtype, device=device)
        
        if q.shape[0] != self.n:
            raise ValueError(f"Configuration length {q.shape[0]} != dof {self.n}")
        
        # Build forward transforms
        T_parent = torch.eye(4, dtype=dtype, device=device)
        q_map = {js.name: q[js.index] for js in self.model._actuated}
        
        p_list = [None] * self.n
        z_list = [None] * self.n
        
        end_T = T_parent
        
        for urdf_joint in self.model._chain_joints:
            R_origin = self._rpy_matrix_torch(
                torch.tensor(urdf_joint.origin_rpy[0], dtype=dtype, device=device),
                torch.tensor(urdf_joint.origin_rpy[1], dtype=dtype, device=device),
                torch.tensor(urdf_joint.origin_rpy[2], dtype=dtype, device=device),
            )
            t_origin = torch.tensor(urdf_joint.origin_xyz, dtype=dtype, device=device)
            
            T_origin = torch.eye(4, dtype=dtype, device=device)
            T_origin[:3, :3] = R_origin
            T_origin[:3, 3] = t_origin
            T_joint_origin = T_parent @ T_origin
            
            if urdf_joint.joint_type in ("revolute", "prismatic"):
                js = next(js for js in self.model._actuated if js.name == urdf_joint.name)
                axis_local = torch.tensor(urdf_joint.axis, dtype=dtype, device=device)
                axis_norm = torch.linalg.norm(axis_local)
                if axis_norm > 1e-10:
                    axis_local = axis_local / axis_norm
                z_i = T_joint_origin[:3, :3] @ axis_local
                p_i = T_joint_origin[:3, 3].clone()
                p_list[js.index] = p_i
                z_list[js.index] = z_i
            
            R_motion = torch.eye(3, dtype=dtype, device=device)
            t_motion = torch.zeros(3, dtype=dtype, device=device)
            
            if urdf_joint.joint_type == "revolute":
                theta = q_map.get(urdf_joint.name, torch.tensor(0.0, dtype=dtype, device=device))
                R_motion = self._axis_rotation_torch(
                    torch.tensor(urdf_joint.axis, dtype=dtype, device=device),
                    theta
                )
            elif urdf_joint.joint_type == "prismatic":
                d = q_map.get(urdf_joint.name, torch.tensor(0.0, dtype=dtype, device=device))
                t_motion = self._axis_translation_torch(
                    torch.tensor(urdf_joint.axis, dtype=dtype, device=device),
                    d
                )
            
            T_motion = torch.eye(4, dtype=dtype, device=device)
            T_motion[:3, :3] = R_motion
            T_motion[:3, 3] = t_motion
            
            T_child = T_joint_origin @ T_motion
            T_parent = T_child
            end_T = T_child
        
        p_end = end_T[:3, 3]
        
        # Assemble Jacobian
        J_geo = torch.zeros((6, self.n), dtype=dtype, device=device)
        for i in range(self.n):
            z_i = z_list[i]
            p_i = p_list[i]
            if z_i is None or p_i is None:
                raise RuntimeError("Internal error: missing joint axis or origin position")
            
            js = self.model._actuated[i]
            if js.joint_type == "revolute":
                J_geo[:3, i] = torch.linalg.cross(z_i, (p_end - p_i))
                J_geo[3:6, i] = z_i
            elif js.joint_type == "prismatic":
                J_geo[:3, i] = z_i
        
        # Transform angular part to end-effector frame
        R_end = end_T[:3, :3]
        J = J_geo.clone()
        J[3:6, :] = R_end.T @ J_geo[3:6, :]
        return J
    
    def _solve_numeric(self, q, epsilon, use_central_diff, device, dtype) -> Tensor:
        """Compute numeric Jacobian using finite differences."""
        # Import FK solver here to avoid circular dependency
        from ..fk_utils.fk_solver_torch import FKSolverTorch
        
        if not torch.is_tensor(q):
            q = torch.tensor(q, dtype=dtype, device=device)
        else:
            q = q.to(dtype=dtype, device=device)
        
        J = torch.zeros((6, self.n), dtype=dtype, device=device)
        fk_solver = FKSolverTorch(self.model)
        
        if use_central_diff:
            T_ref = fk_solver.solve(q, return_end_only=True, device=device, dtype=dtype)["end"]
            R_ref = T_ref[:3, :3].clone()
            
            for i in range(self.n):
                qp = q.clone()
                qp[i] += epsilon
                T_pos = fk_solver.solve(qp, return_end_only=True, device=device, dtype=dtype)["end"]
                p_pos = T_pos[:3, 3]
                R_pos = T_pos[:3, :3]
                
                qn = q.clone()
                qn[i] -= epsilon
                T_neg = fk_solver.solve(qn, return_end_only=True, device=device, dtype=dtype)["end"]
                p_neg = T_neg[:3, 3]
                R_neg = T_neg[:3, :3]
                
                J[:3, i] = (p_pos - p_neg) / (2 * epsilon)
                
                err_pos = orientation_error_torch(R_ref, R_pos)
                err_neg = orientation_error_torch(R_ref, R_neg)
                J[3:6, i] = (err_pos - err_neg) / (2 * epsilon)
        else:
            T_ref = fk_solver.solve(q, return_end_only=True, device=device, dtype=dtype)["end"]
            R_ref = T_ref[:3, :3]
            p_ref = T_ref[:3, 3]
            
            for i in range(self.n):
                qp = q.clone()
                qp[i] += epsilon
                T_pos = fk_solver.solve(qp, return_end_only=True, device=device, dtype=dtype)["end"]
                p_pos = T_pos[:3, 3]
                R_pos = T_pos[:3, :3]
                
                J[:3, i] = (p_pos - p_ref) / epsilon
                err = orientation_error_torch(R_ref, R_pos)
                J[3:6, i] = err / epsilon
        
        return J
    
    def _solve_autograd(self, q, device, dtype) -> Tensor:
        """Compute Jacobian using PyTorch autograd."""
        from ..fk_utils.fk_solver_torch import FKSolverTorch
        
        if not torch.is_tensor(q):
            q = torch.tensor(q, dtype=dtype, device=device, requires_grad=True)
        else:
            q = q.to(dtype=dtype, device=device)
            q.requires_grad_(True)
        
        fk_solver = FKSolverTorch(self.model)
        
        def pose_vec(q_):
            T = fk_solver.solve(q_, return_end_only=True, device=device, dtype=dtype)["end"]
            p = T[:3, 3]
            R = T[:3, :3].reshape(-1)
            return torch.cat([p, R])  # (12,)
        
        J_big = torch.autograd.functional.jacobian(
            pose_vec, q, create_graph=False, vectorize=False
        )  # (12,n)
        pJ = J_big[:3, :]  # (3,n)
        R_flat_J = J_big[3:, :]  # (9,n)
        
        # Current pose
        T_ref = fk_solver.solve(q, return_end_only=True, device=device, dtype=dtype)["end"]
        R = T_ref[:3, :3]
        
        J = torch.zeros((6, self.n), dtype=dtype, device=device)
        J[:3, :] = pJ
        R_T = R.transpose(0, 1)
        
        for j in range(self.n):
            dR_flat = R_flat_J[:, j]
            dR = dR_flat.view(3, 3)
            skew = dR @ R_T
            wx = (skew[2, 1] - skew[1, 2]) * 0.5
            wy = (skew[0, 2] - skew[2, 0]) * 0.5
            wz = (skew[1, 0] - skew[0, 1]) * 0.5
            w_world = torch.stack([wx, wy, wz])
            w_end = R_T @ w_world
            J[3:6, j] = w_end
        
        return J.detach()
    
    # ============================================================================
    # Helper functions
    # ============================================================================
    
    @staticmethod
    def _rpy_matrix_torch(r, p, y):
        """Compute rotation matrix from roll-pitch-yaw."""
        sr, cr = torch.sin(r), torch.cos(r)
        sp, cp = torch.sin(p), torch.cos(p)
        sy, cy = torch.sin(y), torch.cos(y)
        return torch.stack([
            torch.stack([cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr]),
            torch.stack([sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr]),
            torch.stack([-sp, cp * sr, cp * cr]),
        ])
    
    @staticmethod
    def _axis_rotation_torch(axis, theta):
        """Compute rotation matrix for rotation about axis by angle."""
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
    
    @staticmethod
    def _axis_translation_torch(axis, d):
        """Compute translation along axis."""
        norm = torch.linalg.norm(axis)
        if norm < 1e-12:
            return torch.zeros(3, dtype=axis.dtype, device=axis.device)
        return axis / norm * d
