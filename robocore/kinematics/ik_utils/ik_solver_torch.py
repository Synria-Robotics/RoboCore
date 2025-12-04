"""PyTorch 版本逆运动学求解器 (IKSolverTorch)。

Copyright (c) 2025 Synria Robotics Co., Ltd.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see <https://www.gnu.org/licenses/>.

Author: Synria Robotics Team
Website: https://synriarobotics.ai
"""

from __future__ import annotations

import math
from typing import Dict, Optional, TYPE_CHECKING, Sequence

try:
    import torch
except ImportError as e:  # pragma: no cover
    raise ImportError("ik_solver_torch 需要 PyTorch, 请先: pip install torch") from e

from ..jacobian_utils.jacobian_solver_torch import JacobianSolverTorch
from ..fk_utils.fk_solver_torch import FKSolverTorch
from robocore.utils.backend import set_backend, get_backend
from robocore.transform import rotation_error, rpy_to_matrix, axis_angle_to_matrix
try:  # 可能存在设备选择工具
    from ...utils.torch_utils import select_device  # type: ignore
except Exception:  # pragma: no cover
    def select_device():  # 兜底，仅支持 cpu/cuda
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

if TYPE_CHECKING:
    from robocore.modeling.robot_model import RobotModel
    from torch import Tensor
else:
    Tensor = torch.Tensor


class IKSolverTorch:
    def __init__(
        self,
        model,
        max_iters: int = 100,
        pos_tol: float = 1e-4,
        ori_tol: float = 1e-3,
        min_damping: float = 1e-4,  # 与 NumPy 一致
        max_damping: float = 5e-2,  # 与 NumPy 一致
        base_step: float = 1.0,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        self.model = model
        self.max_iters = max_iters
        self.pos_tol = pos_tol
        self.ori_tol = ori_tol
        self.min_damping = min_damping
        self.max_damping = max_damping
        self.base_step = base_step
        self.device = device if device is not None else select_device()

        # 统一 dtype 默认 float64（用户可覆盖）
        self.dtype = dtype if dtype is not None else torch.float64
        # 关节数量（假定 model._actuated 与 numpy 版本一致）
        self.n = len(getattr(model, "_actuated"))
        # Initialize FK solver
        self.fk_solver = FKSolverTorch(model)
        # Initialize Jacobian solver
        self.jacobian_solver = JacobianSolverTorch(model)

        # 预分配常用tensor以减少内存分配开销
        self._eye6 = torch.eye(6, dtype=self.dtype, device=self.device)

    # -------------------- 主求解 --------------------
    def solve(
        self,
        target_pose,
        q0,
        *,
        method: str = "pinv",
        pos_weight: float = 1.0,
        ori_weight: float = 1.0,
        # Local task options
        target_link: str | None = None,
        row_mask: Optional[Sequence[int | bool]] = None,
        # Redundancy / nullspace
        nullspace_gain: float = 0.0,
        joint_centering: bool = True,
        joint_center_gain: float = 0.2,
        joint_center_weights: Optional[Sequence[float]] = None,
        transpose_gain: Optional[float] = None,
        adaptive_damping: bool = True,
        adaptive_step: bool = True,
        use_numeric_jacobian: bool = False,
        use_central_diff: bool = True,
        max_step_norm: float = 0.5,  # 与 NumPy 一致（原来是 0.3）
        backtrack: bool = False,  # 默认关闭以加速
        refine: bool = False,  # 默认关闭以加速
        refine_iters: int = 5,  # 减少refine迭代次数
        refine_pos_tol: Optional[float] = None,
        refine_ori_tol: Optional[float] = None,
        # 额外增强参数
        restarts: int = 0,
        restart_noise: float = 0.25,  # 相对随机扰动幅度 (弧度)
        random_seed: Optional[int] = None,
        verbose: bool = False,  # 批处理模式使用
    ) -> Dict:
        """Solve inverse kinematics (supports both single and batch modes).
        
        Modes:
        - Single mode: target_pose [4,4], q0 [n] → returns dict with 'q', 'success', etc.
        - Batch mode: target_pose [B,4,4], q0 [B,n] → returns dict with 'q' [B,n], 'success' [B], etc.
        """
        # Convert inputs to tensors
        if not torch.is_tensor(target_pose):
            target_pose = torch.tensor(target_pose, dtype=self.dtype, device=self.device)
        else:
            target_pose = target_pose.to(dtype=self.dtype, device=self.device)
        
        if not torch.is_tensor(q0):
            q0 = torch.tensor(q0, dtype=self.dtype, device=self.device)
        else:
            q0 = q0.to(dtype=self.dtype, device=self.device)
        
        # Detect batch mode
        is_batch = target_pose.ndim == 3  # [B, 4, 4]
        
        if is_batch:
            # Batch mode
            return self._solve_batch(
                target_pose, q0,
                method=method,
                pos_weight=pos_weight,
                ori_weight=ori_weight,
                max_step_norm=max_step_norm,
                verbose=verbose
            )
        else:
            # Single mode - existing implementation
            return self._solve_single(
                target_pose, q0,
                method=method,
                pos_weight=pos_weight,
                ori_weight=ori_weight,
                target_link=target_link,
                row_mask=row_mask,
                nullspace_gain=nullspace_gain,
                joint_centering=joint_centering,
                joint_center_gain=joint_center_gain,
                joint_center_weights=joint_center_weights,
                transpose_gain=transpose_gain,
                adaptive_damping=adaptive_damping,
                adaptive_step=adaptive_step,
                use_numeric_jacobian=use_numeric_jacobian,
                use_central_diff=use_central_diff,
                max_step_norm=max_step_norm,
                backtrack=backtrack,
                refine=refine,
                refine_iters=refine_iters,
                refine_pos_tol=refine_pos_tol,
                refine_ori_tol=refine_ori_tol,
                restarts=restarts,
                restart_noise=restart_noise,
                random_seed=random_seed
            )
    
    def _solve_single(
        self,
        target_pose,
        q0,
        *,
        method: str = "pinv",
        pos_weight: float = 1.0,
        ori_weight: float = 1.0,
        target_link: str | None = None,
        row_mask: Optional[Sequence[int | bool]] = None,
        nullspace_gain: float = 0.0,
        joint_centering: bool = True,
        joint_center_gain: float = 0.2,
        joint_center_weights: Optional[Sequence[float]] = None,
        transpose_gain: Optional[float] = None,
        adaptive_damping: bool = True,
        adaptive_step: bool = True,
        use_numeric_jacobian: bool = False,
        use_central_diff: bool = True,
        max_step_norm: float = 0.3,
        backtrack: bool = False,
        refine: bool = False,
        refine_iters: int = 5,
        refine_pos_tol: Optional[float] = None,
        refine_ori_tol: Optional[float] = None,
        restarts: int = 0,
        restart_noise: float = 0.25,
        random_seed: Optional[int] = None,
    ) -> Dict:
        """Single-mode IK solver (original implementation)."""
        method = method.lower()
        if method not in ("dls", "pinv", "transpose"):
            raise ValueError(f"未知 IK 方法 {method}")

        if random_seed is not None:
            torch.manual_seed(random_seed)

        base_q0 = q0.clone() if torch.is_tensor(q0) else torch.tensor(q0, dtype=self.dtype, device=self.device)
        if base_q0.numel() != self.n:
            raise ValueError(f"q0 size {base_q0.numel()} != dof {self.n}")

        # target_link support: treat intermediate link as effective end-effector if provided
        R_target = target_pose[:3, :3]
        p_target = target_pose[:3, 3]

        if row_mask is not None:
            mask_bool = [bool(m) for m in row_mask]
            if len(mask_bool) != 6:
                raise ValueError("row_mask must have length 6")
        else:
            mask_bool = None

        attempt_results = []

        def run_one(q_init: Tensor):
            q = q_init.clone()
            best_q = q
            best_err = torch.tensor(math.inf, dtype=self.dtype, device=self.device)
            plateau_counter = 0
            prev_err_norm = torch.tensor(math.inf, dtype=self.dtype, device=self.device)
            jac_type_local = "analytic"
            final_pos_err = float('inf')
            final_ori_err = float('inf')

            for it in range(1, self.max_iters + 1):
                # Set backend to torch for transform operations
                set_backend("torch", device=str(self.device), dtype=self.dtype)
                
                if target_link is None:
                    T_cur = self.fk_solver.solve(q, return_end_only=True, device=self.device, dtype=self.dtype)["end"]
                else:
                    T_cur = self._fk_until_torch(q, target_link)
                R_cur = T_cur[:3, :3]
                p_cur = T_cur[:3, 3]
                pos_err_v = p_target - p_cur
                
                # Use transform API's rotation_error
                ori_err_v = rotation_error(R_cur, R_target)
                # Ensure consistent dtype and device (transform may return different)
                ori_err_v = ori_err_v.to(device=self.device, dtype=self.dtype)
                
                pos_err_norm_t = torch.linalg.norm(pos_err_v)  # 保持为tensor
                ori_err_norm_t = torch.linalg.norm(ori_err_v)  # 保持为tensor

                # 与 NumPy 一致：不使用动态姿态权重调整
                full_err = torch.cat([pos_weight * pos_err_v, ori_weight * ori_err_v])
                if mask_bool is not None:
                    err = full_err[mask_bool]
                else:
                    err = full_err
                err_norm = torch.linalg.norm(err)

                # 更新最优解 - 使用tensor比较
                if err_norm < best_err:
                    best_err = err_norm
                    best_q = q.clone()
                    final_pos_err = pos_err_norm_t.item()
                    final_ori_err = ori_err_norm_t.item()

                if prev_err_norm - err_norm < 1e-8:
                    plateau_counter += 1
                else:
                    plateau_counter = 0
                prev_err_norm = err_norm

                # 收敛检查 - 只在这里调用.item()
                pos_err_norm = pos_err_norm_t.item()
                ori_err_norm = ori_err_norm_t.item()

                if pos_err_norm < self.pos_tol and ori_err_norm < self.ori_tol:
                    if refine:
                        r_pos_tol = refine_pos_tol or (self.pos_tol * 0.2)
                        r_ori_tol = refine_ori_tol or (self.ori_tol * 0.2)
                        q_ref = q.clone()
                        for _ in range(refine_iters):
                            set_backend("torch", device=str(self.device), dtype=self.dtype)
                            T_r = self.fk_solver.solve(q_ref, return_end_only=True, device=self.device, dtype=self.dtype)["end"]
                            p_r = T_r[:3, 3]; R_r = T_r[:3, :3]
                            p_e = p_target - p_r
                            o_e = rotation_error(R_r, R_target).to(device=self.device, dtype=self.dtype)
                            p_e_norm = torch.linalg.norm(p_e)
                            o_e_norm = torch.linalg.norm(o_e)
                            if p_e_norm < r_pos_tol and o_e_norm < r_ori_tol:
                                q = q_ref
                                final_pos_err = p_e_norm.item()
                                final_ori_err = o_e_norm.item()
                                best_q = q.clone()
                                best_err = torch.sqrt(p_e_norm**2 + o_e_norm**2)
                                break
                            J_ref = self.jacobian_solver.solve(
                                q_ref,
                                method="analytic",
                                device=self.device,
                                dtype=self.dtype
                            )
                            if pos_weight != 1.0:
                                J_ref[:3, :] *= pos_weight
                            if ori_weight != 1.0:
                                J_ref[3:6, :] *= ori_weight
                            dq_ref = self._solve_pinv(J_ref, torch.cat([pos_weight * p_e, ori_weight * o_e]), self.min_damping)
                            dq_norm_ref = torch.linalg.norm(dq_ref)
                            if dq_norm_ref > 0.2:
                                dq_ref = dq_ref * (0.2 / (dq_norm_ref + 1e-15))
                            q_ref = self._apply_joint_limits(q_ref + dq_ref)
                        q = q_ref
                    return {
                        "q": q.detach().cpu().tolist(),
                        "success": True,
                        "iters": it,
                        "err_norm": float(best_err.item() if torch.is_tensor(best_err) else best_err),
                        "pos_err": float(final_pos_err),
                        "ori_err": float(final_ori_err),
                        "method": method,
                        "jacobian": jac_type_local,
                    }

                # Jacobian using solver
                if use_numeric_jacobian:
                    J_full = self.jacobian_solver.solve(
                        q,
                        method="numeric",
                        use_central_diff=use_central_diff,
                        device=self.device,
                        dtype=self.dtype
                    )
                    jac_type_local = "numeric_central" if use_central_diff else "numeric_forward"
                    J = J_full
                else:
                    J_full = self.jacobian_solver.solve(
                        q,
                        method="analytic",
                        device=self.device,
                        dtype=self.dtype,
                        target_link=target_link,
                    )
                    jac_type_local = "analytic"
                    J = J_full
                if pos_weight != 1.0:
                    J[:3, :] *= pos_weight
                if ori_weight != 1.0:
                    J[3:6, :] *= ori_weight
                if mask_bool is not None:
                    J_eff = J[mask_bool, :]
                else:
                    J_eff = J

                # 阻尼
                if adaptive_damping:
                    damping = self._compute_adaptive_damping(J_eff, pos_err_norm, ori_err_norm)
                else:
                    damping = 0.5 * (self.min_damping + self.max_damping)
                if plateau_counter >= 4:
                    damping = max(damping * 2.0, self.max_damping)
                if plateau_counter >= 8:
                    # 进一步加大阻尼并略微减小步长
                    damping = max(damping * 1.5, self.max_damping * 2.0)
                # 解
                if method == "dls":
                    dq = self._solve_dls(J_eff, err, damping)
                elif method == "pinv":
                    dq = self._solve_pinv(J_eff, err, damping)
                else:
                    # Jacobian Transpose 方法
                    # 使用自适应增益：alpha = ||err||² / ||J @ J.T @ err||²
                    if transpose_gain is not None:
                        alpha = transpose_gain
                    else:
                        # 自适应增益计算（更稳定的收敛）
                        J_err = J.transpose(0, 1) @ err
                        JJt_err = J @ J_err
                        err_norm_sq = torch.dot(err, err)
                        JJt_err_norm_sq = torch.dot(JJt_err, JJt_err)
                        if JJt_err_norm_sq > 1e-12:
                            alpha_raw = (err_norm_sq / JJt_err_norm_sq).item()
                        else:
                            # 回退到固定增益
                            alpha_raw = 0.01
                        # 限制 alpha 范围避免步长过大
                        alpha = max(0.001, min(alpha_raw, 0.5))
                    dq = alpha * (J_eff.transpose(0, 1) @ err)

                # Nullspace redundancy (only if n > task_rows)
                if nullspace_gain > 0 and self.n > J_eff.shape[0]:
                    try:
                        U_ns, S_ns, Vt_ns = torch.linalg.svd(J_eff, full_matrices=False)
                        S_inv_ns = torch.where(S_ns > 1e-9, 1.0 / S_ns, torch.zeros_like(S_ns))
                        J_pinv_eff = (Vt_ns.transpose(0, 1) * S_inv_ns) @ U_ns.transpose(0, 1)
                        N = torch.eye(self.n, dtype=self.dtype, device=self.device) - J_pinv_eff @ J_eff
                        if joint_centering:
                            centers = []
                            for js in self.model._actuated:  # type: ignore[attr-defined]
                                lo, hi = -1.0, 1.0
                                if js.limit:
                                    if js.limit[0] is not None:
                                        lo = js.limit[0]
                                    if js.limit[1] is not None:
                                        hi = js.limit[1]
                                centers.append(0.5 * (lo + hi))
                            centers_t = torch.tensor(centers, dtype=self.dtype, device=self.device)
                            delta_center = centers_t - q
                            if joint_center_weights is not None and len(joint_center_weights) == self.n:
                                w = torch.tensor(joint_center_weights, dtype=self.dtype, device=self.device)
                                delta_center = delta_center * w
                            dq_sec = joint_center_gain * delta_center
                        else:
                            dq_sec = torch.zeros(self.n, dtype=self.dtype, device=self.device)
                        dq = dq + nullspace_gain * (N @ dq_sec)
                    except Exception:
                        pass

                # 步长 (transpose 方法的 alpha 已经是最优步长，不需要额外缩放)
                if method != "transpose" and adaptive_step:
                    step = self._compute_adaptive_step(pos_err_norm, ori_err_norm)
                else:
                    step = 1.0 if method == "transpose" else self.base_step

                if plateau_counter >= 8 and method != "transpose":
                    step *= 0.5

                dq_step = step * dq
                dq_norm = torch.linalg.norm(dq_step)
                if dq_norm > max_step_norm:
                    dq_step = dq_step * (max_step_norm / (dq_norm + 1e-15))
                new_q = self._apply_joint_limits(q + dq_step)

                if backtrack:
                    prev_total = err_norm
                    for _bt in range(3):
                        set_backend("torch", device=str(self.device), dtype=self.dtype)
                        T_bt = self.fk_solver.solve(new_q, return_end_only=True, device=self.device, dtype=self.dtype)["end"]
                        p_bt = T_bt[:3, 3]; R_bt = T_bt[:3, :3]
                        pos_bt = torch.linalg.norm(p_target - p_bt).item()
                        ori_err = rotation_error(R_bt, R_target).to(device=self.device, dtype=self.dtype)
                        ori_bt = torch.linalg.norm(ori_err).item()
                        total_bt = pos_bt + ori_bt
                        if total_bt <= prev_total:
                            break
                        dq_step = dq_step * 0.5
                        new_q = self._apply_joint_limits(q + dq_step)
                    q = new_q
                else:
                    q = new_q
            # 未在迭代内收敛
            return {
                "q": best_q.detach().cpu().tolist(),
                "success": False,
                "iters": self.max_iters,
                "err_norm": float(best_err.item() if torch.is_tensor(best_err) else best_err),
                "method": method,
                "jacobian": jac_type_local,
                "pos_err": float(final_pos_err),
                "ori_err": float(final_ori_err),
            }

        # 主尝试 + 重启
        attempt_results.append(run_one(base_q0))
        if restarts > 0:
            for _ in range(restarts):
                noise = torch.randn_like(base_q0) * restart_noise
                q_init = self._apply_joint_limits(base_q0 + noise)
                attempt_results.append(run_one(q_init))

        # 优先返回成功里误差最小，其次返回总体误差最小
        success_runs = [r for r in attempt_results if r.get('success')]
        if success_runs:
            success_runs.sort(key=lambda r: r['err_norm'])
            return success_runs[0]
        attempt_results.sort(key=lambda r: r['err_norm'])
        return attempt_results[0]

    # -------------------- helpers --------------------
    def _solve_dls(self, J: Tensor, err: Tensor, damping: float) -> Tensor:
        A = J @ J.transpose(0, 1) + (damping ** 2) * self._eye6
        try:
            y = torch.linalg.solve(A, err)
        except Exception:
            # CPU fallback (e.g., MPS not supporting op)
            y = torch.linalg.solve(A.cpu(), err.cpu()).to(device=J.device)
        return J.transpose(0, 1) @ y

    def _solve_pinv(self, J: Tensor, err: Tensor, damping: float) -> Tensor:
        # 统一实现（cpu / cuda）
        try:
            U, S, Vh = torch.linalg.svd(J, full_matrices=False)
        except RuntimeError:
            # 其他设备的 CPU 回退
            try:
                Uc, Sc, Vhc = torch.linalg.svd(J.cpu(), full_matrices=False)
                U, S, Vh = Uc.to(J.device), Sc.to(J.device), Vhc.to(J.device)
            except Exception:
                return self._solve_dls(J, err, damping)

        if damping > 0:
            S_inv = S / (S * S + damping * damping)
        else:
            tol = 1e-9 * max(J.shape)
            S_inv = torch.where(S > tol, 1.0 / S, torch.zeros_like(S))
        return (Vh.transpose(0, 1) * S_inv) @ (U.transpose(0, 1) @ err)

    def _compute_adaptive_damping(self, J: Tensor, pos_err: float, ori_err: float) -> float:
        # 使用 SVD 计算条件数
        try:
            S = torch.linalg.svdvals(J)
            s_max = S[0].item()
            s_min = S[-1].item()
            cond = s_max / max(s_min, 1e-12)
        except Exception:
            # 完全回退
            cond = 100.0

        err_combo = pos_err + 0.5 * ori_err
        if cond > 200 or err_combo > 0.05:
            return self.max_damping
        elif cond < 30 and err_combo < 0.01:
            return self.min_damping
        else:
            return 0.5 * (self.min_damping + self.max_damping)

    def _compute_adaptive_step(self, pos_err: float, ori_err: float) -> float:
        norm_pos = pos_err / 0.01
        norm_ori = ori_err / 0.087  # ~5°
        m = max(norm_pos, norm_ori)
        if m > 2.0:
            return self.base_step * 0.6
        elif m > 1.0:
            return self.base_step
        elif m > 0.5:
            return self.base_step * 1.2
        else:
            return self.base_step * 0.6

    def _apply_joint_limits(self, q: Tensor) -> Tensor:
        out = q.clone()
        for js in self.model._actuated:  # type: ignore[attr-defined]
            if js.limit is not None:
                lo, hi = js.limit
                if lo is not None:
                    out[js.index] = torch.clamp(out[js.index], min=float(lo))
                if hi is not None:
                    out[js.index] = torch.clamp(out[js.index], max=float(hi))
        return out

    # ==================== Partial FK (single) ====================
    def _fk_until_torch(self, q: Tensor, target_link: str) -> Tensor:
        """Compute 4x4 pose of an intermediate link (target_link).

        Mirrors the early-stop traversal logic used in the Torch Jacobian solver
        (analytic path) to ensure consistent frames and axis extraction.
        """
        if not torch.is_tensor(q):
            q = torch.tensor(q, dtype=self.dtype, device=self.device)
        q_map = {js.name: q[js.index] for js in self.model._actuated}  # type: ignore[attr-defined]
        T_parent = torch.eye(4, dtype=self.dtype, device=self.device)
        for urdf_joint in self.model._chain_joints:  # type: ignore[attr-defined]
            R_origin = self.jacobian_solver._rpy_matrix_torch(
                torch.tensor(urdf_joint.origin_rpy[0], dtype=self.dtype, device=self.device),
                torch.tensor(urdf_joint.origin_rpy[1], dtype=self.dtype, device=self.device),
                torch.tensor(urdf_joint.origin_rpy[2], dtype=self.dtype, device=self.device),
            )
            t_origin = torch.tensor(urdf_joint.origin_xyz, dtype=self.dtype, device=self.device)
            T_origin = torch.eye(4, dtype=self.dtype, device=self.device)
            T_origin[:3, :3] = R_origin
            T_origin[:3, 3] = t_origin
            T_joint_origin = T_parent @ T_origin
            R_motion = torch.eye(3, dtype=self.dtype, device=self.device)
            t_motion = torch.zeros(3, dtype=self.dtype, device=self.device)
            if urdf_joint.joint_type == "revolute":
                theta = q_map.get(urdf_joint.name, torch.tensor(0.0, dtype=self.dtype, device=self.device))
                R_motion = self.jacobian_solver._axis_rotation_torch(
                    torch.tensor(urdf_joint.axis, dtype=self.dtype, device=self.device), theta
                )
            elif urdf_joint.joint_type == "prismatic":
                d = q_map.get(urdf_joint.name, torch.tensor(0.0, dtype=self.dtype, device=self.device))
                t_motion = self.jacobian_solver._axis_translation_torch(
                    torch.tensor(urdf_joint.axis, dtype=self.dtype, device=self.device), d
                )
            T_motion = torch.eye(4, dtype=self.dtype, device=self.device)
            T_motion[:3, :3] = R_motion
            T_motion[:3, 3] = t_motion
            T_child = T_joint_origin @ T_motion
            T_parent = T_child
            if urdf_joint.child == target_link:
                return T_child
        raise ValueError(f"target_link '{target_link}' not found in kinematic chain")
    
    # ==================== Batch Mode IK ====================
    
    def _solve_batch(
        self,
        target_poses_batch: Tensor,
        q_init_batch: Tensor,
        *,
        method: str = "dls",
        pos_weight: float = 1.0,
        ori_weight: float = 1.0,
        max_step_norm: float = 0.5,  # 与 NumPy 一致
        verbose: bool = False,
        damping: float | None = None,
    ) -> Dict:
        """Vectorized batch IK using Damped Least Squares (TRUE BATCH).

        The entire batch advances per-iteration without Python per-sample loops.

        :param target_poses_batch: [B,4,4]
        :param q_init_batch: [B,n]
        :param method: only 'dls' supported
        :param pos_weight: position error weight
        :param ori_weight: orientation error weight
        :param max_step_norm: max ||dq|| per-iteration (per sample)
        :param verbose: print brief convergence stats
        :param damping: override damping (λ). If None auto = geometric mean(min,max)
        :return: dict with fields: q, success, iterations, method, pos_err, ori_err
        """
        if method != "dls":
            raise NotImplementedError("Batch mode currently supports only 'dls'")

        B, n = q_init_batch.shape
        if n != self.n:
            raise ValueError(f"q_init_batch n={n} != model dof {self.n}")
        if target_poses_batch.shape != (B, 4, 4):
            raise ValueError("target_poses_batch must be [B,4,4]")

        q = q_init_batch.clone()
        success = torch.zeros(B, dtype=torch.bool, device=self.device)
        iters = torch.zeros(B, dtype=torch.int32, device=self.device)
        active = torch.ones(B, dtype=torch.bool, device=self.device)
        # Plateau & error tracking
        prev_err_norm = torch.full((B,), float('inf'), dtype=self.dtype, device=self.device)
        plateau_counter = torch.zeros(B, dtype=torch.int32, device=self.device)

        # Targets
        p_target = target_poses_batch[:, :3, 3]
        R_target = target_poses_batch[:, :3, :3]

        # Base damping (will be adapted per-sample)
        if damping is None:
            base_lam = math.sqrt(self.min_damping * self.max_damping)
        else:
            base_lam = float(damping)

        eye6 = torch.eye(6, device=self.device, dtype=self.dtype).unsqueeze(0)  # [1,6,6]

        # Per-iteration loop
        for it in range(1, self.max_iters + 1):
            if not active.any():
                break

            act_idx = torch.where(active)[0]
            q_act = q[act_idx]

            # FK cache (vectorized)
            p_end, R_end, p_joint, z_axis = self._batch_fk_cache(q_act)

            # Errors (position in world frame, orientation axis-angle in end-effector frame)
            p_err = p_target[act_idx] - p_end  # [Ba,3]
            # match single-mode: rotation_error(R_cur, R_target)
            R_err_vec = self._batch_rotation_error(R_end, R_target[act_idx])  # [Ba,3]  (cur, target)

            # Weighted error vector e: [Ba,6]
            # Dynamic orientation weight scaling per sample (replicate single-mode heuristic)
            ori_norms = torch.linalg.norm(R_err_vec, dim=1)
            # thresholds: >1.0 ->0.3, >0.7->0.5, >0.4->0.8 else 1.0
            ori_scale = torch.ones_like(ori_norms)
            ori_scale = torch.where(ori_norms > 1.0, torch.full_like(ori_scale, 0.3), ori_scale)
            ori_scale = torch.where((ori_norms <= 1.0) & (ori_norms > 0.7), torch.full_like(ori_scale, 0.5), ori_scale)
            ori_scale = torch.where((ori_norms <= 0.7) & (ori_norms > 0.4), torch.full_like(ori_scale, 0.8), ori_scale)
            ori_weight_dyn = ori_weight * ori_scale  # per-sample
            e = torch.cat([pos_weight * p_err, (ori_weight_dyn.unsqueeze(1) * R_err_vec)], dim=1)

            # Jacobian build (analytic, vectorized) - 构建世界坐标系 Jacobian
            # J_lin[:,j] = z_j x (p_end - p_j); J_ang[:,j] = z_j (revolute)
            Ba = q_act.shape[0]
            J_geo = torch.zeros(Ba, 6, n, device=self.device, dtype=self.dtype)
            p_end_exp = p_end.unsqueeze(1)  # [Ba,1,3]
            for js in self.model._actuated:  # type: ignore[attr-defined]
                j = js.index
                z = z_axis[:, j, :]  # [Ba,3]
                p_j = p_joint[:, j, :]  # [Ba,3]
                if js.joint_type == 'revolute':
                    J_geo[:, 0:3, j] = torch.cross(z, (p_end_exp[:, 0, :] - p_j), dim=1)
                    J_geo[:, 3:6, j] = z  # world frame angular
                elif js.joint_type == 'prismatic':
                    J_geo[:, 0:3, j] = z
                    # angular part zero
                else:  # fixed
                    pass

            # Transform angular rows into end-effector frame (match single-mode Jacobian convention)
            # J_ang_body = R_end^T * J_ang_world
            R_end_T = R_end.transpose(1, 2)  # [Ba,3,3]
            J = J_geo.clone()
            J[:, 3:6, :] = torch.bmm(R_end_T, J_geo[:, 3:6, :])  # [Ba,3,n]

            # Apply position & dynamic orientation weights to Jacobian rows
            if pos_weight != 1.0:
                J[:, 0:3, :] *= pos_weight
            # Broadcast ori_weight_dyn per sample
            J[:, 3:6, :] *= ori_weight_dyn.view(Ba, 1, 1)

            # Norms & convergence (pre-update)
            pos_norm = torch.linalg.norm(p_err, dim=1)
            ori_norm = torch.linalg.norm(R_err_vec, dim=1)
            conv_mask = (pos_norm < self.pos_tol) & (ori_norm < self.ori_tol)

            # Error norm for plateau detection (use weighted error vector)
            err_norm_act = torch.linalg.norm(e.view(e.shape[0], -1), dim=1)
            act_idx_full = act_idx  # preserve current mapping
            # Update plateau counters for all active samples BEFORE removing converged ones
            delta = prev_err_norm[act_idx_full] - err_norm_act
            plateau_inc = delta < 1e-8
            # Reset where improvement
            plateau_counter[act_idx_full] = torch.where(
                plateau_inc,
                plateau_counter[act_idx_full] + 1,
                torch.zeros_like(plateau_counter[act_idx_full])
            )
            prev_err_norm[act_idx_full] = err_norm_act
            if conv_mask.any():
                g_idx = act_idx[conv_mask]
                success[g_idx] = True
                iters[g_idx] = it
                active[g_idx] = False

            # Remove converged from further computation (filter current active subset)
            if (~active).all():
                break
            mask_keep = ~conv_mask  # among current active set (before filtering)
            # Filter tensors to active subset (those not converged in this iteration)
            J = J[mask_keep]
            e = e[mask_keep]
            # Slice auxiliary tensors accordingly
            p_err_active = p_err[mask_keep]
            ori_norms_active = ori_norms[mask_keep]
            ori_weight_dyn_active = ori_weight_dyn[mask_keep]
            plateau_sub = plateau_counter[act_idx_full[mask_keep]]

            # IMPORTANT: q_sub should be taken BEFORE updating act_idx!
            q_sub = q[act_idx_full[mask_keep]]

            # Now update act_idx to point to still-active samples globally
            act_idx = act_idx_full[mask_keep]
            if act_idx.numel() == 0:
                break

            # Adaptive damping per-sample using batch SVD (much faster on GPU)
            try:
                # torch.linalg.svd on [Ba,6,n] -> U:[Ba,6,6], S:[Ba,6], Vh:[Ba,n,n] (full=False) when n>=6
                # If n < 6, shape adjusts; we only need singular values.
                S_all = torch.linalg.svdvals(J)  # [Ba', min(6,n)]
                s_max = S_all[:, 0]
                s_min = S_all[:, -1].clamp(min=1e-12)
                conds = s_max / s_min
            except RuntimeError:
                # Fallback: approximate using Fro norm ratio (cheap)
                # cond ≈ ||J||_F * ||pseudo-inverse||_F (approx). Here we simplify to scale by row norms.
                fro = torch.linalg.norm(J, dim=(1, 2))
                # Use min singular proxy ≈ fro / (sqrt(6)*max_col_norm) -> coarse
                col_norms = torch.linalg.norm(J, dim=1)  # [Ba', n]
                max_col = col_norms.max(dim=1).values.clamp(min=1e-6)
                approx_s_min = fro / (math.sqrt(6.0)*max_col)
                conds = fro / approx_s_min.clamp(min=1e-12)
            pos_norm_sub = torch.linalg.norm(p_err_active, dim=1)
            ori_norm_sub = ori_norms_active
            err_combo = pos_norm_sub + 0.5 * ori_norm_sub
            lam_vec = torch.full_like(conds, base_lam)
            high_mask = (conds > 200) | (err_combo > 5e-2)
            low_mask = (conds < 30) & (err_combo < 1e-2)
            lam_vec[high_mask] = self.max_damping
            lam_vec[low_mask] = self.min_damping

            # Plateau-based damping escalation
            lam_vec = torch.where(plateau_sub >= 4, torch.clamp(lam_vec * 2.0, max=self.max_damping*2), lam_vec)
            lam_vec = torch.where(plateau_sub >= 8, torch.clamp(lam_vec * 1.5, max=self.max_damping*4), lam_vec)

            # DLS solve: per-sample damping
            JJt = torch.bmm(J, J.transpose(1, 2))  # [Ba',6,6]
            lam_sq = (lam_vec ** 2).view(-1, 1, 1)
            JJt_damped = JJt + lam_sq * eye6
            try:
                y = torch.linalg.solve(JJt_damped, e.unsqueeze(2)).squeeze(2)  # [Ba',6]
            except RuntimeError:
                # fallback per-sample solve
                y = torch.zeros_like(e)
                for k in range(J.shape[0]):
                    try:
                        y[k] = torch.linalg.solve(JJt_damped[k], e[k])
                    except RuntimeError:
                        Jk = J[k]
                        alt = (Jk @ Jk.transpose(0, 1) + (lam_vec[k]**2)*torch.eye(6,
                               device=self.device, dtype=self.dtype)).pinverse() @ e[k]
                        y[k] = alt
            dq = torch.bmm(J.transpose(1, 2), y.unsqueeze(2)).squeeze(2)  # [Ba',n]

            # Adaptive step (vectorized similar to single-mode)
            norm_pos = pos_norm_sub / 0.01
            norm_ori = ori_norm_sub / 0.087
            m = torch.maximum(norm_pos, norm_ori)
            step = torch.full_like(m, self.base_step)
            step = torch.where(m > 2.0, self.base_step * 0.6, step)
            step = torch.where((m <= 2.0) & (m > 1.0), self.base_step * 1.0, step)
            step = torch.where((m <= 1.0) & (m > 0.5), self.base_step * 1.2, step)
            step = torch.where(m <= 0.5, self.base_step * 0.6, step)
            # Plateau slowdown
            step = torch.where(plateau_sub >= 8, step * 0.5, step)
            dq = dq * step.unsqueeze(1)

            # Step norm clipping
            dq_norm = torch.linalg.norm(dq, dim=1, keepdim=True)
            scale = torch.clamp(max_step_norm / (dq_norm + 1e-12), max=1.0)
            dq = dq * scale

            # Joint limit enforcement
            q_new = q_sub + dq
            # apply joint limits vectorized
            for js in self.model._actuated:  # type: ignore[attr-defined]
                if js.limit is not None:
                    lo, hi = js.limit
                    j = js.index
                    if lo is not None:
                        q_new[:, j] = torch.clamp(q_new[:, j], min=float(lo))
                    if hi is not None:
                        q_new[:, j] = torch.clamp(q_new[:, j], max=float(hi))
            q[act_idx] = q_new

        # For samples never converged, record iterations
        remaining = active.nonzero(as_tuple=False).flatten()
        if remaining.numel() > 0:
            iters[remaining] = self.max_iters

        # Final errors (for reporting)
        p_end_all, R_end_all, _, _ = self._batch_fk_cache(q)
        p_err_final = torch.linalg.norm(p_target - p_end_all, dim=1)
        ori_err_final = torch.linalg.norm(self._batch_rotation_error(R_end_all, R_target), dim=1)

        if verbose:
            sr = success.float().mean().item() * 100.0
            print(f"Batch IK DLS success={sr:.1f}% avg_iter={(iters[success].float().mean().item() if success.any() else 0):.1f}")

        return {
            "q": q,
            "success": success,
            "iterations": iters,
            "method": method,
            "pos_err": p_err_final,
            "ori_err": ori_err_final,
        }

    # --------------------------------------------------------------------- #
    # Batch FK cache for Jacobian construction
    # --------------------------------------------------------------------- #
    def _batch_fk_cache(self, q_batch: Tensor):
        """Compute per-joint world origins and axes for a batch.

        :param q_batch: [B,n]
        :return: (p_end [B,3], R_end [B,3,3], p_joint [B,n,3], z_axis [B,n,3])
        """
        B = q_batch.shape[0]
        device = q_batch.device
        dtype = q_batch.dtype
        # Storage
        p_joint = torch.zeros(B, self.n, 3, device=device, dtype=dtype)
        z_axis = torch.zeros(B, self.n, 3, device=device, dtype=dtype)

        # Running transform T_parent = [R|t; 0|1] per sample (track as R, t separately)
        R_parent = torch.eye(3, device=device, dtype=dtype).unsqueeze(0).repeat(B, 1, 1)  # [B,3,3]
        t_parent = torch.zeros(B, 3, device=device, dtype=dtype)  # [B,3]

        # Build q_map for joint values (map joint name -> [B] tensor of values)
        q_map = {js.name: q_batch[:, js.index] for js in self.model._actuated}  # type: ignore[attr-defined]

        # CRITICAL: iterate over full chain (including fixed joints) like single-mode does
        for urdf_joint in self.model._chain_joints:  # type: ignore[attr-defined]
            # Origin rotation & translation (fixed per joint, broadcast to batch)
            rpy_vals = urdf_joint.origin_rpy  # (roll, pitch, yaw)
            r = torch.full((B,), float(rpy_vals[0]), device=device, dtype=dtype)
            p = torch.full((B,), float(rpy_vals[1]), device=device, dtype=dtype)
            y = torch.full((B,), float(rpy_vals[2]), device=device, dtype=dtype)
            cr, sr = torch.cos(r), torch.sin(r)
            cp, sp = torch.cos(p), torch.sin(p)
            cy, sy = torch.cos(y), torch.sin(y)
            # R = Rz(yaw) @ Ry(pitch) @ Rx(roll) (ZYX Euler)
            R_origin = torch.zeros(B, 3, 3, device=device, dtype=dtype)
            R_origin[:, 0, 0] = cy*cp
            R_origin[:, 0, 1] = cy*sp*sr - sy*cr
            R_origin[:, 0, 2] = cy*sp*cr + sy*sr
            R_origin[:, 1, 0] = sy*cp
            R_origin[:, 1, 1] = sy*sp*sr + cy*cr
            R_origin[:, 1, 2] = sy*sp*cr - cy*sr
            R_origin[:, 2, 0] = -sp
            R_origin[:, 2, 1] = cp*sr
            R_origin[:, 2, 2] = cp*cr
            t_origin = torch.tensor(urdf_joint.origin_xyz, device=device,
                                    dtype=dtype).unsqueeze(0).expand(B, 3)  # [B,3]

            # T_joint_origin = T_parent @ T_origin
            R_joint_origin = torch.bmm(R_parent, R_origin)  # [B,3,3]
            t_joint_origin = t_parent + torch.bmm(R_parent, t_origin.unsqueeze(2)).squeeze(2)  # [B,3]

            # For actuated joints: cache axis and origin position
            if urdf_joint.joint_type in ("revolute", "prismatic"):
                # Find corresponding JointSpec to get index
                js = next((j for j in self.model._actuated if j.name ==
                          urdf_joint.name), None)  # type: ignore[attr-defined]
                if js is not None:
                    j = js.index
                    # Axis in world frame
                    axis_local = torch.tensor(urdf_joint.axis, device=device,
                                              dtype=dtype).unsqueeze(0).repeat(B, 1)  # [B,3]
                    # Normalize axis
                    axis_norm = torch.linalg.norm(axis_local, dim=1, keepdim=True).clamp(min=1e-10)
                    axis_local = axis_local / axis_norm
                    z_world = torch.bmm(R_joint_origin, axis_local.unsqueeze(2)).squeeze(2)  # [B,3]

                    # Cache position and axis
                    p_joint[:, j, :] = t_joint_origin
                    z_axis[:, j, :] = z_world

            # Motion transform (depends on joint type and value)
            R_motion = torch.eye(3, device=device, dtype=dtype).unsqueeze(0).repeat(B, 1, 1)  # [B,3,3]
            t_motion = torch.zeros(B, 3, device=device, dtype=dtype)  # [B,3]

            if urdf_joint.joint_type == "revolute":
                # Get joint value (theta) from q_map
                theta = q_map.get(urdf_joint.name, torch.zeros(B, device=device, dtype=dtype))
                # Rodrigues rotation around axis
                axis_local = torch.tensor(urdf_joint.axis, device=device, dtype=dtype).unsqueeze(0).repeat(B, 1)
                axis_norm = torch.linalg.norm(axis_local, dim=1, keepdim=True).clamp(min=1e-10)
                k = axis_local / axis_norm
                # Skew-symmetric matrix K
                K = torch.zeros(B,3,3, device=device, dtype=dtype)
                K[:, 0, 1] = -k[:, 2]
                K[:, 0, 2] = k[:, 1]
                K[:, 1, 0] = k[:, 2]
                K[:, 1, 2] = -k[:, 0]
                K[:, 2, 0] = -k[:, 1]
                K[:, 2, 1] = k[:, 0]
                K2 = torch.bmm(K, K)
                sin_t = torch.sin(theta).view(B,1,1)
                cos_t = torch.cos(theta).view(B,1,1)
                I = torch.eye(3, device=device, dtype=dtype).unsqueeze(0)
                R_motion = I + sin_t * K + (1 - cos_t) * K2  # [B,3,3]
            elif urdf_joint.joint_type == "prismatic":
                # Get joint value (d) from q_map
                d = q_map.get(urdf_joint.name, torch.zeros(B, device=device, dtype=dtype))
                # Translation along axis
                axis_local = torch.tensor(urdf_joint.axis, device=device, dtype=dtype).unsqueeze(0).repeat(B, 1)
                axis_norm = torch.linalg.norm(axis_local, dim=1, keepdim=True).clamp(min=1e-10)
                axis_local = axis_local / axis_norm
                t_motion = axis_local * d.unsqueeze(1)  # [B,3]
            # else: fixed joint, motion is identity

            # T_child = T_joint_origin @ T_motion
            R_child = torch.bmm(R_joint_origin, R_motion)  # [B,3,3]
            t_child = t_joint_origin + torch.bmm(R_joint_origin, t_motion.unsqueeze(2)).squeeze(2)  # [B,3]

            # Update parent for next iteration
            R_parent = R_child
            t_parent = t_child

        p_end = t_parent
        R_end = R_parent
        return p_end, R_end, p_joint, z_axis
    
    @staticmethod
    def _batch_rotation_error(R_current: Tensor, R_target: Tensor) -> Tensor:
        """Compute rotation error in angle-axis representation for batch.
        
        Error = log(R_current^T @ R_target) converted to axis-angle (match single-sample rotation_error)
        
        :param R_current: [B, 3, 3] current rotation matrices
        :param R_target: [B, 3, 3] target rotation matrices
        :return: [B, 3] rotation error vectors
        """
        # R_error = R_current^T @ R_target
        R_error = torch.bmm(R_current.transpose(1, 2), R_target)
        
        # Convert to angle-axis
        return IKSolverTorch._batch_rotation_matrix_to_axis_angle(R_error)
    
    @staticmethod
    def _batch_rotation_matrix_to_axis_angle(R: Tensor) -> Tensor:
        """Convert batch of rotation matrices to axis-angle representation.
        
        :param R: [B, 3, 3] rotation matrices
        :return: [B, 3] axis-angle vectors
        """
        batch_size = R.shape[0]
        device = R.device
        dtype = R.dtype
        
        # Angle from trace: cos(θ) = (trace(R) - 1) / 2
        trace = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]
        angle = torch.acos(torch.clamp((trace - 1) / 2, -1.0, 1.0))  # [B]
        
        # Axis from skew-symmetric part
        axis = torch.zeros(batch_size, 3, device=device, dtype=dtype)
        
        # Handle small angles (θ ≈ 0) - use linear approximation
        small_angle = angle < 1e-6
        if small_angle.any():
            # For small angles: axis-angle ≈ [R[2,1]-R[1,2], R[0,2]-R[2,0], R[1,0]-R[0,1]] / 2
            axis[small_angle, 0] = (R[small_angle, 2, 1] - R[small_angle, 1, 2]) / 2
            axis[small_angle, 1] = (R[small_angle, 0, 2] - R[small_angle, 2, 0]) / 2
            axis[small_angle, 2] = (R[small_angle, 1, 0] - R[small_angle, 0, 1]) / 2
        
        # Handle normal angles
        normal_angle = ~small_angle
        if normal_angle.any():
            # axis = [R[2,1]-R[1,2], R[0,2]-R[2,0], R[1,0]-R[0,1]] / (2*sin(θ))
            sin_angle = torch.sin(angle[normal_angle])
            axis[normal_angle, 0] = (R[normal_angle, 2, 1] - R[normal_angle, 1, 2]) / (2 * sin_angle)
            axis[normal_angle, 1] = (R[normal_angle, 0, 2] - R[normal_angle, 2, 0]) / (2 * sin_angle)
            axis[normal_angle, 2] = (R[normal_angle, 1, 0] - R[normal_angle, 0, 1]) / (2 * sin_angle)
        
        # Axis-angle = angle * axis
        axis_angle = axis * angle.unsqueeze(1)
        
        return axis_angle
