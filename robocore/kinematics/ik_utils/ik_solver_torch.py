"""PyTorch 版本逆运动学求解器 (IKSolverTorch)。

Copyright (c) 2025 Synria Robotics Co., Ltd.

Licensed under the MIT License.

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
from robocore.transform import rotation_error, rpy_to_matrix, axis_angle_to_matrix
from robocore.kinematics.utils import ensure_batch, restore_single
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
        max_iters: int = 200,  # Increased from 100 for better convergence (88.5% success rate at n=200)
        pos_tol: float = 1e-3,  # Relaxed from 1e-4 for better success rate
        ori_tol: float = 1e-3,  # Kept at 1e-3 (good balance)
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
        # 关节数量
        # NumPy 版使用 model._chain_dof_list / model.num_chain_dof，这里保持一致
        if hasattr(model, "num_chain_dof"):
            self.n = int(model.num_chain_dof)
        elif hasattr(model, "_chain_dof_list"):
            self.n = len(model._chain_dof_list)  # type: ignore[attr-defined]
        else:
            raise AttributeError(
                "RobotModel instance missing 'num_chain_dof' / '_chain_dof_list'; "
                "expected recent RobotModel implementation."
            )
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
        # For API compatibility with NumPy solver; when not None, it overrides
        # use_numeric_jacobian below (use_analytic_jacobian=True -> use_numeric_jacobian=False)
        use_analytic_jacobian: Optional[bool] = None,
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
        **_: object,
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
        
        # Map NumPy-style flag if provided
        if use_analytic_jacobian is not None:
            # NumPy: use_analytic_jacobian=True → analytic; here that means NOT numeric
            use_numeric_jacobian = not bool(use_analytic_jacobian)
        
        # Handle batch mode - detect before ensure_batch
        target_ndim_orig = target_pose.ndim
        q0_ndim_orig = q0.ndim
        
        # Normalize to batch format for processing
        if target_pose.ndim == 2:
            target_pose = target_pose.unsqueeze(0)  # [4, 4] -> [1, 4, 4]
        if q0.ndim == 1:
            q0 = q0.unsqueeze(0)  # [n] -> [1, n]

        # Track if original was single
        was_single = (target_ndim_orig == 2) and (q0_ndim_orig == 1)

        if target_pose.shape[0] != q0.shape[0]:
            raise ValueError(f"Batch size mismatch: target_pose {target_pose.shape[0]} vs q0 {q0.shape[0]}")

        # Common kwargs for _solve_single
        single_kwargs = dict(
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
            random_seed=random_seed,
        )

        if was_single:
            # Single mode
            return self._solve_single(target_pose[0], q0[0], **single_kwargs)

        # Batch mode - use vectorized batch solver
        return self._solve_batch(
            target_pose, q0,
            method=method,
            pos_weight=pos_weight,
            ori_weight=ori_weight,
            adaptive_damping=adaptive_damping,
            adaptive_step=adaptive_step,
            use_numeric_jacobian=use_numeric_jacobian,
            use_central_diff=use_central_diff,
            transpose_gain=transpose_gain,
            max_step_norm=max_step_norm,
            refine=refine,
            refine_iters=refine_iters,
            refine_pos_tol=refine_pos_tol,
            refine_ori_tol=refine_ori_tol,
            target_link=target_link,
            row_mask=row_mask,
            nullspace_gain=nullspace_gain,
            joint_centering=joint_centering,
            joint_center_gain=joint_center_gain,
            joint_center_weights=joint_center_weights,
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
        max_step_norm: float = 0.5,
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
            best_q = q.clone()  # Ensure best_q is a copy, not a reference
            best_err = torch.tensor(math.inf, dtype=self.dtype, device=self.device)
            plateau_counter = 0
            prev_err_norm = torch.tensor(math.inf, dtype=self.dtype, device=self.device)
            jac_type_local = "analytic"
            final_pos_err = float('inf')
            final_ori_err = float('inf')

            for it in range(1, self.max_iters + 1):
                if target_link is None:
                    T_cur_result = self.fk_solver.solve(q, return_end_only=True, device=self.device, dtype=self.dtype)
                    # Handle both dict and tensor returns
                    if isinstance(T_cur_result, dict):
                        T_cur = T_cur_result["end"]
                    else:
                        T_cur = T_cur_result
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
                            for js in self.model._chain_dof_list:  # type: ignore[attr-defined]
                                lo, hi = -1.0, 1.0
                                if js.limit_lower is not None:
                                    lo = js.limit_lower
                                if js.limit_upper is not None:
                                    hi = js.limit_upper
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
                
                # Only apply max_step_norm clipping when adaptive_step is True
                # When adaptive_step=False, user controls step via base_step (like pytorch_kinematics lr)
                if adaptive_step and max_step_norm is not None and max_step_norm > 0:
                    dq_norm = torch.linalg.norm(dq_step)
                    if dq_norm > max_step_norm:
                        dq_step = dq_step * (max_step_norm / (dq_norm + 1e-15))
                new_q = self._apply_joint_limits(q + dq_step)

                if backtrack:
                    prev_total = err_norm
                    for _bt in range(3):
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
        # Use eye6 directly if already on correct device/dtype, otherwise create on-the-fly
        # Creating eye6 on-the-fly is faster than .to() for small matrices
        if J.device == self.device and J.dtype == self.dtype:
            eye6 = self._eye6
        else:
            eye6 = torch.eye(6, device=J.device, dtype=J.dtype)
        A = J @ J.transpose(0, 1) + (damping ** 2) * eye6
        try:
            y = torch.linalg.solve(A, err)
        except Exception:
            # CPU fallback (e.g., MPS not supporting op)
            y = torch.linalg.solve(A.cpu(), err.cpu()).to(device=J.device, dtype=J.dtype)
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
        """Compute adaptive step size based on error magnitude.
        
        More aggressive for large errors to escape local minima,
        matching NumPy solver behavior for better workspace boundary handling.
        """
        norm_pos = pos_err / 0.01
        norm_ori = ori_err / 0.087  # ~5°
        m = max(norm_pos, norm_ori)
        if m > 2.0:
            # Large error: be more aggressive
            return self.base_step * 1.0
        elif m > 1.0:
            return self.base_step * 1.2
        elif m > 0.5:
            return self.base_step * 1.0
        else:
            # Near convergence: smaller steps for precision (matching NumPy)
            return self.base_step * 0.5

    def _apply_joint_limits(self, q: Tensor) -> Tensor:
        out = q.clone()
        for js in self.model._chain_dof_list:  # type: ignore[attr-defined]
            if js.limit_lower is not None or js.limit_upper is not None:
                if js.limit_lower is not None:
                    out[js.index] = torch.clamp(out[js.index], min=float(js.limit_lower))
                if js.limit_upper is not None:
                    out[js.index] = torch.clamp(out[js.index], max=float(js.limit_upper))
        return out

    # ==================== Partial FK (single) ====================
    def _fk_until_torch(self, q: Tensor, target_link: str) -> Tensor:
        """Compute 4x4 pose of an intermediate link (target_link).

        Mirrors the early-stop traversal logic used in the Torch Jacobian solver
        (analytic path) to ensure consistent frames and axis extraction.
        """
        if not torch.is_tensor(q):
            q = torch.tensor(q, dtype=self.dtype, device=self.device)
        q_map = {js.name: q[js.index] for js in self.model._chain_dof_list}  # type: ignore[attr-defined]
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
        adaptive_damping: bool = True,
        adaptive_step: bool = True,
        use_numeric_jacobian: bool = False,
        use_central_diff: bool = True,
        transpose_gain: float | None = None,
        max_step_norm: float = 0.5,
        refine: bool = False,
        refine_iters: int = 5,
        refine_pos_tol: float | None = None,
        refine_ori_tol: float | None = None,
        target_link: str | None = None,
        row_mask: Sequence[int | bool] | None = None,
        nullspace_gain: float = 0.0,
        joint_centering: bool = True,
        joint_center_gain: float = 0.2,
        joint_center_weights: Sequence[float] | None = None,
    ) -> Dict:
        """Vectorized batch IK with full adaptive logic.

        :param target_poses_batch: [B,4,4]
        :param q_init_batch: [B,n]
        :return: dict with q, success, iters, method, pos_err, ori_err
        """
        B, n = q_init_batch.shape
        q = q_init_batch.clone()
        best_q = q.clone()
        best_err = torch.full((B,), float('inf'), dtype=self.dtype, device=self.device)
        success = torch.zeros(B, dtype=torch.bool, device=self.device)
        iters = torch.zeros(B, dtype=torch.int32, device=self.device)
        active = torch.ones(B, dtype=torch.bool, device=self.device)
        prev_err = torch.full((B,), float('inf'), dtype=self.dtype, device=self.device)
        plateau = torch.zeros(B, dtype=torch.int32, device=self.device)

        p_target = target_poses_batch[:, :3, 3]
        R_target = target_poses_batch[:, :3, :3]
        eye6 = torch.eye(6, device=self.device, dtype=self.dtype).unsqueeze(0)

        # Pre-compute row mask
        if row_mask is not None:
            mask_bool = torch.tensor([bool(m) for m in row_mask], dtype=torch.bool, device=self.device)
            if len(mask_bool) != 6:
                raise ValueError("row_mask must have length 6")
        else:
            mask_bool = None

        # Jacobian method
        jac_method = "numeric" if use_numeric_jacobian else "analytic"

        for it in range(1, self.max_iters + 1):
            if not active.any():
                break

            act_idx = torch.where(active)[0]
            q_act = q[act_idx]
            Ba = q_act.shape[0]

            # FK
            T_end = self.fk_solver.solve(q_act, return_end_only=True, device=self.device, dtype=self.dtype)
            if isinstance(T_end, dict):
                T_end = T_end.get('end', list(T_end.values())[-1])
            if T_end.ndim == 2:
                T_end = T_end.unsqueeze(0)
            p_end = T_end[:, :3, 3]
            R_end = T_end[:, :3, :3]

            # Errors
            p_err = p_target[act_idx] - p_end
            R_err_vec = self._batch_rotation_error(R_end, R_target[act_idx])
            pos_norm = torch.linalg.norm(p_err, dim=1)
            ori_norm = torch.linalg.norm(R_err_vec, dim=1)
            # Use simple sum for best_err tracking (matching NumPy)
            err_norm = pos_norm + ori_norm

            # Weighted error vector (for solver)
            full_err = torch.cat([pos_weight * p_err, ori_weight * R_err_vec], dim=1)
            if mask_bool is not None:
                err = full_err[:, mask_bool]
            else:
                err = full_err

            # Track best solution
            improved = err_norm < best_err[act_idx]
            best_err[act_idx] = torch.where(improved, err_norm, best_err[act_idx])
            for i, idx in enumerate(act_idx):
                if improved[i]:
                    best_q[idx] = q_act[i]

            # Plateau detection
            delta = prev_err[act_idx] - err_norm
            plateau[act_idx] = torch.where(delta < 1e-8, plateau[act_idx] + 1, torch.zeros_like(plateau[act_idx]))
            prev_err[act_idx] = err_norm

            # Convergence check
            conv_mask = (pos_norm < self.pos_tol) & (ori_norm < self.ori_tol)
            if conv_mask.any():
                g_idx = act_idx[conv_mask]
                # Refine phase for converged samples
                if refine:
                    r_pos_tol = refine_pos_tol or (self.pos_tol * 0.2)
                    r_ori_tol = refine_ori_tol or (self.ori_tol * 0.2)
                    for idx in g_idx:
                        q_ref = q[idx].clone()
                        for _r in range(refine_iters):
                            T_r = self.fk_solver.solve(q_ref.unsqueeze(0), return_end_only=True, device=self.device, dtype=self.dtype)
                            if isinstance(T_r, dict):
                                T_r = T_r.get('end', list(T_r.values())[-1])
                            if T_r.ndim == 2:
                                T_r = T_r.unsqueeze(0)
                            p_r = T_r[0, :3, 3]
                            R_r = T_r[0, :3, :3]
                            p_err_r = p_target[idx] - p_r
                            R_err_r = self._batch_rotation_error(R_r.unsqueeze(0), R_target[idx:idx+1])[0]
                            if torch.linalg.norm(p_err_r) < r_pos_tol and torch.linalg.norm(R_err_r) < r_ori_tol:
                                q[idx] = q_ref
                                break
                            J_ref = self.jacobian_solver.solve(q_ref.unsqueeze(0), method="analytic", device=self.device, dtype=self.dtype)
                            if J_ref.ndim == 2:
                                J_ref = J_ref.unsqueeze(0)
                            if pos_weight != 1.0:
                                J_ref[:, :3, :] *= pos_weight
                            if ori_weight != 1.0:
                                J_ref[:, 3:6, :] *= ori_weight
                            e_ref = torch.cat([pos_weight * p_err_r, ori_weight * R_err_r])
                            dq_ref = self._solve_pinv(J_ref[0], e_ref, self.min_damping)
                            dq_norm_ref = torch.linalg.norm(dq_ref)
                            if dq_norm_ref > 0.2:
                                dq_ref = dq_ref * (0.2 / (dq_norm_ref + 1e-15))
                            q_ref = self._apply_joint_limits(q_ref + dq_ref)
                        q[idx] = q_ref

                success[g_idx] = True
                iters[g_idx] = it
                active[g_idx] = False

            if (~active).all():
                break

            # Filter to non-converged
            mask_keep = ~conv_mask
            if not mask_keep.any():
                break

            q_sub = q_act[mask_keep]
            p_err_sub = p_err[mask_keep]
            R_err_sub = R_err_vec[mask_keep]
            pos_norm_sub = pos_norm[mask_keep]
            ori_norm_sub = ori_norm[mask_keep]
            plateau_sub = plateau[act_idx[mask_keep]]
            act_idx = act_idx[mask_keep]
            Ba_sub = q_sub.shape[0]

            # Jacobian
            J = self.jacobian_solver.solve(
                q_sub,
                method=jac_method,
                use_central_diff=use_central_diff,
                device=self.device,
                dtype=self.dtype,
                target_link=target_link,
            )
            if J.ndim == 2:
                J = J.unsqueeze(0)

            # Apply weights to Jacobian
            if pos_weight != 1.0:
                J[:, :3, :] *= pos_weight
            if ori_weight != 1.0:
                J[:, 3:6, :] *= ori_weight

            # Apply row mask
            if mask_bool is not None:
                J_eff = J[:, mask_bool, :]
                e_eff = err[mask_keep]
            else:
                J_eff = J
                e_eff = err

            # Adaptive damping per sample
            if adaptive_damping:
                lam = torch.full((Ba_sub,), math.sqrt(self.min_damping * self.max_damping), device=self.device, dtype=self.dtype)
                for k in range(Ba_sub):
                    try:
                        S = torch.linalg.svdvals(J_eff[k])
                        cond = S[0] / max(S[-1], 1e-12)
                    except RuntimeError:
                        cond = 100.0
                    err_combo = pos_norm_sub[k] + 0.5 * ori_norm_sub[k]
                    if cond > 200 or err_combo > 0.05:
                        lam[k] = self.max_damping
                    elif cond < 30 and err_combo < 0.01:
                        lam[k] = self.min_damping
                    else:
                        lam[k] = (self.min_damping + self.max_damping) * 0.5
                    if plateau_sub[k] >= 4:
                        lam[k] = min(lam[k] * 2.0, self.max_damping * 2)
                    if plateau_sub[k] >= 8:
                        lam[k] = min(lam[k] * 1.5, self.max_damping * 4)
            else:
                lam = torch.full((Ba_sub,), (self.min_damping + self.max_damping) / 2, device=self.device, dtype=self.dtype)

            # Solve per sample - avoid .item() calls in hot loop
            dq = torch.zeros((Ba_sub, n), dtype=self.dtype, device=self.device)
            # Convert lam to list once to avoid repeated .item() calls
            lam_list = lam.tolist() if isinstance(lam, torch.Tensor) else [lam] * Ba_sub if not isinstance(lam, (list, tuple)) else lam
            for k in range(Ba_sub):
                Jk = J_eff[k]
                ek = e_eff[k]
                lam_k = lam_list[k] if isinstance(lam_list, (list, tuple)) else lam[k].item()
                if method == "dls":
                    dq[k] = self._solve_dls(Jk, ek, lam_k)
                elif method == "pinv":
                    dq[k] = self._solve_pinv(Jk, ek, lam_k)
                else:  # transpose
                    if transpose_gain is not None:
                        alpha = transpose_gain
                    else:
                        J_err = Jk.T @ ek
                        JJt_err = Jk @ J_err
                        err_norm_sq = torch.dot(ek, ek)
                        JJt_err_norm_sq = torch.dot(JJt_err, JJt_err)
                        if JJt_err_norm_sq > 1e-12:
                            alpha = err_norm_sq / JJt_err_norm_sq
                        else:
                            alpha = 0.01
                        alpha = torch.clamp(alpha, 0.001, 0.5)
                    dq[k] = alpha * (Jk.T @ ek)

            # Nullspace redundancy handling
            if nullspace_gain > 0 and n > J_eff.shape[1]:
                for k in range(Ba_sub):
                    try:
                        Jk = J_eff[k]
                        U_ns, S_ns, Vt_ns = torch.linalg.svd(Jk, full_matrices=False)
                        S_inv_ns = torch.where(S_ns > 1e-9, 1.0 / S_ns, torch.zeros_like(S_ns))
                        J_pinv_eff = (Vt_ns.T * S_inv_ns) @ U_ns.T
                        N = torch.eye(n, dtype=self.dtype, device=self.device) - J_pinv_eff @ Jk
                        if joint_centering:
                            centers = []
                            for js in self.model._chain_dof_list:
                                lo, hi = -1.0, 1.0
                                if js.limit_lower is not None:
                                    lo = js.limit_lower
                                if js.limit_upper is not None:
                                    hi = js.limit_upper
                                centers.append(0.5 * (lo + hi))
                            centers_t = torch.tensor(centers, dtype=self.dtype, device=self.device)
                            delta_center = centers_t - q_sub[k]
                            if joint_center_weights is not None and len(joint_center_weights) == n:
                                w = torch.tensor(joint_center_weights, dtype=self.dtype, device=self.device)
                                delta_center = delta_center * w
                            dq_sec = joint_center_gain * delta_center
                        else:
                            dq_sec = torch.zeros(n, dtype=self.dtype, device=self.device)
                        dq[k] = dq[k] + nullspace_gain * (N @ dq_sec)
                    except Exception:
                        pass

            # Adaptive step (per sample, matching NumPy implementation)
            if adaptive_step and method != "transpose":
                step = torch.zeros(Ba_sub, dtype=self.dtype, device=self.device)
                for k in range(Ba_sub):
                    step[k] = self._compute_adaptive_step(pos_norm_sub[k].item(), ori_norm_sub[k].item())
                # Plateau slowdown
                step = torch.where(plateau_sub >= 8, step * 0.5, step)
                dq = dq * step.unsqueeze(1)
            elif method == "transpose":
                pass  # Step already in alpha
            else:
                dq = dq * self.base_step

            # Step norm clipping (match NumPy _solve_batch: clip whenever max_step_norm is set)
            if max_step_norm is not None and max_step_norm > 0:
                dq_norm = torch.linalg.norm(dq, dim=1, keepdim=True)
                scale = torch.clamp(max_step_norm / (dq_norm + 1e-12), max=1.0)
                dq = dq * scale

            # Update with joint limits
            q_new = q_sub + dq
            for js in self.model._chain_dof_list:
                j = js.index
                if js.limit_lower is not None:
                    q_new[:, j] = torch.clamp(q_new[:, j], min=float(js.limit_lower))
                if js.limit_upper is not None:
                    q_new[:, j] = torch.clamp(q_new[:, j], max=float(js.limit_upper))
            q[act_idx] = q_new

        # Use best solution for non-converged
        remaining = active.nonzero(as_tuple=False).flatten()
        if remaining.numel() > 0:
            iters[remaining] = self.max_iters
            q[remaining] = best_q[remaining]

        # Final errors
        T_final = self.fk_solver.solve(q, return_end_only=True, device=self.device, dtype=self.dtype)
        if isinstance(T_final, dict):
            T_final = T_final.get('end', list(T_final.values())[-1])
        if T_final.ndim == 2:
            T_final = T_final.unsqueeze(0)
        p_err_final = torch.linalg.norm(p_target - T_final[:, :3, 3], dim=1)
        ori_err_final = torch.linalg.norm(self._batch_rotation_error(T_final[:, :3, :3], R_target), dim=1)

        return {
            "q": q,
            "success": success,
            "iters": iters,
            "method": method,
            "pos_err": p_err_final,
            "ori_err": ori_err_final,
        }

    
    @staticmethod
    def _batch_rotation_error(R_current: Tensor, R_target: Tensor) -> Tensor:
        """Compute rotation error in angle-axis representation for batch.
        
        Matches transform.rotation_error: R_error = R_target @ R_current^T
        
        :param R_current: [B, 3, 3] current rotation matrices
        :param R_target: [B, 3, 3] target rotation matrices
        :return: [B, 3] rotation error vectors
        """
        # R_error = R_target @ R_current^T (matches rotation_error in transform)
        R_error = torch.bmm(R_target, R_current.transpose(1, 2))
        
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
