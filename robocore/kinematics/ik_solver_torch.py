"""PyTorch 版本逆运动学求解器 (IKSolverTorch)。

特性：
1. 支持三种方法：dls / pinv / transpose
2. 自适应阻尼 + 自适应步长（与 NumPy 参考实现一致思想）
3. Plateau 检测：若误差长时间不下降，放大阻尼避免震荡
4. 动态姿态权重：角度误差 > 0.7rad 时暂时降低一半 (Ori weight * 0.5)
5. 可选使用数值 Jacobian（中心或前向差分）验证解析正确性
6. 精修阶段（refine）：在收敛阈值内继续用解析 Jacobian 做若干小步以压缩残差
"""
from __future__ import annotations

import math
from typing import Dict, Optional

try:
    import torch
except ImportError as e:  # pragma: no cover
    raise ImportError("ik_solver_torch 需要 PyTorch, 请先: pip install torch") from e

from .jacobian_analytic_torch import analytic_jacobian_torch
from .fk_torch import forward_kinematics_torch_end
from .jacobian_numeric_torch import numeric_jacobian_torch
try:  # 可能存在设备选择工具
    from .torch_utils import select_device  # type: ignore
except Exception:  # pragma: no cover
    def select_device():  # 兜底
        return torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))

Tensor = torch.Tensor


def _orientation_error_torch(Ra: Tensor, Rb: Tensor) -> Tensor:
    """返回从 Ra 旋转到 Rb 的轴角误差向量 (axis * angle)。

    算法：利用 R_err = Ra^T Rb，angle = acos((trace(R_err)-1)/2)，
    axis 采用标准反对称提取并对 0/π 奇异点做稳健处理。
    """
    R_err = Ra.transpose(0, 1) @ Rb
    trace_val = torch.clamp((torch.trace(R_err) - 1.0) * 0.5, -1.0, 1.0)
    angle = torch.acos(trace_val)
    if angle < 1e-12:
        return torch.zeros(3, dtype=Ra.dtype, device=Ra.device)
    if angle > math.pi - 1e-6:
        # 近似 π，使用对角线辅助找轴
        axis = torch.stack([
            R_err[2, 1] - R_err[1, 2],
            R_err[0, 2] - R_err[2, 0],
            R_err[1, 0] - R_err[0, 1],
        ])
        if torch.linalg.norm(axis) < 1e-8:
            diag = torch.diag(R_err)
            axis = torch.sqrt(torch.clamp(diag + 1.0, min=0))
        axis = axis / (torch.linalg.norm(axis) + 1e-15)
        return axis * angle
    denom = 2.0 * torch.sin(angle)
    axis = torch.stack([
        (R_err[2, 1] - R_err[1, 2]) / (denom + 1e-15),
        (R_err[0, 2] - R_err[2, 0]) / (denom + 1e-15),
        (R_err[1, 0] - R_err[0, 1]) / (denom + 1e-15),
    ])
    return axis * angle


class IKSolverTorch:
    def __init__(
        self,
        model,
        max_iters: int = 100,
        pos_tol: float = 1e-4,
        ori_tol: float = 1e-3,
        min_damping: float = 1e-6,
        max_damping: float = 1e-2,
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
        if dtype is None:
            # MPS 对 float64 支持不完善
            if str(self.device) == "mps":
                self.dtype = torch.float32
            else:
                self.dtype = torch.float64
        else:
            self.dtype = dtype
        # 关节数量（假定 model._actuated 与 numpy 版本一致）
        self.n = len(getattr(model, "_actuated"))

    # -------------------- 主求解 --------------------
    def solve(
        self,
        target_pose,
        q0,
        *,
        method: str = "pinv",
        pos_weight: float = 1.0,
        ori_weight: float = 1.0,
        transpose_gain: Optional[float] = None,
        adaptive_damping: bool = True,
        adaptive_step: bool = True,
        use_numeric_jacobian: bool = False,
        use_central_diff: bool = True,
        max_step_norm: float = 0.3,
        backtrack: bool = True,
        refine: bool = True,
        refine_iters: int = 10,
        refine_pos_tol: Optional[float] = None,
        refine_ori_tol: Optional[float] = None,
        # 额外增强参数
        restarts: int = 0,
        restart_noise: float = 0.25,  # 相对随机扰动幅度 (弧度)
        random_seed: Optional[int] = None,
    ) -> Dict:
        method = method.lower()
        if method not in ("dls", "pinv", "transpose"):
            raise ValueError(f"未知 IK 方法 {method}")

        if not torch.is_tensor(target_pose):
            target_pose = torch.tensor(target_pose, dtype=self.dtype, device=self.device)
        else:
            target_pose = target_pose.to(dtype=self.dtype, device=self.device)

        if random_seed is not None:
            torch.manual_seed(random_seed)

        if not torch.is_tensor(q0):
            base_q0 = torch.tensor(q0, dtype=self.dtype, device=self.device)
        else:
            base_q0 = q0.clone().to(dtype=self.dtype, device=self.device)
        if base_q0.numel() != self.n:
            raise ValueError(f"q0 size {base_q0.numel()} != dof {self.n}")

        R_target = target_pose[:3, :3]
        p_target = target_pose[:3, 3]

        attempt_results = []

        def run_one(q_init: Tensor):
            q = q_init.clone()
            best_q = q.clone()
            best_err = math.inf
            plateau_counter = 0
            prev_err_norm = math.inf
            jac_type_local = "analytic"
            final_pos_err = float('inf')
            final_ori_err = float('inf')

            for it in range(1, self.max_iters + 1):
                T_cur = forward_kinematics_torch_end(self.model, q, device=self.device, dtype=self.dtype)
                R_cur = T_cur[:3, :3]
                p_cur = T_cur[:3, 3]
                pos_err_v = p_target - p_cur
                ori_err_v = _orientation_error_torch(R_cur, R_target)
                pos_err_norm = torch.linalg.norm(pos_err_v).item()
                ori_err_norm = torch.linalg.norm(ori_err_v).item()
                # 分段动态姿态权重
                if ori_err_norm > 1.0:
                    ori_scale = 0.3
                elif ori_err_norm > 0.7:
                    ori_scale = 0.5
                elif ori_err_norm > 0.4:
                    ori_scale = 0.8
                else:
                    ori_scale = 1.0
                ori_weight_dyn = ori_weight * ori_scale
                err = torch.cat([pos_weight * pos_err_v, ori_weight_dyn * ori_err_v])
                err_norm = torch.linalg.norm(err).item()
                if err_norm < best_err:
                    best_err = err_norm
                    best_q = q.clone()
                    final_pos_err = pos_err_norm
                    final_ori_err = ori_err_norm

                if prev_err_norm - err_norm < 1e-8:
                    plateau_counter += 1
                else:
                    plateau_counter = 0
                prev_err_norm = err_norm

                # 收敛
                if pos_err_norm < self.pos_tol and ori_err_norm < self.ori_tol:
                    if refine:
                        r_pos_tol = refine_pos_tol or (self.pos_tol * 0.2)
                        r_ori_tol = refine_ori_tol or (self.ori_tol * 0.2)
                        q_ref = q.clone()
                        for _ in range(refine_iters):
                            T_r = forward_kinematics_torch_end(self.model, q_ref, device=self.device, dtype=self.dtype)
                            p_r = T_r[:3, 3]; R_r = T_r[:3, :3]
                            p_e = p_target - p_r
                            o_e = _orientation_error_torch(R_r, R_target)
                            if torch.linalg.norm(p_e) < r_pos_tol and torch.linalg.norm(o_e) < r_ori_tol:
                                q = q_ref
                                final_pos_err = torch.linalg.norm(p_e).item()
                                final_ori_err = torch.linalg.norm(o_e).item()
                                best_q = q.clone()
                                best_err = math.sqrt(final_pos_err**2 + final_ori_err**2)
                                break
                            J_ref = analytic_jacobian_torch(self.model, q_ref, device=self.device, dtype=self.dtype)
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
                        "err_norm": float(best_err),
                        "pos_err": float(final_pos_err),
                        "ori_err": float(final_ori_err),
                        "method": method,
                        "jacobian": jac_type_local,
                    }

                # Jacobian
                if use_numeric_jacobian:
                    J = numeric_jacobian_torch(self.model, q, use_central_diff=use_central_diff, device=self.device, dtype=self.dtype)
                    jac_type_local = "numeric_central" if use_central_diff else "numeric_forward"
                else:
                    J = analytic_jacobian_torch(self.model, q, device=self.device, dtype=self.dtype)
                    jac_type_local = "analytic"
                if pos_weight != 1.0:
                    J[:3, :] *= pos_weight
                if ori_weight_dyn != 1.0:
                    J[3:6, :] *= ori_weight_dyn

                # 阻尼
                if adaptive_damping:
                    damping = self._compute_adaptive_damping(J, pos_err_norm, ori_err_norm)
                else:
                    damping = 0.5 * (self.min_damping + self.max_damping)
                if plateau_counter >= 4:
                    damping = max(damping * 2.0, self.max_damping)
                if plateau_counter >= 8:
                    # 进一步加大阻尼并略微减小步长
                    damping = max(damping * 1.5, self.max_damping * 2.0)
                # 解
                if method == "dls":
                    dq = self._solve_dls(J, err, damping)
                elif method == "pinv":
                    dq = self._solve_pinv(J, err, damping)
                else:
                    if transpose_gain is not None:
                        alpha = transpose_gain
                    else:
                        try:
                            smax = torch.linalg.svdvals(J)[0].item()
                        except Exception:
                            smax = 1.0
                        alpha = 0.9 / (smax * smax + 1e-9)
                    dq = alpha * (J.transpose(0, 1) @ err)

                # 步长
                if adaptive_step:
                    step = self._compute_adaptive_step(pos_err_norm, ori_err_norm)
                else:
                    step = self.base_step
                if plateau_counter >= 8:
                    step *= 0.5

                dq_step = step * dq
                dq_norm = torch.linalg.norm(dq_step)
                if dq_norm > max_step_norm:
                    dq_step = dq_step * (max_step_norm / (dq_norm + 1e-15))
                new_q = self._apply_joint_limits(q + dq_step)

                if backtrack:
                    prev_total = err_norm
                    for _bt in range(3):
                        T_bt = forward_kinematics_torch_end(self.model, new_q, device=self.device, dtype=self.dtype)
                        p_bt = T_bt[:3, 3]; R_bt = T_bt[:3, :3]
                        pos_bt = torch.linalg.norm(p_target - p_bt).item()
                        ori_bt = torch.linalg.norm(_orientation_error_torch(R_bt, R_target)).item()
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
                "err_norm": float(best_err),
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
        A = J @ J.transpose(0, 1) + (damping ** 2) * torch.eye(6, dtype=J.dtype, device=J.device)
        try:
            y = torch.linalg.solve(A, err)
        except Exception:
            # CPU fallback (e.g., MPS not supporting op)
            y = torch.linalg.solve(A.cpu(), err.cpu()).to(device=J.device)
        return J.transpose(0, 1) @ y

    def _solve_pinv(self, J: Tensor, err: Tensor, damping: float) -> Tensor:
        try:
            U, S, Vh = torch.linalg.svd(J, full_matrices=False)
        except RuntimeError:
            # 尝试 CPU 退化
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
        JJt = J @ J.transpose(0, 1)
        try:
            eigvals = torch.linalg.eigvalsh(JJt)
        except Exception:
            try:
                eigvals = torch.linalg.eigvalsh(JJt.cpu()).to(J.device)
            except Exception:
                eigvals = None
        if eigvals is not None:
            s_max = torch.sqrt(torch.max(eigvals)).item()
            s_min = torch.sqrt(torch.clamp(torch.min(eigvals), min=1e-12)).item()
            cond = s_max / s_min
        else:
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


def inverse_kinematics_torch(model, target_pose, q0, **kwargs):
    solver = IKSolverTorch(model)
    return solver.solve(target_pose, q0, **kwargs)

__all__ = ["IKSolverTorch", "inverse_kinematics_torch"]
