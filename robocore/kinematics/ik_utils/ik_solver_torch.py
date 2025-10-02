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
from typing import Dict, Optional, TYPE_CHECKING

try:
    import torch
except ImportError as e:  # pragma: no cover
    raise ImportError("ik_solver_torch 需要 PyTorch, 请先: pip install torch") from e

from ..jacobian_utils.jacobian_solver_torch import JacobianSolverTorch
from ..fk_utils.fk_solver_torch import FKSolverTorch
from robocore.transform.transform_core import orientation_error_torch
try:  # 可能存在设备选择工具
    from ...utils.torch_utils import select_device  # type: ignore
except Exception:  # pragma: no cover
    def select_device():  # 兜底
        return torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))

if TYPE_CHECKING:
    from robocore.modeling.robot_model import RobotModel

Tensor = torch.Tensor


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

        # MPS 设备特殊处理
        self.is_mps = str(self.device) == "mps"
        if dtype is None:
            # MPS 对 float64 支持不完善
            if self.is_mps:
                self.dtype = torch.float32
            else:
                self.dtype = torch.float64
        else:
            self.dtype = dtype
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
        transpose_gain: Optional[float] = None,
        adaptive_damping: bool = True,
        adaptive_step: bool = True,
        use_numeric_jacobian: bool = False,
        use_central_diff: bool = True,
        max_step_norm: float = 0.3,
        backtrack: bool = False,  # 默认关闭以加速
        refine: bool = False,  # 默认关闭以加速
        refine_iters: int = 5,  # 减少refine迭代次数
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
            best_q = q
            best_err = torch.tensor(math.inf, dtype=self.dtype, device=self.device)
            plateau_counter = 0
            prev_err_norm = torch.tensor(math.inf, dtype=self.dtype, device=self.device)
            jac_type_local = "analytic"
            final_pos_err = float('inf')
            final_ori_err = float('inf')

            for it in range(1, self.max_iters + 1):
                T_cur = self.fk_solver.solve(q, return_end_only=True, device=self.device, dtype=self.dtype)["end"]
                R_cur = T_cur[:3, :3]
                p_cur = T_cur[:3, 3]
                pos_err_v = p_target - p_cur
                ori_err_v = orientation_error_torch(R_cur, R_target)
                pos_err_norm_t = torch.linalg.norm(pos_err_v)  # 保持为tensor
                ori_err_norm_t = torch.linalg.norm(ori_err_v)  # 保持为tensor

                # 分段动态姿态权重 - 延迟.item()调用
                ori_err_norm_val = ori_err_norm_t.item()
                if ori_err_norm_val > 1.0:
                    ori_scale = 0.3
                elif ori_err_norm_val > 0.7:
                    ori_scale = 0.5
                elif ori_err_norm_val > 0.4:
                    ori_scale = 0.8
                else:
                    ori_scale = 1.0
                ori_weight_dyn = ori_weight * ori_scale
                err = torch.cat([pos_weight * pos_err_v, ori_weight_dyn * ori_err_v])
                err_norm = torch.linalg.norm(err)

                # 更新最优解 - 使用tensor比较
                if err_norm < best_err:
                    best_err = err_norm
                    best_q = q.clone()
                    final_pos_err = pos_err_norm_t.item()
                    final_ori_err = ori_err_norm_val

                if prev_err_norm - err_norm < 1e-8:
                    plateau_counter += 1
                else:
                    plateau_counter = 0
                prev_err_norm = err_norm

                # 收敛检查 - 只在这里调用.item()
                pos_err_norm = pos_err_norm_t.item()
                ori_err_norm = ori_err_norm_val

                if pos_err_norm < self.pos_tol and ori_err_norm < self.ori_tol:
                    if refine:
                        r_pos_tol = refine_pos_tol or (self.pos_tol * 0.2)
                        r_ori_tol = refine_ori_tol or (self.ori_tol * 0.2)
                        q_ref = q.clone()
                        for _ in range(refine_iters):
                            T_r = self.fk_solver.solve(q_ref, return_end_only=True, device=self.device, dtype=self.dtype)["end"]
                            p_r = T_r[:3, 3]; R_r = T_r[:3, :3]
                            p_e = p_target - p_r
                            o_e = orientation_error_torch(R_r, R_target)
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
                    J = self.jacobian_solver.solve(
                        q,
                        method="numeric",
                        use_central_diff=use_central_diff,
                        device=self.device,
                        dtype=self.dtype
                    )
                    jac_type_local = "numeric_central" if use_central_diff else "numeric_forward"
                else:
                    J = self.jacobian_solver.solve(
                        q,
                        method="analytic",
                        device=self.device,
                        dtype=self.dtype
                    )
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
                    dq = alpha * (J.transpose(0, 1) @ err)

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
                        T_bt = self.fk_solver.solve(new_q, return_end_only=True, device=self.device, dtype=self.dtype)["end"]
                        p_bt = T_bt[:3, 3]; R_bt = T_bt[:3, :3]
                        pos_bt = torch.linalg.norm(p_target - p_bt).item()
                        ori_bt = torch.linalg.norm(orientation_error_torch(R_bt, R_target)).item()
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
        # MPS 不支持 SVD，直接在 CPU 上计算
        if self.is_mps:
            J_cpu = J.cpu()
            err_cpu = err.cpu()
            try:
                U, S, Vh = torch.linalg.svd(J_cpu, full_matrices=False)
            except Exception:
                return self._solve_dls(J, err, damping)

            if damping > 0:
                S_inv = S / (S * S + damping * damping)
            else:
                tol = 1e-9 * max(J_cpu.shape)
                S_inv = torch.where(S > tol, 1.0 / S, torch.zeros_like(S))

            result = (Vh.transpose(0, 1) * S_inv) @ (U.transpose(0, 1) @ err_cpu)
            return result.to(J.device)

        # 非 MPS 设备
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
        # 使用SVD计算条件数（比特征值分解更快更稳定）
        # MPS 不支持 svdvals，在 CPU 上计算
        try:
            if self.is_mps:
                S = torch.linalg.svdvals(J.cpu())
            else:
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