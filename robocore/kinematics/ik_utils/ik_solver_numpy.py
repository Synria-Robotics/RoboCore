"""NumPy-accelerated inverse kinematics solver.

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

from typing import TYPE_CHECKING, Dict, Sequence
import numpy as np
from ..jacobian_utils.jacobian_solver_numpy import JacobianSolverNumPy
from robocore.transform import rotation_error
from robocore.kinematics.fk import forward_kinematics

if TYPE_CHECKING:
    from robocore.modeling.robot_model import RobotModel


class IKSolverNumPy:
    """NumPy-accelerated IK solver with adaptive strategies.

    Features:
    - Damped Least Squares (DLS) method
    - Adaptive damping based on condition number and error
    - Adaptive step size based on error magnitude
    - Central difference Jacobian for better accuracy
    - Joint limit clamping
    """

    def __init__(
        self,
        model: "RobotModel",
        max_iters: int = 100,
        pos_tol: float = 1e-3,
        ori_tol: float = 1e-3,
        min_damping: float = 1e-4,  # 调整: 与JS版本一致 (原来 1e-6)
        max_damping: float = 5e-2,  # 调整: 与JS版本一致 (原来 1e-2), 对奇异点处理至关重要
        base_step: float = 1.0,
    ):
        """Initialize IK solver.

        :param model: robot model.
        :param max_iters: maximum iterations.
        :param pos_tol: position convergence tolerance (meters).
        :param ori_tol: orientation convergence tolerance (radians).
        :param min_damping: minimum damping factor.
        :param max_damping: maximum damping factor.
        :param base_step: base step size multiplier.
        """
        self.model = model
        self.max_iters = max_iters
        self.pos_tol = pos_tol
        self.ori_tol = ori_tol
        self.min_damping = min_damping
        self.max_damping = max_damping
        self.base_step = base_step
        self.n = model.num_dof()
        # Initialize Jacobian solver
        self.jacobian_solver = JacobianSolverNumPy(model)

    def solve(
        self,
        target_pose: np.ndarray,
        q0: np.ndarray,
        pos_weight: float = 1.0,
        ori_weight: float = 1.0,
        adaptive_damping: bool = True,
        adaptive_step: bool = True,
        use_central_diff: bool = True,
        use_analytic_jacobian: bool = False,
        method: str = "dls",
        transpose_gain: float | None = None,
        max_step_norm: float = 0.5,  # Increased from 0.3 for better workspace boundary handling
        refine: bool = False,
        refine_iters: int = 10,
        refine_pos_tol: float | None = None,
        refine_ori_tol: float | None = None,
        # Local task extension
        target_link: str | None = None,
        row_mask: Sequence[int | bool] | None = None,
        # Redundancy control
        nullspace_gain: float = 0.0,
        joint_centering: bool = True,
        joint_center_gain: float = 0.2,
        joint_center_weights: Sequence[float] | None = None,
    ) -> Dict[str, object]:
        """Solve IK with selectable method.

        Supported methods:
          - dls: damped least squares (JJ^T regularization)
          - pinv: SVD pseudoinverse with Tikhonov damping
          - transpose: J^T * error with adaptive gain
        """
        method = method.lower()
        if method not in ("dls", "pinv", "transpose"):
            raise ValueError(f"Unknown IK method '{method}'")
        q0 = np.asarray(q0, dtype=np.float64)
        target_pose = np.asarray(target_pose, dtype=np.float64)
        
        if q0.shape[0] != self.n:
            raise ValueError(f"Expected q0 with {self.n} elements, got {q0.shape[0]}")
        
        # Extract target position and rotation
        R_target = target_pose[:3, :3]
        p_target = target_pose[:3, 3]
        
        q = q0.copy()
        best_q = q.copy()
        best_err = np.inf
        best_pos_err = np.inf
        best_ori_err = np.inf
        jac_type = "analytic" if use_analytic_jacobian else ("numeric_central" if use_central_diff else "numeric_forward")

        # Pre-compute row mask
        if row_mask is not None:
            mask_bool = [bool(m) for m in row_mask]
            if len(mask_bool) != 6:
                raise ValueError("row_mask must have length 6")
        else:
            mask_bool = None

        for it in range(1, self.max_iters + 1):
            # Compute current pose - use standalone FK to avoid circular dependency
            if target_link is None:
                fk = forward_kinematics(self.model, q.tolist(), backend='numpy', return_end=True)
            else:
                # Partial FK: reuse Jacobian solver's helper (or replicate minimal logic)
                fk = self.jacobian_solver._fk_until(q, target_link) if hasattr(
                    self.jacobian_solver, '_fk_until') else forward_kinematics(self.model, q.tolist(), backend='numpy', return_end=True)
            if isinstance(fk, np.ndarray):
                R_current = fk[:3, :3]
                p_current = fk[:3, 3]
            else:  # List format
                R_current = np.array([row[:3] for row in fk[:3]], dtype=np.float64)
                p_current = np.array([fk[0][3], fk[1][3], fk[2][3]], dtype=np.float64)

            # Compute errors
            pos_err = p_target - p_current
            ori_err = rotation_error(R_current, R_target)

            pos_err_norm = np.linalg.norm(pos_err)
            ori_err_norm = np.linalg.norm(ori_err)

            # Weighted error vector
            full_err = np.concatenate([pos_weight * pos_err, ori_weight * ori_err])
            if mask_bool is not None:
                err = full_err[mask_bool]
            else:
                err = full_err
            err_norm = np.linalg.norm(err)
            
            # Track best solution
            if err_norm < best_err:
                best_err = err_norm
                best_q = q.copy()
                best_pos_err = pos_err_norm
                best_ori_err = ori_err_norm
            
            # Check convergence
            if pos_err_norm < self.pos_tol and ori_err_norm < self.ori_tol:
                # Optional refinement phase for tighter residuals
                if refine:
                    r_pos_tol = refine_pos_tol or (self.pos_tol * 0.2)
                    r_ori_tol = refine_ori_tol or (self.ori_tol * 0.2)
                    q_ref = q.copy()
                    for _r in range(refine_iters):
                        fk_r = forward_kinematics(self.model, q_ref.tolist(), backend='numpy', return_end=True)
                        if isinstance(fk_r, np.ndarray):
                            R_r = fk_r[:3, :3]; p_r = fk_r[:3, 3]
                        else:
                            R_r = np.array([row[:3] for row in fk_r[:3]], dtype=np.float64)
                            p_r = np.array([fk_r[0][3], fk_r[1][3], fk_r[2][3]], dtype=np.float64)
                        p_err_r = p_target - p_r
                        o_err_r = rotation_error(R_r, R_target)
                        if np.linalg.norm(p_err_r) < r_pos_tol and np.linalg.norm(o_err_r) < r_ori_tol:
                            q = q_ref
                            pos_err_norm = np.linalg.norm(p_err_r)
                            ori_err_norm = np.linalg.norm(o_err_r)
                            err = np.concatenate([pos_weight * p_err_r, ori_weight * o_err_r])
                            err_norm = np.linalg.norm(err)
                            break
                        # Always use analytic Jacobian for refinement & pseudoinverse
                        J_ref = self.jacobian_solver.solve(q_ref, method="analytic", target_link=target_link)
                        if pos_weight != 1.0:
                            J_ref[:3, :] *= pos_weight
                        if ori_weight != 1.0:
                            J_ref[3:6, :] *= ori_weight
                        # small damping
                        dq = self._solve_pinv(J_ref, np.concatenate([pos_weight * p_err_r, ori_weight * o_err_r]), self.min_damping)
                        # Limit very large jumps in refine
                        dq_norm = np.linalg.norm(dq)
                        if dq_norm > 0.2:
                            dq *= 0.2 / dq_norm
                        q_ref += dq
                        q_ref = self._apply_joint_limits(q_ref)
                    q = q_ref
                return {
                    "q": q.tolist(),
                    "success": True,
                    "iters": it,
                    "err_norm": float(err_norm),
                    "pos_err": float(pos_err_norm),
                    "ori_err": float(ori_err_norm),
                    "method": method,
                    "jacobian": "analytic" if use_analytic_jacobian else ("numeric_central" if use_central_diff else "numeric_forward"),
                }
            
            # Compute Jacobian using solver
            if use_analytic_jacobian:
                J = self.jacobian_solver.solve(q, method="analytic", target_link=target_link)
                jac_type = "analytic"
            else:
                J = self.jacobian_solver.solve(
                    q, 
                    method="numeric",
                    use_central_diff=use_central_diff,
                    target_link=target_link,
                )
                jac_type = "numeric_central" if use_central_diff else "numeric_forward"

            if pos_weight != 1.0:
                J[:3, :] *= pos_weight
            if ori_weight != 1.0:
                J[3:6, :] *= ori_weight
            if mask_bool is not None:
                J_eff = J[mask_bool, :]
            else:
                J_eff = J

            # Damping (used by dls/pinv)
            if adaptive_damping:
                damping = self._compute_adaptive_damping(J_eff, pos_err_norm, ori_err_norm)
            else:
                damping = (self.min_damping + self.max_damping) / 2

            # Solve per method
            if method == "dls":
                dq = self._solve_dls(J_eff, err, damping)
            elif method == "pinv":
                dq = self._solve_pinv(J_eff, err, damping)
            else:  # transpose
                # Jacobian Transpose 方法
                # 使用自适应增益：alpha = ||err||² / ||J @ J.T @ err||²
                if transpose_gain is not None:
                    alpha = transpose_gain
                else:
                    # 自适应增益计算（更稳定的收敛）
                    J_err = J.T @ err
                    JJt_err = J @ J_err
                    err_norm_sq = np.dot(err, err)
                    JJt_err_norm_sq = np.dot(JJt_err, JJt_err)
                    if JJt_err_norm_sq > 1e-12:
                        alpha_raw = err_norm_sq / JJt_err_norm_sq
                    else:
                        # 回退到固定增益
                        alpha_raw = 0.01
                    # 限制 alpha 范围避免步长过大
                    alpha = np.clip(alpha_raw, 0.001, 0.5)
                dq = alpha * (J_eff.T @ err)

            # Nullspace redundancy handling (only if more joints than task rows and gain>0)
            if nullspace_gain > 0 and self.n > J_eff.shape[0]:
                # Compute pseudoinverse of effective task Jacobian
                try:
                    U_ns, S_ns, Vt_ns = np.linalg.svd(J_eff, full_matrices=False)
                    S_inv_ns = np.array([1/s if s > 1e-9 else 0.0 for s in S_ns])
                    J_pinv_eff = (Vt_ns.T * S_inv_ns) @ U_ns.T
                    N = np.eye(self.n) - J_pinv_eff @ J_eff
                    if joint_centering:
                        centers = []
                        for js in self.model._actuated:
                            lo, hi = -1.0, 1.0
                            if js.limit:
                                if js.limit[0] is not None:
                                    lo = js.limit[0]
                                if js.limit[1] is not None:
                                    hi = js.limit[1]
                            centers.append(0.5 * (lo + hi))
                        centers = np.asarray(centers)
                        delta_center = centers - q
                        if joint_center_weights is not None and len(joint_center_weights) == self.n:
                            w = np.asarray(joint_center_weights)
                            delta_center = delta_center * w
                        dq_sec = joint_center_gain * delta_center
                    else:
                        dq_sec = np.zeros(self.n)
                    dq += nullspace_gain * (N @ dq_sec)
                except Exception:
                    pass  # fallback ignore

            # Step scaling (transpose 方法的 alpha 已经是最优步长，不需要额外缩放)
            if method != "transpose" and adaptive_step:
                step = self._compute_adaptive_step(pos_err_norm, ori_err_norm)
            else:
                step = 1.0 if method == "transpose" else self.base_step

            dq_step = step * dq
            # 添加步长限制（与 Torch solver 一致）
            dq_norm = np.linalg.norm(dq_step)
            if dq_norm > max_step_norm:
                dq_step = dq_step * (max_step_norm / (dq_norm + 1e-15))
            
            # Update joint angles
            q += dq_step
            
            # Apply joint limits
            q = self._apply_joint_limits(q)
        
        # Return best solution found
        # 失败：返回迭代中最优残差对应的 pos/ori 误差（若未更新保持最后一次计算）
        if not np.isfinite(best_pos_err) or not np.isfinite(best_ori_err):
            fk_best = forward_kinematics(self.model, best_q.tolist(), backend='numpy', return_end=True)
            if isinstance(fk_best, np.ndarray):
                R_best = fk_best[:3, :3]; p_best = fk_best[:3, 3]
            else:
                R_best = np.array([row[:3] for row in fk_best[:3]], dtype=np.float64)
                p_best = np.array([fk_best[0][3], fk_best[1][3], fk_best[2][3]], dtype=np.float64)
            best_pos_err = float(np.linalg.norm(p_target - p_best))
            best_ori_err = float(np.linalg.norm(rotation_error(R_best, R_target)))
        return {
            "q": best_q.tolist(),
            "success": False,
            "iters": self.max_iters,
            "err_norm": float(best_err),
            "method": method,
            "jacobian": jac_type,
            "pos_err": float(best_pos_err),
            "ori_err": float(best_ori_err),
        }

    def _solve_dls(self, J: np.ndarray, err: np.ndarray, damping: float) -> np.ndarray:
        """Solve damped least squares: dq = J^T (J J^T + λ²I)^{-1} err.

        :param J: 6×n Jacobian matrix.
        :param err: 6 error vector.
        :param damping: damping factor λ.
        :return: n joint velocity vector.
        """
        # A = J @ J^T + λ²I (6×6 matrix)
        A = J @ J.T + (damping ** 2) * np.eye(6, dtype=np.float64)
        
        # Solve A @ y = err for y
        y = np.linalg.solve(A, err)
        
        # dq = J^T @ y
        dq = J.T @ y
        
        return dq

    def _compute_adaptive_damping(self, J: np.ndarray, pos_err: float, ori_err: float) -> float:
        """Adaptive damping based on condition number and current error."""
        JJt = J @ J.T
        try:
            eigvals = np.linalg.eigvalsh(JJt)
            s_max = np.sqrt(np.max(eigvals))
            s_min = np.sqrt(np.max(np.min(eigvals), 1e-12))
            cond = s_max / s_min
        except Exception:
            cond = 100.0
        err = pos_err + 0.5 * ori_err
        if cond > 200 or err > 0.05:
            return self.max_damping
        elif cond < 30 and err < 0.01:
            return self.min_damping
        else:
            return (self.min_damping + self.max_damping) * 0.5

    def _compute_adaptive_step(self, pos_err: float, ori_err: float) -> float:
        """Compute adaptive step size based on error magnitude.
        
        More aggressive for large errors to escape local minima,
        matching JS solver behavior for better workspace boundary handling.
        """
        norm_pos_err = pos_err / 0.01
        norm_ori_err = ori_err / 0.087
        max_norm_err = max(norm_pos_err, norm_ori_err)
        if max_norm_err > 2.0:
            # Large error: be more aggressive (like JS 0.8, not 0.6)
            return self.base_step * 1.0
        elif max_norm_err > 1.0:
            return self.base_step * 1.2
        elif max_norm_err > 0.5:
            return self.base_step * 1.0
        else:
            # Near convergence: smaller steps for precision
            return self.base_step * 0.5

    def _solve_pinv(self, J: np.ndarray, err: np.ndarray, damping: float) -> np.ndarray:
        try:
            U, S, Vt = np.linalg.svd(J, full_matrices=False)
        except np.linalg.LinAlgError:
            return self._solve_dls(J, err, damping)
        if damping > 0:
            S_inv = S / (S * S + damping * damping)
        else:
            tol = 1e-9 * max(J.shape)
            S_inv = np.array([1 / s if s > tol else 0.0 for s in S])
        return (Vt.T * S_inv) @ (U.T @ err)

    def _apply_joint_limits(self, q: np.ndarray) -> np.ndarray:
        """Clamp joint angles to limits.

        :param q: joint configuration (n,).
        :return: clamped configuration.
        """
        q_clamped = q.copy()
        for js in self.model._actuated:
            if js.limit is not None:
                if js.limit[0] is not None:
                    q_clamped[js.index] = max(js.limit[0], q_clamped[js.index])
                if js.limit[1] is not None:
                    q_clamped[js.index] = min(js.limit[1], q_clamped[js.index])
        return q_clamped
