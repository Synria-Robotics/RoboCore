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

import math
from typing import TYPE_CHECKING, Dict, Sequence
import numpy as np
from ..jacobian_utils.jacobian_solver_numpy import JacobianSolverNumPy
from ..fk_utils.fk_solver_numpy import FKSolverNumPy
from robocore.transform import rotation_error
from robocore.kinematics.fk import forward_kinematics
from robocore.kinematics.utils import ensure_batch, restore_single

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
        max_iters: int = 200,  # Increased from 100 for better convergence (93% success rate at n=100)
        pos_tol: float = 1e-3,  # Kept at 1e-3 (good balance)
        ori_tol: float = 1e-3,  # Kept at 1e-3 (good balance)
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
        self.n = model.num_chain_dof
        # Initialize FK and Jacobian solvers
        self.fk_solver = FKSolverNumPy(model)
        self.jacobian_solver = JacobianSolverNumPy(model)

    def solve(
        self,
        target_pose: np.ndarray,
        q0: np.ndarray,
        pos_weight: float = 1.0,
        ori_weight: float = 1.0,
        adaptive_damping: bool = True,
        adaptive_step: bool = False,
        use_central_diff: bool = True,
        use_analytic_jacobian: bool = True,
        method: str = "dls",
        transpose_gain: float | None = None,
        max_step_norm: float = 0.5,
        refine: bool = False,
        refine_iters: int = 10,
        refine_pos_tol: float | None = None,
        refine_ori_tol: float | None = None,
        target_link: str | None = None,
        row_mask: Sequence[int | bool] | None = None,
        nullspace_gain: float = 0.0,
        joint_centering: bool = True,
        joint_center_gain: float = 0.2,
        joint_center_weights: Sequence[float] | None = None,
    ) -> Dict[str, object] | Dict[str, list]:
        """Solve IK with selectable method (supports both single and batch).

        Supported methods:
          - dls: damped least squares (JJ^T regularization)
          - pinv: SVD pseudoinverse with Tikhonov damping
          - transpose: J^T * error with adaptive gain
        
        :param target_pose: target pose(s) [4, 4] or [B, 4, 4]
        :param q0: initial configuration(s) [n] or [B, n]
        :return: IK result dict (single) or dict with batch results (batch)
        """
        method = method.lower()
        if method not in ("dls", "pinv", "transpose"):
            raise ValueError(f"Unknown IK method '{method}'")

        target_pose = np.asarray(target_pose, dtype=np.float64)
        q0 = np.asarray(q0, dtype=np.float64)

        # Handle batch mode - normalize target_pose shape first
        if target_pose.ndim == 2:
            # Single 4x4 matrix -> reshape to [1, 4, 4]
            target_pose = target_pose.reshape(1, 4, 4)
        elif target_pose.ndim == 3:
            # Already batch format [B, 4, 4]
            pass
        else:
            raise ValueError(f"Expected target_pose with shape [4, 4] or [B, 4, 4], got {target_pose.shape}")

        # Now ensure batch format
        was_single_t = target_pose.shape[0] == 1
        q0, was_single_q = ensure_batch(q0)
        was_single = was_single_t and was_single_q

        if q0.shape[1] != self.n:
            raise ValueError(f"Expected q0 with {self.n} elements, got {q0.shape[1]}")

        if target_pose.shape[0] != q0.shape[0]:
            raise ValueError(f"Batch size mismatch: target_pose {target_pose.shape[0]} vs q0 {q0.shape[0]}")

        result = self._solve_batch(
            target_pose, q0,
            pos_weight=pos_weight,
            ori_weight=ori_weight,
            adaptive_damping=adaptive_damping,
            adaptive_step=adaptive_step,
            use_central_diff=use_central_diff,
            use_analytic_jacobian=use_analytic_jacobian,
            method=method,
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

        return restore_single(result, was_single)

    def _solve_batch(
        self,
        target_poses_batch: np.ndarray,
        q0_batch: np.ndarray,
        pos_weight: float = 1.0,
        ori_weight: float = 1.0,
        adaptive_damping: bool = True,
        adaptive_step: bool = True,
        use_central_diff: bool = True,
        use_analytic_jacobian: bool = False,
        method: str = "dls",
        transpose_gain: float | None = None,
        max_step_norm: float = 0.5,
        refine: bool = False,
        refine_iters: int = 10,
        refine_pos_tol: float | None = None,
        refine_ori_tol: float | None = None,
        target_link: str | None = None,
        row_mask: Sequence[int | bool] | None = None,
        nullspace_gain: float = 0.0,
        joint_centering: bool = True,
        joint_center_gain: float = 0.2,
        joint_center_weights: Sequence[float] | None = None,
    ) -> Dict[str, list]:
        """Solve IK for batch of configurations (vectorized).
        
        :param target_poses_batch: target poses [B, 4, 4]
        :param q0_batch: initial configurations [B, n]
        :return: dict with batch results (each field is a list of length B)
        """
        B = target_poses_batch.shape[0]
        n = self.n

        q = q0_batch.copy()
        best_q = q.copy()
        best_err = np.full(B, np.inf)
        success = np.zeros(B, dtype=bool)
        iters = np.zeros(B, dtype=np.int32)
        active = np.ones(B, dtype=bool)
        prev_err = np.full(B, np.inf)
        plateau = np.zeros(B, dtype=np.int32)
        q_initial = q.copy()  # Track initial configuration to detect large cumulative jumps

        p_target = target_poses_batch[:, :3, 3]
        R_target = target_poses_batch[:, :3, :3]
        eye6 = np.eye(6, dtype=np.float64)

        jac_method = "analytic" if use_analytic_jacobian else "numeric"

        for it in range(1, self.max_iters + 1):
            if not active.any():
                break

            act_idx = np.where(active)[0]
            q_act = q[act_idx]
            Ba = len(act_idx)

            # FK (batch)
            T_end = self.fk_solver.solve(q_act, return_end_only=True)
            if T_end.ndim == 2:
                T_end = T_end[np.newaxis, ...]
            p_end = T_end[:, :3, 3]
            R_end = T_end[:, :3, :3]

            # Errors
            p_err = p_target[act_idx] - p_end
            R_err_vec = self._batch_rotation_error(R_end, R_target[act_idx])
            pos_norm = np.linalg.norm(p_err, axis=1)
            ori_norm = np.linalg.norm(R_err_vec, axis=1)
            err_norm = pos_norm + ori_norm

            # Track best solution
            improved = err_norm < best_err[act_idx]
            best_err[act_idx] = np.where(improved, err_norm, best_err[act_idx])
            for i, idx in enumerate(act_idx):
                if improved[i]:
                    best_q[idx] = q_act[i]

            # Plateau detection
            delta = prev_err[act_idx] - err_norm
            plateau[act_idx] = np.where(delta < 1e-8, plateau[act_idx] + 1, 0)
            prev_err[act_idx] = err_norm

            # Convergence check
            conv_mask = (pos_norm < self.pos_tol) & (ori_norm < self.ori_tol)
            if conv_mask.any():
                g_idx = act_idx[conv_mask]
                success[g_idx] = True
                iters[g_idx] = it
                active[g_idx] = False

            if not active.any():
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
            Ba_sub = len(act_idx)

            # Jacobian (batch)
            J = self.jacobian_solver.solve(q_sub, method=jac_method, use_central_diff=use_central_diff)
            if J.ndim == 2:
                J = J[np.newaxis, ...]

            # Apply weights to Jacobian
            if pos_weight != 1.0:
                J[:, :3, :] *= pos_weight
            if ori_weight != 1.0:
                J[:, 3:6, :] *= ori_weight

            # Error vector [Ba_sub, 6]
            e = np.concatenate([pos_weight * p_err_sub, ori_weight * R_err_sub], axis=1)

            # Adaptive damping per sample
            if adaptive_damping:
                lam = np.full(Ba_sub, math.sqrt(self.min_damping * self.max_damping))
                for k in range(Ba_sub):
                    try:
                        S = np.linalg.svd(J[k], compute_uv=False)
                        if len(S) == 0:
                            cond = 100.0
                        else:
                            cond = S[0] / max(S[-1], 1e-12)
                    except (np.linalg.LinAlgError, IndexError):
                        cond = 100.0
                    err_combo = pos_norm_sub[k] + 0.5 * ori_norm_sub[k]
                    if cond > 200 or err_combo > 0.05:
                        lam[k] = self.max_damping
                    elif cond < 30 and err_combo < 0.01:
                        lam[k] = self.min_damping
                    if plateau_sub[k] >= 4:
                        lam[k] = min(lam[k] * 2.0, self.max_damping * 2)
            else:
                lam = np.full(Ba_sub, (self.min_damping + self.max_damping) / 2)

            # Solve per sample
            dq = np.zeros((Ba_sub, n), dtype=np.float64)
            for k in range(Ba_sub):
                Jk = J[k]
                ek = e[k]
                qk = q_sub[k]

                # Project Jacobian to handle joint limits: zero out columns for joints that would be blocked
                Jk_projected = self._project_jacobian_for_limits(Jk, qk, ek, lam[k], method)

                if method == "dls":
                    dq[k] = self._solve_dls(Jk_projected, ek, lam[k])
                elif method == "pinv":
                    dq[k] = self._solve_pinv(Jk_projected, ek, lam[k])
                else:  # transpose
                    if transpose_gain is not None:
                        alpha = transpose_gain
                    else:
                        J_err = Jk_projected.T @ ek
                        JJt_err = Jk_projected @ J_err
                        alpha = np.dot(ek, ek) / (np.dot(JJt_err, JJt_err) + 1e-12)
                        alpha = np.clip(alpha, 0.001, 0.5)
                    dq[k] = alpha * (Jk_projected.T @ ek)

            # Adaptive step
            if adaptive_step and method != "transpose":
                m = np.maximum(pos_norm_sub / 0.01, ori_norm_sub / 0.087)
                step = np.where(m > 2.0, 1.0, np.where(m > 1.0, 1.2, np.where(m > 0.5, 1.0, 0.6)))
                step = np.where(plateau_sub >= 8, step * 0.5, step)
                dq = dq * step[:, np.newaxis]

            # Step norm clipping
            if max_step_norm is not None and max_step_norm > 0:
                dq_norm = np.linalg.norm(dq, axis=1, keepdims=True)
                scale = np.clip(max_step_norm / (dq_norm + 1e-12), 0, 1)
                dq = dq * scale

            # Update with joint limits - but also detect large jumps that might indicate discontinuity
            q_new = q_sub + dq
            for js in self.model._chain_dof_list:
                j = js.index
                if js.limit_lower is not None:
                    q_new[:, j] = np.maximum(q_new[:, j], js.limit_lower)
                if js.limit_upper is not None:
                    q_new[:, j] = np.minimum(q_new[:, j], js.limit_upper)

            # Detect and prevent large jumps near limits (indicates discontinuity)
            for k in range(Ba_sub):
                orig_idx = act_idx[k]
                for js in self.model._chain_dof_list:
                    j = js.index
                    if js.limit_lower is not None and js.limit_upper is not None:
                        joint_range = js.limit_upper - js.limit_lower
                        dist_to_lower = q_sub[k, j] - js.limit_lower
                        dist_to_upper = js.limit_upper - q_sub[k, j]
                        threshold = joint_range * 0.1  # 10% of range - more aggressive

                        # Check both single-step jump and cumulative jump from initial
                        step_jump = abs(q_new[k, j] - q_sub[k, j])
                        cumulative_jump = abs(q_new[k, j] - q_initial[orig_idx, j])

                        # If near limit and jump is large, it might be a discontinuity
                        if (dist_to_lower < threshold or dist_to_upper < threshold):
                            # Check single-step jump
                            if step_jump > joint_range * 0.15:  # 15% of range per step
                                max_step = joint_range * 0.05  # Max 5% of range per step
                                if q_new[k, j] > q_sub[k, j] + max_step:
                                    q_new[k, j] = q_sub[k, j] + max_step
                                elif q_new[k, j] < q_sub[k, j] - max_step:
                                    q_new[k, j] = q_sub[k, j] - max_step
                            # Also check cumulative jump - if too large, clamp to initial + small step
                            elif cumulative_jump > joint_range * 0.4:  # 40% of range total
                                # Allow only small deviation from initial when near limit
                                max_deviation = joint_range * 0.1
                                if q_new[k, j] > q_initial[orig_idx, j] + max_deviation:
                                    q_new[k, j] = q_initial[orig_idx, j] + max_deviation
                                elif q_new[k, j] < q_initial[orig_idx, j] - max_deviation:
                                    q_new[k, j] = q_initial[orig_idx, j] - max_deviation

                            # Re-apply limits
                            if js.limit_lower is not None:
                                q_new[k, j] = max(q_new[k, j], js.limit_lower)
                            if js.limit_upper is not None:
                                q_new[k, j] = min(q_new[k, j], js.limit_upper)

            q[act_idx] = q_new

        # Use best solution for non-converged
        remaining = np.where(active)[0]
        if len(remaining) > 0:
            iters[remaining] = self.max_iters
            q[remaining] = best_q[remaining]

        # Final errors
        T_final = self.fk_solver.solve(q, return_end_only=True)
        if T_final.ndim == 2:
            T_final = T_final[np.newaxis, ...]
        p_err_final = np.linalg.norm(p_target - T_final[:, :3, 3], axis=1)
        ori_err_final = np.linalg.norm(self._batch_rotation_error(T_final[:, :3, :3], R_target), axis=1)

        jac_type = "analytic" if use_analytic_jacobian else ("numeric_central" if use_central_diff else "numeric_forward")

        return {
            'q': [q[i].tolist() for i in range(B)],
            'success': success.tolist(),
            'iters': iters.tolist(),
            'err_norm': best_err.tolist(),
            'method': [method] * B,
            'jacobian': [jac_type] * B,
            'pos_err': p_err_final.tolist(),
            'ori_err': ori_err_final.tolist(),
        }

    @staticmethod
    def _batch_rotation_error(R_current: np.ndarray, R_target: np.ndarray) -> np.ndarray:
        """Compute rotation error for batch.
        
        :param R_current: [B, 3, 3]
        :param R_target: [B, 3, 3]
        :return: [B, 3] axis-angle error vectors
        """
        B = R_current.shape[0]
        result = np.zeros((B, 3), dtype=np.float64)
        for i in range(B):
            result[i] = rotation_error(R_current[i], R_target[i])
        return result

    def _solve_single(
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
        max_step_norm: float = 0.5,
        refine: bool = False,
        refine_iters: int = 10,
        refine_pos_tol: float | None = None,
        refine_ori_tol: float | None = None,
        target_link: str | None = None,
        row_mask: Sequence[int | bool] | None = None,
        nullspace_gain: float = 0.0,
        joint_centering: bool = True,
        joint_center_gain: float = 0.2,
        joint_center_weights: Sequence[float] | None = None,
    ) -> Dict[str, object]:
        """Solve IK for single configuration (original implementation)."""
        method = method.lower()
        if method not in ("dls", "pinv", "transpose"):
            raise ValueError(f"Unknown IK method '{method}'")
        
        if q0.shape[0] != self.n:
            raise ValueError(f"Expected q0 with {self.n} elements, got {q0.shape[0]}")
        
        # Extract target position and rotation
        R_target = target_pose[:3, :3]
        p_target = target_pose[:3, 3]
        
        q = q0.copy()
        q_initial = q.copy()  # Track initial configuration to detect large cumulative jumps
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
                fk = forward_kinematics(self.model, q.tolist(), return_end=True)
            else:
                # Partial FK: reuse Jacobian solver's helper (or replicate minimal logic)
                fk = self.jacobian_solver._fk_until(q, target_link) if hasattr(
                    self.jacobian_solver, '_fk_until') else forward_kinematics(self.model, q.tolist(), return_end=True)
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
                        fk_r = forward_kinematics(self.model, q_ref.tolist(), return_end=True)
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

            # Project Jacobian to handle joint limits before solving
            J_eff_projected = self._project_jacobian_for_limits(J_eff, q, err, damping, method)

            # Solve per method
            if method == "dls":
                dq = self._solve_dls(J_eff_projected, err, damping)
            elif method == "pinv":
                dq = self._solve_pinv(J_eff_projected, err, damping)
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
                dq = alpha * (J_eff_projected.T @ err)

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
                        for js in self.model._chain_dof_list:
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
            
            # Optional step norm clipping (only when adaptive_step is True)
            # When adaptive_step=False, user controls step size via base_step (like pytorch_kinematics lr)
            # max_step_norm is only applied when explicitly needed for stability
            if adaptive_step and max_step_norm is not None and max_step_norm > 0:
                dq_norm = np.linalg.norm(dq_step)
                if dq_norm > max_step_norm:
                    dq_step = dq_step * (max_step_norm / (dq_norm + 1e-15))
            
            # Update joint angles
            q_new = q + dq_step
            
            # Apply joint limits
            q_new = self._apply_joint_limits(q_new)

            # Detect and prevent large jumps near limits (indicates discontinuity)
            for js in self.model._chain_dof_list:
                j = js.index
                if js.limit_lower is not None and js.limit_upper is not None:
                    joint_range = js.limit_upper - js.limit_lower
                    dist_to_lower = q[j] - js.limit_lower
                    dist_to_upper = js.limit_upper - q[j]
                    threshold = joint_range * 0.1  # 10% of range - more aggressive

                    # Check both single-step jump and cumulative jump from initial
                    step_jump = abs(q_new[j] - q[j])
                    cumulative_jump = abs(q_new[j] - q_initial[j])

                    # If near limit and jump is large, it might be a discontinuity
                    if (dist_to_lower < threshold or dist_to_upper < threshold):
                        # Check single-step jump
                        if step_jump > joint_range * 0.15:  # 15% of range per step
                            max_step = joint_range * 0.05  # Max 5% of range per step
                            if q_new[j] > q[j] + max_step:
                                q_new[j] = q[j] + max_step
                            elif q_new[j] < q[j] - max_step:
                                q_new[j] = q[j] - max_step
                        # Also check cumulative jump - if too large, clamp to initial + small step
                        elif cumulative_jump > joint_range * 0.4:  # 40% of range total
                            # Allow only small deviation from initial when near limit
                            max_deviation = joint_range * 0.1
                            if q_new[j] > q_initial[j] + max_deviation:
                                q_new[j] = q_initial[j] + max_deviation
                            elif q_new[j] < q_initial[j] - max_deviation:
                                q_new[j] = q_initial[j] - max_deviation

                        # Re-apply limits
                        if js.limit_lower is not None:
                            q_new[j] = max(q_new[j], js.limit_lower)
                        if js.limit_upper is not None:
                            q_new[j] = min(q_new[j], js.limit_upper)

            q = q_new
        
        # Return best solution found
        # 失败：返回迭代中最优残差对应的 pos/ori 误差（若未更新保持最后一次计算）
        if not np.isfinite(best_pos_err) or not np.isfinite(best_ori_err):
            fk_best = forward_kinematics(self.model, best_q.tolist(), return_end=True)
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

    def _project_jacobian_for_limits(self, J: np.ndarray, q: np.ndarray, err: np.ndarray,
                                     damping: float, method: str = "dls", max_iter: int = 3) -> np.ndarray:
        """Project Jacobian to handle joint limits by zeroing out columns for blocked joints.
        
        A joint is considered "blocked" if it's at a limit and the computed dq would
        try to move it further toward/outside the limit.
        
        This prevents DLS from trying to use blocked joints to compensate for errors,
        which causes jumps in other joints.
        
        :param J: Jacobian matrix [6, n]
        :param q: Current joint configuration [n]
        :param err: Error vector [6]
        :param damping: Damping factor
        :param method: IK method
        :param max_iter: Maximum iterations for projection
        :return: Projected Jacobian matrix [6, n]
        """
        J_projected = J.copy()
        eps = 1e-6  # Small threshold to detect "at limit"
        limit_threshold_ratio = 0.01  # Consider joint "near limit" if within 1% of range

        # First pass: detect joints that are at or very close to limits
        initially_blocked = np.zeros(len(q), dtype=bool)
        for js in self.model._chain_dof_list:
            j = js.index
            if js.limit_lower is not None and js.limit_upper is not None:
                joint_range = js.limit_upper - js.limit_lower
                dist_to_lower = q[j] - js.limit_lower
                dist_to_upper = js.limit_upper - q[j]
                threshold = joint_range * limit_threshold_ratio

                # If very close to limit, check if Jacobian suggests movement toward limit
                if dist_to_lower < max(eps, threshold):
                    # Estimate dq direction for this joint: J^T @ err gives approximate direction
                    dq_estimate = J[:, j].T @ err
                    if dq_estimate < 0:  # Would try to go lower
                        initially_blocked[j] = True
                elif dist_to_upper < max(eps, threshold):
                    dq_estimate = J[:, j].T @ err
                    if dq_estimate > 0:  # Would try to go higher
                        initially_blocked[j] = True

        # Zero out initially blocked joints
        J_projected[:, initially_blocked] = 0.0

        # Iteratively refine: compute dq and check for additional blocked joints
        for _ in range(max_iter):
            # Compute dq with current projection
            if method == "dls":
                dq = self._solve_dls(J_projected, err, damping)
            elif method == "pinv":
                dq = self._solve_pinv(J_projected, err, damping)
            else:
                # For transpose, use simple estimate
                dq = J_projected.T @ err
                dq_norm = np.linalg.norm(dq)
                if dq_norm > 0:
                    dq = dq * min(0.1 / dq_norm, 1.0)

            # Check which joints would be blocked
            newly_blocked = np.zeros(len(q), dtype=bool)
            for js in self.model._chain_dof_list:
                j = js.index
                if initially_blocked[j]:
                    continue  # Already blocked

                if js.limit_lower is not None and js.limit_upper is not None:
                    q_candidate = q[j] + dq[j]

                    # Check if joint would exceed limits
                    if q_candidate < js.limit_lower - eps:
                        newly_blocked[j] = True
                    elif q_candidate > js.limit_upper + eps:
                        newly_blocked[j] = True
                    # Also check if at limit and trying to push further
                    elif q[j] <= js.limit_lower + eps and dq[j] < -eps:
                        newly_blocked[j] = True
                    elif q[j] >= js.limit_upper - eps and dq[j] > eps:
                        newly_blocked[j] = True

            if not newly_blocked.any():
                # No more blocked joints
                break

            # Zero out columns for newly blocked joints
            J_projected[:, newly_blocked] = 0.0
            initially_blocked = initially_blocked | newly_blocked

        return J_projected

    def _apply_joint_limits(self, q: np.ndarray) -> np.ndarray:
        """Clamp joint angles to limits.

        :param q: joint configuration (n,).
        :return: clamped configuration.
        """
        q_clamped = q.copy()
        for js in self.model._chain_dof_list:
            if js.limit_lower is not None and js.limit_upper is not None:
                if js.limit_lower is not None:
                    q_clamped[js.index] = max(js.limit_lower, q_clamped[js.index])
                if js.limit_upper is not None:
                    q_clamped[js.index] = min(js.limit_upper, q_clamped[js.index])
        return q_clamped
