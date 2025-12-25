"""Independent bimanual IK solver (NumPy).

Provides a thin wrapper that calls single-arm RobotModel.ik for left and right
targets independently and returns a combined result dict.
"""
from __future__ import annotations
from typing import Optional, Sequence, Dict, Any

from robocore.modeling.robot_model import RobotModel
from robocore.kinematics.ik import inverse_kinematics


class BiIndependentIKSolverNumpy:
    """Independent bimanual IK solver (NumPy-oriented).

    Usage:
        solver = BiIndependentIKSolverNumpy(left_model, right_model)
        res = solver.solve(target_left, target_right, q0_left, q0_right, **ik_kwargs)
    """

    def __init__(self, left_model: RobotModel, right_model: RobotModel):
        self.left = left_model
        self.right = right_model

    def solve(self,
              target_left: Optional[Sequence[Sequence[float]]],
              target_right: Optional[Sequence[Sequence[float]]],
              q0_left: Optional[Sequence[float]] = None,
              q0_right: Optional[Sequence[float]] = None,
              **ik_kwargs) -> Dict[str, Any]:
        """Solve bimanual IK independently.
        
        :param target_left: Left target pose(s) - 4x4 or [B, 4, 4]
        :param target_right: Right target pose(s) - 4x4 or [B, 4, 4]
        :param q0_left: Initial left configuration (optional, used as base for strategies)
        :param q0_right: Initial right configuration (optional, used as base for strategies)
        :param ik_kwargs: IK solver parameters (including num_initial_guesses, initial_guess_strategy, etc.)
        :return: Result dict with q_left, q_right, success_left, success_right, etc.
        """
        # q0 is optional now - inverse_kinematics will generate initial guesses if not provided
        # Check if batch mode
        import numpy as np
        is_batch = False
        if target_left is not None:
            tgt_left_arr = np.asarray(target_left)
            is_batch = tgt_left_arr.ndim == 3 and tgt_left_arr.shape[1:] == (4, 4)
        elif target_right is not None:
            tgt_right_arr = np.asarray(target_right)
            is_batch = tgt_right_arr.ndim == 3 and tgt_right_arr.shape[1:] == (4, 4)

        tgt_left = target_left.tolist() if hasattr(target_left, 'tolist') else target_left
        tgt_right = target_right.tolist() if hasattr(target_right, 'tolist') else target_right

        res_left = inverse_kinematics(self.left, tgt_left, q0_left, **ik_kwargs)
        res_right = inverse_kinematics(self.right, tgt_right, q0_right, **ik_kwargs)

        # Helper to pick the best candidate from a list based on pose error
        def _select_best(res, q_fallback, dof):
            import numpy as np
            if isinstance(res, list) and len(res) > 0:
                # Choose entry with minimum (pos_err + ori_err); fall back to first
                best_idx = 0
                best_score = None
                for i, r in enumerate(res):
                    pos_err = float(r.get('pos_err', 0.0))
                    ori_err = float(r.get('ori_err', 0.0))
                    score = abs(pos_err) + abs(ori_err)
                    if best_score is None or score < best_score:
                        best_score = score
                        best_idx = i
                return res[best_idx]
            elif isinstance(res, dict):
                return res
            else:
                # Unexpected type – return a simple dummy dict
                return {'q': q_fallback if q_fallback is not None else [0.0] * dof,
                        'success': False, 'pos_err': np.inf, 'ori_err': np.inf}

        # Handle batch mode (returns list of dicts)
        if is_batch and isinstance(res_left, list) and isinstance(res_right, list):
            # Return list of combined results
            results = []
            for r_l, r_r in zip(res_left, res_right):
                best_l = _select_best(r_l, q0_left, self.left.num_chain_dof) if isinstance(r_l, list) else r_l
                best_r = _select_best(r_r, q0_right, self.right.num_chain_dof) if isinstance(r_r, list) else r_r
                results.append({
                    'q_left': best_l.get('q', q0_left if q0_left is not None else [0.0] * self.left.num_chain_dof),
                    'q_right': best_r.get('q', q0_right if q0_right is not None else [0.0] * self.right.num_chain_dof),
                    'success_left': bool(best_l.get('success', False)),
                    'success_right': bool(best_r.get('success', False)),
                    'res_left': best_l,
                    'res_right': best_r,
                })
            return results

        # Single mode (returns dict) – select best candidate if lists were returned
        best_left = _select_best(res_left, q0_left, self.left.num_chain_dof)
        best_right = _select_best(res_right, q0_right, self.right.num_chain_dof)

        return {
            'q_left': best_left.get('q', q0_left if q0_left is not None else [0.0] * self.left.num_chain_dof),
            'q_right': best_right.get('q', q0_right if q0_right is not None else [0.0] * self.right.num_chain_dof),
            'success_left': bool(best_left.get('success', False)),
            'success_right': bool(best_right.get('success', False)),
            'res_left': best_left,
            'res_right': best_right,
        }


class BiRelativeIKSolverNumpy(BiIndependentIKSolverNumpy):
    """Relative bimanual IK solver (NumPy).

    Solves two-arm IK with relative constraints by composing per-arm IK and
    using a relative Jacobian in higher-level loops (to be extended).
    """

    def solve(self,
              target_left: Optional[Sequence[Sequence[float]]],
              target_right: Optional[Sequence[Sequence[float]]],
              q0_left: Optional[Sequence[float]] = None,
              q0_right: Optional[Sequence[float]] = None,
              constraint_type: str = 'pose',
              T_rel_grasp=None,
              **ik_kwargs) -> Dict[str, Any]:
        """
        :param target_left: Left target pose
        :param target_right: Right target pose
        :param q0_left: Initial left configuration
        :param q0_right: Initial right configuration
        :param constraint_type: 'pose'|'position'|'orientation'
        :param T_rel_grasp: 相对抓取变换（左^-1 @ 右）
        :return: Result dict
        """
        # 1. 先解左臂
        tgt_left = target_left.tolist() if hasattr(target_left, 'tolist') else target_left
        res_left = inverse_kinematics(self.left, tgt_left, q0_left, **ik_kwargs)
        q_left = res_left.get('q', q0_left if q0_left is not None else [0.0] * self.left.num_chain_dof)

        # 2. 用左臂当前末端和 T_rel_grasp 计算右臂目标
        T_left_current = self.left.fk(q_left)['end']
        T_right_constrained = T_left_current @ T_rel_grasp if T_rel_grasp is not None else None

        res_right = inverse_kinematics(self.right, T_right_constrained, q0_right, **ik_kwargs)

        return {
            'q_left': q_left,
            'q_right': res_right.get('q', q0_right),
            'success_left': bool(res_left.get('success', False)),
            'success_right': bool(res_right.get('success', False)),
            'res_left': res_left,
            'res_right': res_right,
        }


class BiMirrorIKSolverNumpy(BiIndependentIKSolverNumpy):
    """Mirror-symmetric bimanual IK solver (NumPy).

    Uses independent IK but allows providing only one target by mirroring.
    """

    def solve(self,
              target_left: Optional[Sequence[Sequence[float]]],
              target_right: Optional[Sequence[Sequence[float]]],
              q0_left: Optional[Sequence[float]] = None,
              q0_right: Optional[Sequence[float]] = None,
              mirror_axis: str = 'y',
              T_left_initial=None,
              T_right_initial=None,
              **ik_kwargs) -> Dict[str, Any]:
        """
        :param target_left: Left target pose
        :param target_right: Right target pose
        :param q0_left: Initial left configuration
        :param q0_right: Initial right configuration
        :param mirror_axis: Mirror axis 'x'|'y'|'z'
        :param T_left_initial: 左臂初始参考位姿
        :param T_right_initial: 右臂初始参考位姿
        :return: Result dict
        """
        import numpy as np
        # 只处理左臂拖动，右臂镜像
        if target_left is not None and T_left_initial is not None and T_right_initial is not None:
            # 1. 镜像位置（根据初始左右手位置，对称面是 Y=0，镜像 Y 轴）
            pos_left = np.array(target_left)[0:3, 3]
            if mirror_axis == 'x':
                pos_right_mirrored = np.array([-pos_left[0], pos_left[1], pos_left[2]])
                M_mirror = np.diag([-1, 1, 1])
            elif mirror_axis == 'y':
                pos_right_mirrored = np.array([pos_left[0], -pos_left[1], pos_left[2]])
                M_mirror = np.diag([1, -1, 1])
            elif mirror_axis == 'z':
                pos_right_mirrored = np.array([pos_left[0], pos_left[1], -pos_left[2]])
                M_mirror = np.diag([1, 1, -1])
            else:
                raise ValueError(f"Invalid mirror_axis: {mirror_axis}, must be 'x', 'y', or 'z'")

            # 2. 镜像旋转
            R_left_current = np.array(target_left)[0:3, 0:3]
            R_left_initial = np.array(T_left_initial)[0:3, 0:3]
            R_delta_left = R_left_current @ R_left_initial.T
            R_delta_right = M_mirror @ R_delta_left @ M_mirror.T
            R_right_initial = np.array(T_right_initial)[0:3, 0:3]
            R_right_mirrored = R_delta_right @ R_right_initial

            # 构造右臂目标
            T_right_mirrored = np.eye(4)
            T_right_mirrored[0:3, 3] = pos_right_mirrored
            T_right_mirrored[0:3, 0:3] = R_right_mirrored
            tgt_right = T_right_mirrored
        else:
            tgt_right = target_right

        tgt_left = target_left

        return super().solve(tgt_left, tgt_right, q0_left, q0_right, **ik_kwargs)
