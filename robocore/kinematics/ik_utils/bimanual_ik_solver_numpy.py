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
        if q0_left is None:
            q0_left = [0.0] * self.left.num_dof()
        if q0_right is None:
            q0_right = [0.0] * self.right.num_dof()

        tgt_left = target_left.tolist() if hasattr(target_left, 'tolist') else target_left
        tgt_right = target_right.tolist() if hasattr(target_right, 'tolist') else target_right

        res_left = (inverse_kinematics(self.left, tgt_left, q0_left, backend='numpy', **ik_kwargs)
                    if tgt_left is not None else {'q': q0_left, 'success': True})
        res_right = (inverse_kinematics(self.right, tgt_right, q0_right, backend='numpy', **ik_kwargs)
                     if tgt_right is not None else {'q': q0_right, 'success': True})

        return {
            'q_left': res_left.get('q', q0_left),
            'q_right': res_right.get('q', q0_right),
            'success_left': bool(res_left.get('success', False)),
            'success_right': bool(res_right.get('success', False)),
            'res_left': res_left,
            'res_right': res_right,
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
        res_left = (inverse_kinematics(self.left, tgt_left, q0_left, backend='numpy', **ik_kwargs)
                    if tgt_left is not None else {'q': q0_left, 'success': True})
        q_left = res_left.get('q', q0_left)

        # 2. 用左臂当前末端和 T_rel_grasp 计算右臂目标
        T_left_current = self.left.fk(q_left)['end']
        T_right_constrained = T_left_current @ T_rel_grasp if T_rel_grasp is not None else None

        res_right = (inverse_kinematics(self.right, T_right_constrained, q0_right, backend='numpy', **ik_kwargs)
                     if T_right_constrained is not None else {'q': q0_right, 'success': True})

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
              mirror_axis: str = 'x',
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
            # 1. 镜像位置
            pos_left = np.array(target_left)[0:3, 3]
            pos_right_mirrored = np.array([-pos_left[0], pos_left[1], pos_left[2]])

            # 2. 镜像旋转
            R_left_current = np.array(target_left)[0:3, 0:3]
            R_left_initial = np.array(T_left_initial)[0:3, 0:3]
            R_delta_left = R_left_current @ R_left_initial.T
            M_mirror = np.diag([-1, 1, 1])
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
