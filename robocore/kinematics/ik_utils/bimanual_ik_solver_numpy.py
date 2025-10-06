"""Independent bimanual IK solver (NumPy).

Provides a thin wrapper that calls single-arm RobotModel.ik for left and right
targets independently and returns a combined result dict.
"""
from __future__ import annotations
from typing import Optional, Sequence, Dict, Any

from robocore.modeling.robot_model import RobotModel


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
        # Delegate to single-arm IK for each arm separately.
        if q0_left is None:
            q0_left = [0.0] * self.left.num_dof()
        if q0_right is None:
            q0_right = [0.0] * self.right.num_dof()

        res_left = {'q': q0_left, 'success': True, 'pos_err': 0.0, 'ori_err': 0.0, 'iters': 0}
        res_right = {'q': q0_right, 'success': True, 'pos_err': 0.0, 'ori_err': 0.0, 'iters': 0}

        if target_left is not None:
            tgt_left = target_left.tolist() if hasattr(target_left, 'tolist') else target_left
            res_left = self.left.ik(tgt_left, q_initial=q0_left, **ik_kwargs)

        if target_right is not None:
            tgt_right = target_right.tolist() if hasattr(target_right, 'tolist') else target_right
            res_right = self.right.ik(tgt_right, q_initial=q0_right, **ik_kwargs)

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
              **ik_kwargs) -> Dict[str, Any]:
        """
        :param target_left: Left target pose
        :param target_right: Right target pose
        :param q0_left: Initial left configuration
        :param q0_right: Initial right configuration
        :param constraint_type: 'pose'|'position'|'orientation'
        :return: Result dict
        """
        # For now, delegate to independent IK per arm; future work will use
        # relative Jacobian-based coupling.
        return super().solve(target_left, target_right, q0_left, q0_right, **ik_kwargs)


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
              **ik_kwargs) -> Dict[str, Any]:
        """
        :param target_left: Left target pose
        :param target_right: Right target pose
        :param q0_left: Initial left configuration
        :param q0_right: Initial right configuration
        :param mirror_axis: Mirror axis 'x'|'y'|'z'
        :return: Result dict
        """
        # If only one side provided, mirror it to the other side.
        def mirror_pose(T):
            import numpy as np
            M = np.eye(4)
            if mirror_axis == 'x':
                M[0, 0] = -1
            elif mirror_axis == 'y':
                M[1, 1] = -1
            elif mirror_axis == 'z':
                M[2, 2] = -1
            return T @ M

        if target_left is None and target_right is not None:
            T_r = target_right.tolist() if hasattr(target_right, 'tolist') else target_right
            target_left = mirror_pose(T_r)
        elif target_right is None and target_left is not None:
            T_l = target_left.tolist() if hasattr(target_left, 'tolist') else target_left
            target_right = mirror_pose(T_l)

        return super().solve(target_left, target_right, q0_left, q0_right, **ik_kwargs)
