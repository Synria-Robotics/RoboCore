"""Independent bimanual IK solver (Torch-aware).

Thin wrapper that attempts to use RobotModel.ik with backend='torch' and falls
back to numpy if torch backend is not available.
"""
from __future__ import annotations
from typing import Optional, Sequence, Dict, Any

from robocore.modeling.robot_model import RobotModel
from robocore.kinematics.ik import inverse_kinematics


class BiIndependentIKSolverTorch:
    def __init__(self, left_model: RobotModel, right_model: RobotModel):
        self.left = left_model
        self.right = right_model

    def solve(self,
              target_left: Optional[Sequence[Sequence[float]]],
              target_right: Optional[Sequence[Sequence[float]]],
              q0_left: Optional[Sequence[float]] = None,
              q0_right: Optional[Sequence[float]] = None,
              backend: str = 'torch', **ik_kwargs) -> Dict[str, Any]:
        if q0_left is None:
            q0_left = [0.0] * self.left.num_dof()
        if q0_right is None:
            q0_right = [0.0] * self.right.num_dof()

        res_left = {'q': q0_left, 'success': True, 'pos_err': 0.0, 'ori_err': 0.0, 'iters': 0}
        res_right = {'q': q0_right, 'success': True, 'pos_err': 0.0, 'ori_err': 0.0, 'iters': 0}

        if target_left is not None:
            tgt_left = target_left.tolist() if hasattr(target_left, 'tolist') else target_left
            res_left = inverse_kinematics(self.left, tgt_left, q0_left, backend=backend, **ik_kwargs)

        if target_right is not None:
            tgt_right = target_right.tolist() if hasattr(target_right, 'tolist') else target_right
            res_right = inverse_kinematics(self.right, tgt_right, q0_right, backend=backend, **ik_kwargs)

        return {
            'q_left': res_left.get('q', q0_left),
            'q_right': res_right.get('q', q0_right),
            'success_left': bool(res_left.get('success', False)),
            'success_right': bool(res_right.get('success', False)),
            'res_left': res_left,
            'res_right': res_right,
        }


class BiRelativeIKSolverTorch(BiIndependentIKSolverTorch):
    """Relative bimanual IK solver (Torch-aware)."""

    def solve(self,
              target_left: Optional[Sequence[Sequence[float]]],
              target_right: Optional[Sequence[Sequence[float]]],
              q0_left: Optional[Sequence[float]] = None,
              q0_right: Optional[Sequence[float]] = None,
              constraint_type: str = 'pose',
              backend: str = 'torch',
              T_rel_grasp=None,
              **ik_kwargs) -> Dict[str, Any]:
        """
        :param target_left: Left target pose
        :param target_right: Right target pose
        :param q0_left: Initial left configuration
        :param q0_right: Initial right configuration
        :param constraint_type: 'pose'|'position'|'orientation'
        :param backend: Backend string
        :return: Result dict
        """
        # Follow numpy implementation: solve left first, then constrain right
        if q0_left is None:
            q0_left = [0.0] * self.left.num_dof()
        if q0_right is None:
            q0_right = [0.0] * self.right.num_dof()

        tgt_left = target_left.tolist() if hasattr(target_left, 'tolist') else target_left
        res_left = None
        if tgt_left is not None:
            res_left = inverse_kinematics(self.left, tgt_left, q0_left, backend=backend, **ik_kwargs)
        else:
            res_left = {'q': q0_left, 'success': True}

        q_left = res_left.get('q', q0_left)

        # Compute constrained right target using left FK and T_rel_grasp
        T_left_current = self.left.fk(q_left)['end']
        T_right_constrained = T_left_current @ T_rel_grasp if T_rel_grasp is not None else None

        res_right = None
        if T_right_constrained is not None:
            res_right = inverse_kinematics(self.right, T_right_constrained, q0_right, backend=backend, **ik_kwargs)
        else:
            res_right = {'q': q0_right, 'success': True}

        return {
            'q_left': q_left,
            'q_right': res_right.get('q', q0_right),
            'success_left': bool(res_left.get('success', False)),
            'success_right': bool(res_right.get('success', False)),
            'res_left': res_left,
            'res_right': res_right,
        }


class BiMirrorIKSolverTorch(BiIndependentIKSolverTorch):
    """Mirror-symmetric bimanual IK solver (Torch-aware)."""

    def solve(self,
              target_left: Optional[Sequence[Sequence[float]]],
              target_right: Optional[Sequence[Sequence[float]]],
              q0_left: Optional[Sequence[float]] = None,
              q0_right: Optional[Sequence[float]] = None,
              mirror_axis: str = 'x',
              backend: str = 'torch',
              T_left_initial=None,
              T_right_initial=None,
              **ik_kwargs) -> Dict[str, Any]:
        """
        :param target_left: Left target pose
        :param target_right: Right target pose
        :param q0_left: Initial left configuration
        :param q0_right: Initial right configuration
        :param mirror_axis: Mirror axis 'x'|'y'|'z'
        :param backend: Backend string
        :return: Result dict
        """
        import numpy as np
        # If left provided and initials available, compute mirrored right using rotation-delta approach
        if target_left is not None and T_left_initial is not None and T_right_initial is not None:
            pos_left = np.array(target_left)[0:3, 3]
            pos_right_mirrored = np.array([-pos_left[0], pos_left[1], pos_left[2]])

            R_left_current = np.array(target_left)[0:3, 0:3]
            R_left_initial = np.array(T_left_initial)[0:3, 0:3]
            R_delta_left = R_left_current @ R_left_initial.T
            M_mirror = np.diag([-1, 1, 1])
            R_delta_right = M_mirror @ R_delta_left @ M_mirror.T
            R_right_initial = np.array(T_right_initial)[0:3, 0:3]
            R_right_mirrored = R_delta_right @ R_right_initial

            T_right_mirrored = np.eye(4)
            T_right_mirrored[0:3, 3] = pos_right_mirrored
            T_right_mirrored[0:3, 0:3] = R_right_mirrored
            tgt_right = T_right_mirrored
        else:
            tgt_right = target_right

        tgt_left = target_left

        return super().solve(tgt_left, tgt_right, q0_left, q0_right, backend=backend, **ik_kwargs)
