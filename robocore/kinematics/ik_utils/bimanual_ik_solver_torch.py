"""Independent bimanual IK solver (Torch-aware).

Thin wrapper that uses RobotModel.ik with global backend setting.
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
        res_left = {'q': q0_left if q0_left is not None else [0.0] * self.left.num_chain_dof,
                    'success': True, 'pos_err': 0.0, 'ori_err': 0.0, 'iters': 0}
        res_right = {'q': q0_right if q0_right is not None else [0.0] * self.right.num_chain_dof,
                     'success': True, 'pos_err': 0.0, 'ori_err': 0.0, 'iters': 0}

        # Check if batch mode
        import numpy as np
        is_batch = False
        if target_left is not None:
            tgt_left_arr = np.asarray(target_left)
            is_batch = tgt_left_arr.ndim == 3 and tgt_left_arr.shape[1:] == (4, 4)
        elif target_right is not None:
            tgt_right_arr = np.asarray(target_right)
            is_batch = tgt_right_arr.ndim == 3 and tgt_right_arr.shape[1:] == (4, 4)

        if target_left is not None:
            tgt_left = target_left.tolist() if hasattr(target_left, 'tolist') else target_left
            res_left = inverse_kinematics(self.left, tgt_left, q0_left, **ik_kwargs)

        if target_right is not None:
            tgt_right = target_right.tolist() if hasattr(target_right, 'tolist') else target_right
            res_right = inverse_kinematics(self.right, tgt_right, q0_right, **ik_kwargs)

        # Handle batch mode (returns list of dicts)
        if is_batch and isinstance(res_left, list) and isinstance(res_right, list):
            # Return list of combined results
            results = []
            for r_l, r_r in zip(res_left, res_right):
                results.append({
                    'q_left': r_l.get('q', q0_left if q0_left is not None else [0.0] * self.left.num_chain_dof),
                    'q_right': r_r.get('q', q0_right if q0_right is not None else [0.0] * self.right.num_chain_dof),
                    'success_left': bool(r_l.get('success', False)),
                    'success_right': bool(r_r.get('success', False)),
                    'res_left': r_l,
                    'res_right': r_r,
                })
            return results

        # Single mode (returns dict)
        return {
            'q_left': res_left.get('q', q0_left) if isinstance(res_left, dict) else (res_left[0].get('q', q0_left) if isinstance(res_left, list) and len(res_left) > 0 else [0.0] * self.left.num_chain_dof),
            'q_right': res_right.get('q', q0_right) if isinstance(res_right, dict) else (res_right[0].get('q', q0_right) if isinstance(res_right, list) and len(res_right) > 0 else [0.0] * self.right.num_chain_dof),
            'success_left': bool(res_left.get('success', False) if isinstance(res_left, dict) else (res_left[0].get('success', False) if isinstance(res_left, list) and len(res_left) > 0 else False)),
            'success_right': bool(res_right.get('success', False) if isinstance(res_right, dict) else (res_right[0].get('success', False) if isinstance(res_right, list) and len(res_right) > 0 else False)),
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
              T_rel_grasp=None,
              **ik_kwargs) -> Dict[str, Any]:
        """
        :param target_left: Left target pose
        :param target_right: Right target pose
        :param q0_left: Initial left configuration
        :param q0_right: Initial right configuration
        :param constraint_type: 'pose'|'position'|'orientation'
        :return: Result dict
        """
        # Follow numpy implementation: solve left first, then constrain right
        if q0_left is None:
            q0_left = [0.0] * self.left.num_dof
        if q0_right is None:
            q0_right = [0.0] * self.right.num_dof

        tgt_left = target_left.tolist() if hasattr(target_left, 'tolist') else target_left
        res_left = None
        if tgt_left is not None:
            res_left = inverse_kinematics(self.left, tgt_left, q0_left, **ik_kwargs)
        else:
            res_left = {'q': q0_left if q0_left is not None else [0.0] * self.left.num_chain_dof, 'success': True}

        q_left = res_left.get('q', q0_left if q0_left is not None else [0.0] * self.left.num_chain_dof)

        # Compute constrained right target using left FK and T_rel_grasp
        T_left_current = self.left.fk(q_left)['end']
        T_right_constrained = T_left_current @ T_rel_grasp if T_rel_grasp is not None else None

        res_right = None
        if T_right_constrained is not None:
            res_right = inverse_kinematics(self.right, T_right_constrained, q0_right, **ik_kwargs)
        else:
            res_right = {'q': q0_right if q0_right is not None else [0.0] * self.right.num_chain_dof, 'success': True}

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
        :return: Result dict
        """
        import numpy as np
        # If left provided and initials available, compute mirrored right using rotation-delta approach
        if target_left is not None and T_left_initial is not None and T_right_initial is not None:
            # Mirror position (based on initial left/right positions, symmetry plane is Y=0, mirror Y axis)
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

            # Mirror rotation
            R_left_current = np.array(target_left)[0:3, 0:3]
            R_left_initial = np.array(T_left_initial)[0:3, 0:3]
            R_delta_left = R_left_current @ R_left_initial.T
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

        return super().solve(tgt_left, tgt_right, q0_left, q0_right, **ik_kwargs)
