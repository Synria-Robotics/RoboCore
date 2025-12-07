from typing import Any, Dict, Sequence, Union
import torch
import numpy as np

from robocore.modeling.robot_model import RobotModel
from robocore.kinematics.fk_utils.fk_solver_torch import FKSolverTorch


class BiIndependentFKSolverTorch:
    def __init__(self, left_model: RobotModel, right_model: RobotModel):
        self.left_solver = FKSolverTorch(left_model)
        self.right_solver = FKSolverTorch(right_model)

    def fk(self, q_left: Union[Sequence[float], np.ndarray, torch.Tensor],
           q_right: Union[Sequence[float], np.ndarray, torch.Tensor],
           *, return_end: bool = True, device=None, dtype=torch.float64) -> Dict[str, Any]:
        """Compute bimanual forward kinematics.
        
        :param q_left: Left joint configuration(s) - [n] or [B, n]
        :param q_right: Right joint configuration(s) - [n] or [B, n]
        :param return_end: Return only end-effector poses
        :param device: Torch device
        :param dtype: Torch dtype
        :return: {'left': T_left, 'right': T_right} or batch dict
        """
        poses_l = self.left_solver.solve(q_left, return_end_only=return_end, device=device, dtype=dtype)
        poses_r = self.right_solver.solve(q_right, return_end_only=return_end, device=device, dtype=dtype)

        # Handle return format: if return_end=True, solve returns tensor directly, otherwise dict
        if return_end:
            T_l = poses_l if torch.is_tensor(poses_l) else poses_l['end']
            T_r = poses_r if torch.is_tensor(poses_r) else poses_r['end']
        else:
            T_l = poses_l['end'] if isinstance(poses_l, dict) else poses_l
            T_r = poses_r['end'] if isinstance(poses_r, dict) else poses_r

        if not torch.is_tensor(T_l):
            T_l = torch.tensor(T_l, dtype=dtype, device=device)
        if not torch.is_tensor(T_r):
            T_r = torch.tensor(T_r, dtype=dtype, device=device)
        return {'left': T_l, 'right': T_r}


class BiRelativeFKSolverTorch(BiIndependentFKSolverTorch):
    def fk(self, q_left: Sequence[float], q_right: Sequence[float], *, return_end: bool = True, device=None, dtype=torch.float64, constraint_type: str = 'pose') -> Dict[str, Any]:
        """
        :param q_left: Left joint configuration
        :param q_right: Right joint configuration
        :param return_end: Return only end-effector poses
        :param device: Torch device
        :param dtype: Torch dtype
        :param constraint_type: 'pose'|'position'|'orientation'
        :return: {'left': T_left, 'right': T_right, 'relative': T_rel}
        """
        poses = super().fk(q_left, q_right, return_end=return_end, device=device, dtype=dtype)
        T_left = poses['left']
        T_right = poses['right']
        T_rel = torch.linalg.inv(T_left) @ T_right
        result = poses.copy()
        result['relative'] = T_rel
        return result


class BiMirrorFKSolverTorch(BiIndependentFKSolverTorch):
    def fk(self, q_left: Sequence[float], q_right: Sequence[float], *, return_end: bool = True, device=None, dtype=torch.float64, mirror_axis: str = 'y') -> Dict[str, Any]:
        """
        :param q_left: Left joint configuration
        :param q_right: Right joint configuration
        :param return_end: Return only end-effector poses
        :param device: Torch device
        :param dtype: Torch dtype
        :param mirror_axis: Mirror axis 'x'|'y'|'z'
        :return: {'left': T_left, 'right': T_right, 'mirror': T_mirror}
        """
        poses = super().fk(q_left, q_right, return_end=return_end, device=device, dtype=dtype)
        T_left = poses['left']
        mirror_mat = torch.eye(4, dtype=dtype, device=T_left.device)
        if mirror_axis == 'x':
            mirror_mat[0, 0] = -1
        elif mirror_axis == 'y':
            mirror_mat[1, 1] = -1
        elif mirror_axis == 'z':
            mirror_mat[2, 2] = -1
        T_mirror = T_left @ mirror_mat
        result = poses.copy()
        result['mirror'] = T_mirror
        return result
