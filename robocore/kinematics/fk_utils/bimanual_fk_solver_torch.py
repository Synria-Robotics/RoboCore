from typing import Any, Dict, Sequence
import torch

from robocore.modeling.robot_model import RobotModel
from robocore.kinematics.fk_utils.fk_solver_torch import FKSolverTorch


class BiIndependentFKSolverTorch:
    def __init__(self, left_model: RobotModel, right_model: RobotModel):
        self.left_solver = FKSolverTorch(left_model)
        self.right_solver = FKSolverTorch(right_model)

    def fk(self, q_left: Sequence[float], q_right: Sequence[float], *, backend: str = 'torch', return_end: bool = True, device=None, dtype=torch.float64) -> Dict[str, Any]:
        poses_l = self.left_solver.solve(q_left, return_end_only=return_end, device=device, dtype=dtype)
        poses_r = self.right_solver.solve(q_right, return_end_only=return_end, device=device, dtype=dtype)

        T_l = poses_l['end'] if isinstance(poses_l, dict) else poses_l
        T_r = poses_r['end'] if isinstance(poses_r, dict) else poses_r

        if not torch.is_tensor(T_l):
            T_l = torch.tensor(T_l, dtype=dtype)
        if not torch.is_tensor(T_r):
            T_r = torch.tensor(T_r, dtype=dtype)
        return {'left': T_l, 'right': T_r}


class BiRelativeFKSolverTorch(BiIndependentFKSolverTorch):
    def fk(self, q_left: Sequence[float], q_right: Sequence[float], *, backend: str = 'torch', return_end: bool = True, device=None, dtype=torch.float64, constraint_type: str = 'pose') -> Dict[str, Any]:
        """
        :param q_left: Left joint configuration
        :param q_right: Right joint configuration
        :param backend: Backend to use
        :param return_end: Return only end-effector poses
        :param device: Torch device
        :param dtype: Torch dtype
        :param constraint_type: 'pose'|'position'|'orientation'
        :return: {'left': T_left, 'right': T_right, 'relative': T_rel}
        """
        poses = super().fk(q_left, q_right, backend=backend, return_end=return_end, device=device, dtype=dtype)
        T_left = poses['left']
        T_right = poses['right']
        T_rel = torch.linalg.inv(T_left) @ T_right
        result = poses.copy()
        result['relative'] = T_rel
        return result


class BiMirrorFKSolverTorch(BiIndependentFKSolverTorch):
    def fk(self, q_left: Sequence[float], q_right: Sequence[float], *, backend: str = 'torch', return_end: bool = True, device=None, dtype=torch.float64, mirror_axis: str = 'y') -> Dict[str, Any]:
        """
        :param q_left: Left joint configuration
        :param q_right: Right joint configuration
        :param backend: Backend to use
        :param return_end: Return only end-effector poses
        :param device: Torch device
        :param dtype: Torch dtype
        :param mirror_axis: Mirror axis 'x'|'y'|'z'
        :return: {'left': T_left, 'right': T_right, 'mirror': T_mirror}
        """
        poses = super().fk(q_left, q_right, backend=backend, return_end=return_end, device=device, dtype=dtype)
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
