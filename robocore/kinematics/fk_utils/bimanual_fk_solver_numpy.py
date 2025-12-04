from typing import Any, Dict, Sequence
import numpy as np

from robocore.modeling.robot_model import RobotModel
from robocore.kinematics.fk_utils.fk_solver_numpy import FKSolverNumPy


class BiIndependentFKSolverNumpy:
    def __init__(self, left_model: RobotModel, right_model: RobotModel):
        self.left_solver = FKSolverNumPy(left_model)
        self.right_solver = FKSolverNumPy(right_model)

    def fk(self, q_left: Sequence[float], q_right: Sequence[float], *, backend: str = 'numpy', return_end: bool = True) -> Dict[str, Any]:
        poses_l = self.left_solver.solve(q_left, return_end_only=return_end)
        poses_r = self.right_solver.solve(q_right, return_end_only=return_end)

        T_l = poses_l['end'] if return_end else np.array(poses_l['end'])
        T_r = poses_r['end'] if return_end else np.array(poses_r['end'])
        # Ensure arrays
        T_l = np.array(T_l)
        T_r = np.array(T_r)
        return {'left': T_l, 'right': T_r}


class BiRelativeFKSolverNumpy(BiIndependentFKSolverNumpy):
    def fk(self, q_left: Sequence[float], q_right: Sequence[float], *,
           backend: str = 'numpy', return_end: bool = True,
           constraint_type: str = 'pose') -> Dict[str, Any]:
        """
        :param q_left: Left joint configuration
        :param q_right: Right joint configuration
        :param backend: Backend to use
        :param return_end: Return only end-effector poses
        :param constraint_type: 'pose'|'position'|'orientation'
        :return: {'left': T_left, 'right': T_right, 'relative': T_rel}
        """
        poses = super().fk(q_left, q_right, backend=backend, return_end=return_end)
        T_left = poses['left']
        T_right = poses['right']
        T_rel = np.linalg.inv(T_left) @ T_right
        result = poses.copy()
        result['relative'] = T_rel
        return result


class BiMirrorFKSolverNumpy(BiIndependentFKSolverNumpy):
    def fk(self, q_left: Sequence[float], q_right: Sequence[float], *,
           backend: str = 'numpy', return_end: bool = True,
           mirror_axis: str = 'y') -> Dict[str, Any]:
        """
        :param q_left: Left joint configuration
        :param q_right: Right joint configuration
        :param backend: Backend to use
        :param return_end: Return only end-effector poses
        :param mirror_axis: Mirror axis 'x'|'y'|'z'
        :return: {'left': T_left, 'right': T_right, 'mirror': T_mirror}
        """
        poses = super().fk(q_left, q_right, backend=backend, return_end=return_end)
        T_left = poses['left']
        mirror_mat = np.eye(4)
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
