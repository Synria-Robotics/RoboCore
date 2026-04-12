"""Bimanual FK using native C++ chain (Eigen/pybind), per arm.

Copyright (c) 2025 Synria Robotics Co., Ltd.
"""

from __future__ import annotations

from typing import Any, Dict, Sequence, Union

import numpy as np

from robocore.modeling.robot_model import RobotModel
from robocore.kinematics.fk_utils.fk_solver_cpp import FKSolverCpp


class BiIndependentFKSolverCpp:
    def __init__(self, left_model: RobotModel, right_model: RobotModel):
        self.left_solver = FKSolverCpp(left_model)
        self.right_solver = FKSolverCpp(right_model)

    def fk(
        self,
        q_left: Union[Sequence[float], np.ndarray],
        q_right: Union[Sequence[float], np.ndarray],
        *,
        return_end: bool = True,
    ) -> Dict[str, Any]:
        """Compute bimanual forward kinematics (C++ per chain).

        :param q_left: Left joint configuration(s) - [n] or [B, n]
        :param q_right: Right joint configuration(s) - [n] or [B, n]
        :param return_end: Return only end-effector poses
        :return: {'left': T_left, 'right': T_right}
        """
        q_left_arr = np.asarray(q_left)
        q_right_arr = np.asarray(q_right)
        is_batch = q_left_arr.ndim == 2 and q_right_arr.ndim == 2
        if is_batch:
            batch_size = q_left_arr.shape[0]
            if q_right_arr.shape[0] != batch_size:
                raise ValueError(
                    f"Batch size mismatch: left={batch_size}, right={q_right_arr.shape[0]}"
                )

        poses_l = self.left_solver.solve(q_left, return_end_only=return_end)
        poses_r = self.right_solver.solve(q_right, return_end_only=return_end)

        if return_end:
            T_l = poses_l if isinstance(poses_l, np.ndarray) else poses_l["end"]
            T_r = poses_r if isinstance(poses_r, np.ndarray) else poses_r["end"]
        else:
            T_l = poses_l["end"] if isinstance(poses_l, dict) else poses_l
            T_r = poses_r["end"] if isinstance(poses_r, dict) else poses_r

        T_l = np.array(T_l, dtype=np.float64)
        T_r = np.array(T_r, dtype=np.float64)
        return {"left": T_l, "right": T_r}


class BiRelativeFKSolverCpp(BiIndependentFKSolverCpp):
    def fk(
        self,
        q_left: Sequence[float],
        q_right: Sequence[float],
        *,
        return_end: bool = True,
        constraint_type: str = "pose",
    ) -> Dict[str, Any]:
        """
        :param q_left: Left joint configuration
        :param q_right: Right joint configuration
        :param return_end: Return only end-effector poses
        :param constraint_type: unused (API parity with NumPy solver)
        :return: {'left': T_left, 'right': T_right, 'relative': T_rel}
        """
        poses = super().fk(q_left, q_right, return_end=return_end)
        T_left = poses["left"]
        T_right = poses["right"]
        T_rel = np.linalg.inv(T_left) @ T_right
        result = poses.copy()
        result["relative"] = T_rel
        return result


class BiMirrorFKSolverCpp(BiIndependentFKSolverCpp):
    def fk(
        self,
        q_left: Sequence[float],
        q_right: Sequence[float],
        *,
        return_end: bool = True,
        mirror_axis: str = "y",
    ) -> Dict[str, Any]:
        """
        :param q_left: Left joint configuration
        :param q_right: Right joint configuration
        :param return_end: Return only end-effector poses
        :param mirror_axis: Mirror axis 'x'|'y'|'z'
        :return: {'left': T_left, 'right': T_right, 'mirror': T_mirror}
        """
        poses = super().fk(q_left, q_right, return_end=return_end)
        T_left = poses["left"]
        mirror_mat = np.eye(4, dtype=np.float64)
        if mirror_axis == "x":
            mirror_mat[0, 0] = -1
        elif mirror_axis == "y":
            mirror_mat[1, 1] = -1
        elif mirror_axis == "z":
            mirror_mat[2, 2] = -1
        else:
            raise ValueError(f"Invalid mirror_axis: {mirror_axis}, must be 'x', 'y', or 'z'")
        T_mirror = T_left @ mirror_mat
        result = poses.copy()
        result["mirror"] = T_mirror
        return result


__all__ = [
    "BiIndependentFKSolverCpp",
    "BiRelativeFKSolverCpp",
    "BiMirrorFKSolverCpp",
]
