"""NumPy-accelerated forward kinematics solver.

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

from typing import TYPE_CHECKING, Dict, Sequence
import numpy as np
from robocore.transform import (
    rpy_to_matrix,
    axis_angle_to_matrix,
    make_transform,
)
from robocore.utils.backend import set_backend, get_backend

if TYPE_CHECKING:
    from robocore.modeling.robot_model import RobotModel


class FKSolverNumPy:
    """NumPy-accelerated forward kinematics solver.
    
    Features:
    - Fast matrix operations using NumPy
    - Efficient pose computation for all links in chain
    - Support for revolute, prismatic, and fixed joints
    """
    
    def __init__(self, model: "RobotModel"):
        """Initialize FK solver.
        
        :param model: robot model.
        """
        self.model = model
        self.n = model.num_dof()
        self.joint_chain = model._chain_joints
        self.actuated_joints = model._actuated
        self.base_link = model.base_link
        self.end_link = model.end_link
    
    def solve(
        self, 
        q: Sequence[float],
        return_end_only: bool = False
    ) -> Dict[str, np.ndarray]:
        """Compute forward kinematics.
        
        :param q: joint configuration (n,).
        :param return_end_only: if True, only return end-effector pose.
        :return: dict of link names to 4x4 pose matrices as numpy arrays.
        """
        # Ensure NumPy backend for transform functions
        prev_backend = get_backend()
        set_backend('numpy')

        try:
            q = np.asarray(q, dtype=np.float64)

            if q.shape[0] != self.n:
                raise ValueError(f"Expected q with {self.n} elements, got {q.shape[0]}")

            # Build quick lookup
            q_map = {j.name: q[j.index] for j in self.actuated_joints}

            # Base pose
            poses: Dict[str, np.ndarray] = {
                self.base_link: np.eye(4, dtype=np.float64)
            }

            # Traverse chain
            for joint in self.joint_chain:
                parent_pose = poses[joint.parent]

                # Joint origin transform (static)
                T_origin = make_transform(
                    rpy_to_matrix(*joint.origin_rpy),
                    np.array(joint.origin_xyz, dtype=np.float64)
                )

                # Joint motion transform (dynamic)
                if joint.joint_type == "revolute":
                    R_joint = axis_angle_to_matrix(
                        np.array(joint.axis, dtype=np.float64),
                        q_map.get(joint.name, 0.0)
                    )
                    t_joint = np.zeros(3, dtype=np.float64)
                elif joint.joint_type == "prismatic":
                    R_joint = np.eye(3, dtype=np.float64)
                    axis_vec = np.array(joint.axis, dtype=np.float64)
                    t_joint = axis_vec * q_map.get(joint.name, 0.0)
                else:  # fixed
                    R_joint = np.eye(3, dtype=np.float64)
                    t_joint = np.zeros(3, dtype=np.float64)

                T_motion = make_transform(R_joint, t_joint)

                # Compose: parent @ T_origin @ T_motion
                child_pose = parent_pose @ T_origin @ T_motion
                poses[joint.child] = child_pose

            # Add 'end' key
            poses["end"] = poses.get(self.end_link, list(poses.values())[-1])

            if return_end_only:
                return {"end": poses["end"]}

            return poses
        finally:
            # Restore previous backend
            set_backend(prev_backend)


__all__ = ["FKSolverNumPy"]
