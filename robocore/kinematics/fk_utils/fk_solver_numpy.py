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

if TYPE_CHECKING:
    from robocore.modeling.robot_model import RobotModel


class FKSolverNumPy:
    """NumPy-accelerated forward kinematics solver.
    
    Features:
    - Fast matrix operations using NumPy
    - Efficient pose computation for all links in chain
    - Support for revolute, prismatic, and fixed joints
    - Multi-chain FK with transform reuse
    """
    
    def __init__(self, model: "RobotModel"):
        """Initialize FK solver.
        
        :param model: robot model.
        """
        self.model = model
        self.n = model.num_chain_dof
        self.joint_chain = model._chain_joints
        self.actuated_joints = model._chain_actuated
        self.base_link = model.base_link
        self.end_link = model.end_link

        # Multi-chain support
        self._has_multi_chain = hasattr(model, '_link_to_idx')
        if self._has_multi_chain:
            self._link_to_idx = model._link_to_idx
            self._idx_to_link = model._idx_to_link
            self._parent_indices = model._parent_indices
            self._link_joints = model._link_joints
            self._num_links = model._num_links_in_tree

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
        # Backend should already be set correctly by caller
        q = np.asarray(q, dtype=np.float64)

        if q.shape[0] != self.n:
            raise ValueError(f"Expected q with {self.n} elements, got {q.shape[0]}")

        # Build quick lookup
        q_map = {j.name: q[j.index] for j in self.actuated_joints}

        # Determine root link (first joint's parent, which may be 'world' if world_to_base_joint exists)
        root_link = self.joint_chain[0].parent if len(self.joint_chain) > 0 else self.base_link

        # Base pose - start from root link
        poses: Dict[str, np.ndarray] = {
            root_link: np.eye(4, dtype=np.float64)
        }

        # Traverse chain
        for joint in self.joint_chain:
            # Ensure parent pose exists (for cases where parent is not base_link)
            if joint.parent not in poses:
                poses[joint.parent] = np.eye(4, dtype=np.float64)
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

    def solve_multi_chain(
        self,
        q: Sequence[float],
        link_names: Sequence[str] | None = None
    ) -> Dict[str, np.ndarray]:
        """Compute forward kinematics for multiple chains with transform reuse.
        
        Inspired by pytorch_kinematics, this method computes FK for all links
        in depth-first order, reusing transforms where possible for efficiency.
        
        :param q: joint configuration, dict {joint_name: value} or array-like with all DOF values.
        :param link_names: list of link names to compute FK for (None = all links).
        :return: dict mapping link names to 4x4 pose matrices.
        """
        if not self._has_multi_chain:
            raise RuntimeError("Multi-chain FK requires model with multi-chain indexing")

        # Backend should already be set correctly by caller
            # Build joint value mapping
            if isinstance(q, dict):
                q_map = q
            else:
                q_arr = np.asarray(q, dtype=np.float64)
                # Map to joint names using actuated joints
                q_map = {}
                # Use all joints from parsed model for full DOF coverage
                for i, joint_spec in enumerate(self.model.joint_list):
                    if i < len(q_arr):
                        # joint_list contains JointSpec objects, extract name
                        joint_name = joint_spec.name if hasattr(joint_spec, 'name') else joint_spec
                        q_map[joint_name] = q_arr[i]

            # Determine which links to compute
            if link_names is None:
                # Compute all links
                target_indices = list(range(self._num_links))
            else:
                target_indices = [self._link_to_idx[name] for name in link_names]

            # Cache for computed transforms (indexed by link index)
            transform_cache: Dict[int, np.ndarray] = {}

            # Compute transforms for requested links
            for link_idx in target_indices:
                if link_idx in transform_cache:
                    continue  # Already computed

                # Build transform by traversing parent path
                T = np.eye(4, dtype=np.float64)

                for ancestor_idx in self._parent_indices[link_idx]:
                    if ancestor_idx in transform_cache:
                        # Reuse cached transform
                        T = transform_cache[ancestor_idx].copy()
                    else:
                        # Compute transform for this ancestor
                        joint_spec = self._link_joints[ancestor_idx]

                        if joint_spec is None:
                            # Root node - identity
                            transform_cache[ancestor_idx] = np.eye(4, dtype=np.float64)
                        else:
                            # Get parent transform
                            parent_idx = self._parent_indices[ancestor_idx][-2] if len(self._parent_indices[ancestor_idx]) > 1 else -1
                            if parent_idx == -1:
                                T_parent = np.eye(4, dtype=np.float64)
                            else:
                                T_parent = transform_cache.get(parent_idx, np.eye(4, dtype=np.float64))

                            # Compute this joint's transform
                            T_origin = make_transform(
                                rpy_to_matrix(*joint_spec.origin_rpy),
                                np.array(joint_spec.origin_xyz, dtype=np.float64)
                            )

                            # Joint motion
                            if joint_spec.joint_type == "revolute":
                                R_joint = axis_angle_to_matrix(
                                    np.array(joint_spec.axis, dtype=np.float64),
                                    q_map.get(joint_spec.name, 0.0)
                                )
                                t_joint = np.zeros(3, dtype=np.float64)
                            elif joint_spec.joint_type == "prismatic":
                                R_joint = np.eye(3, dtype=np.float64)
                                axis_vec = np.array(joint_spec.axis, dtype=np.float64)
                                t_joint = axis_vec * q_map.get(joint_spec.name, 0.0)
                            else:  # fixed
                                R_joint = np.eye(3, dtype=np.float64)
                                t_joint = np.zeros(3, dtype=np.float64)

                            T_motion = make_transform(R_joint, t_joint)

                            # Compose
                            T_link = T_parent @ T_origin @ T_motion
                            transform_cache[ancestor_idx] = T_link
                            T = T_link

                # Final transform for this link
                transform_cache[link_idx] = T

            # Build result dictionary
            result = {}
            for link_idx in target_indices:
                link_name = self._idx_to_link[link_idx]
                result[link_name] = transform_cache[link_idx]

            # Add 'end' key for compatibility
            if self.end_link in result:
                result['end'] = result[self.end_link]

            return result


__all__ = ["FKSolverNumPy"]
