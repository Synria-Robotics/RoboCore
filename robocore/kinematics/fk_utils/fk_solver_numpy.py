"""NumPy-accelerated forward kinematics solver.

Copyright (c) 2025 Synria Robotics Co., Ltd.

Licensed under the MIT License.

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
from robocore.kinematics.utils import ensure_batch, restore_single

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
        self.actuated_joints = model._chain_dof_list
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
    ) -> Dict[str, np.ndarray] | np.ndarray:
        """Compute forward kinematics.
        
        :param q: joint configuration (n,) or (B, n).
        :param return_end_only: if True, only return end-effector pose.
        :return: 
            - Single mode: dict of link names to 4x4 pose matrices (if return_end_only=False) or 4x4 matrix (if return_end_only=True)
            - Batch mode: [B, 4, 4] array of end-effector poses (if return_end_only=True)
        """
        q = np.asarray(q, dtype=np.float64)
        q, was_single = ensure_batch(q)

        if q.shape[1] != self.n:
            raise ValueError(f"Expected q with {self.n} elements, got {q.shape[1]}")

        if return_end_only:
            result = self._solve_batch(q)
            return restore_single(result, was_single)
        else:
            # For non-end mode, process each sample (can be optimized later)
            batch_size = q.shape[0]
            if batch_size == 1:
                # Single sample - use original logic
                q_single = q[0]
                q_map = {j.name: q_single[j.index] for j in self.actuated_joints}
                root_link = self.joint_chain[0].parent if len(self.joint_chain) > 0 else self.base_link
                poses: Dict[str, np.ndarray] = {root_link: np.eye(4, dtype=np.float64)}

                for joint in self.joint_chain:
                    if joint.parent not in poses:
                        poses[joint.parent] = np.eye(4, dtype=np.float64)
                    parent_pose = poses[joint.parent]

                    T_origin = make_transform(
                        rpy_to_matrix(*joint.origin_rpy),
                        np.array(joint.origin_xyz, dtype=np.float64)
                    )

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
                    else:
                        R_joint = np.eye(3, dtype=np.float64)
                        t_joint = np.zeros(3, dtype=np.float64)

                    T_motion = make_transform(R_joint, t_joint)
                    child_pose = parent_pose @ T_origin @ T_motion
                    poses[joint.child] = child_pose

                poses["end"] = poses.get(self.end_link, list(poses.values())[-1])
                return poses
            else:
                # Batch mode - return dict with batch arrays
                results = {}
                for i in range(batch_size):
                    poses = self.solve(q[i], return_end_only=False)
                    if i == 0:
                        for link_name in poses.keys():
                            results[link_name] = []
                    for link_name, T in poses.items():
                        results[link_name].append(T)
                return {link_name: np.stack(arrays, axis=0) for link_name, arrays in results.items()}

    def _solve_batch(self, q_batch: np.ndarray) -> np.ndarray:
        """Compute forward kinematics for batch of configurations (vectorized).
        
        :param q_batch: joint configurations [B, n]
        :return: end-effector poses [B, 4, 4]
        """
        batch_size = q_batch.shape[0]

        # Initialize batch of identity matrices [B, 4, 4]
        T_batch = np.tile(np.eye(4, dtype=np.float64), (batch_size, 1, 1))

        # Build q_map for joint values (map joint name -> [B] array of values)
        q_map = {}
        for js in self.actuated_joints:
            q_map[js.name] = q_batch[:, js.index]  # [B]

        # Process each joint in the chain (includes all joints, not just actuated)
        for urdf_joint in self.joint_chain:
            # Get joint value if actuated, otherwise 0
            if urdf_joint.name in q_map:
                joint_value = q_map[urdf_joint.name]  # [B]
            else:
                joint_value = np.zeros(batch_size, dtype=np.float64)  # [B]

            # 1. Translation from origin
            origin_xyz = np.array(urdf_joint.origin_xyz, dtype=np.float64)  # [3]

            # 2. Rotation from origin (roll-pitch-yaw)
            origin_rpy = np.array(urdf_joint.origin_rpy, dtype=np.float64)  # [3]
            R_origin = rpy_to_matrix(origin_rpy[0], origin_rpy[1], origin_rpy[2])  # [3, 3]

            # 3. Joint motion transform
            if urdf_joint.joint_type == "revolute":
                axis = np.array(urdf_joint.axis, dtype=np.float64)  # [3]
                R_joint = self._axis_angle_to_rotation_matrix_batch(axis, joint_value)  # [B, 3, 3]
                t_joint = np.zeros((batch_size, 3), dtype=np.float64)  # [B, 3]
            elif urdf_joint.joint_type == "prismatic":
                R_joint = np.tile(np.eye(3, dtype=np.float64), (batch_size, 1, 1))  # [B, 3, 3]
                axis = np.array(urdf_joint.axis, dtype=np.float64)  # [3]
                axis_norm = np.linalg.norm(axis)
                if axis_norm > 1e-10:
                    axis = axis / axis_norm
                t_joint = axis[np.newaxis, :] * joint_value[:, np.newaxis]  # [B, 3]
            else:  # fixed
                R_joint = np.tile(np.eye(3, dtype=np.float64), (batch_size, 1, 1))  # [B, 3, 3]
                t_joint = np.zeros((batch_size, 3), dtype=np.float64)  # [B, 3]

            # Build T_origin: [R_origin | t_origin]
            T_origin = np.eye(4, dtype=np.float64)
            T_origin[:3, :3] = R_origin
            T_origin[:3, 3] = origin_xyz

            # Build T_motion: [R_joint | t_joint] for each sample
            T_motion = np.tile(np.eye(4, dtype=np.float64), (batch_size, 1, 1))
            T_motion[:, :3, :3] = R_joint
            T_motion[:, :3, 3] = t_joint

            # Combine: T_joint = T_origin @ T_motion
            # T_origin is [4, 4], T_motion is [B, 4, 4]
            # Result: [B, 4, 4] where each T_joint[i] = T_origin @ T_motion[i]
            T_joint = np.einsum('ij,bjk->bik', T_origin, T_motion)  # [B, 4, 4]

            # Accumulate transformation: T_batch = T_batch @ T_joint (batch matmul)
            T_batch = np.einsum('bij,bjk->bik', T_batch, T_joint)

        return T_batch

    @staticmethod
    def _axis_angle_to_rotation_matrix_batch(axis: np.ndarray, angles: np.ndarray) -> np.ndarray:
        """Convert batch of axis-angle rotations to rotation matrices.
        
        :param axis: rotation axis [3]
        :param angles: rotation angles [B]
        :return: rotation matrices [B, 3, 3]
        """
        batch_size = angles.shape[0]

        # Normalize axis
        axis_norm = np.linalg.norm(axis)
        if axis_norm < 1e-10:
            return np.tile(np.eye(3, dtype=np.float64), (batch_size, 1, 1))
        k = axis / axis_norm
        kx, ky, kz = k[0], k[1], k[2]
        
        # Skew-symmetric matrix K (same for all samples)
        K = np.array([
            [0, -kz, ky],
            [kz, 0, -kx],
            [-ky, kx, 0]
        ], dtype=np.float64)
        
        # K²
        K2 = K @ K
        
        # Compute for batch
        cos_theta = np.cos(angles)  # [B]
        sin_theta = np.sin(angles)  # [B]
        
        # Broadcast: I + sin(θ)*K + (1-cos(θ))*K²
        I = np.eye(3, dtype=np.float64)
        R_batch = (I[np.newaxis, :, :] +
                   sin_theta[:, np.newaxis, np.newaxis] * K[np.newaxis, :, :] +
                   (1 - cos_theta)[:, np.newaxis, np.newaxis] * K2[np.newaxis, :, :])
        
        return R_batch

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
