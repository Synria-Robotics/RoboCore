"""Robot model abstraction.

Loads from URDF (and later MJCF) and provides forward kinematics.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from .parser.urdf_parser import load_urdf, URDFJoint
from robocore import backend as B
import math

# Optional NumPy acceleration
try:
    import numpy as np
    from robocore.kinematics.fk_numpy import forward_kinematics_numpy
    _HAS_NUMPY = True
except ImportError:
    _HAS_NUMPY = False


@dataclass
class JointSpec:
    """Actuated joint specification.

    :param name: joint name.
    :param index: index in configuration.
    :param joint_type: revolute/prismatic.
    :param axis: axis vector (3,).
    :param parent: parent link.
    :param child: child link.
    :param origin_xyz: translation of joint frame.
    :param origin_rpy: rpy of joint frame.
    :param limit: (lower, upper) or None.
    """

    name: str
    index: int
    joint_type: str
    axis: List[float]
    parent: str
    child: str
    origin_xyz: List[float]
    origin_rpy: List[float]
    limit: Optional[Sequence[Optional[float]]]


class RobotModel:
    """Generic serial chain robot.

    :param file_path: URDF (or MJCF in future) path.
    :param end_link: override end-effector link name.
    """

    def __init__(self, file_path: str | Path, end_link: Optional[str] = None, use_numpy: bool = True):
        """Initialize robot model.

        :param file_path: path to URDF file.
        :param end_link: end-effector link name (auto-detect if None).
        :param use_numpy: if True and NumPy is available, use NumPy-accelerated FK (50-100x faster).
        """
        self.file_path = str(file_path)
        self.use_numpy = use_numpy and _HAS_NUMPY
        parsed = load_urdf(self.file_path)
        self.name = parsed.get("name", "")
        self._raw_joints = parsed["joints"]
        self.base_link = parsed["base_links"][0] if parsed["base_links"] else self._raw_joints[0].parent
        self._graph = self._build_graph(self._raw_joints)
        self._chain_joints = self._linearize_chain(self.base_link, end_link)
        self._actuated = []
        idx = 0
        for j in self._chain_joints:
            if j.joint_type in ("revolute", "prismatic"):
                self._actuated.append(
                    JointSpec(
                        name=j.name,
                        index=idx,
                        joint_type=j.joint_type,
                        axis=j.axis,
                        parent=j.parent,
                        child=j.child,
                        origin_xyz=j.origin_xyz,
                        origin_rpy=j.origin_rpy,
                        limit=(j.limit_lower, j.limit_upper),
                    )
                )
                idx += 1
        self.end_link = end_link or (self._chain_joints[-1].child if self._chain_joints else self.base_link)

    @staticmethod
    def _build_graph(joints: List[URDFJoint]):
        g: Dict[str, List[URDFJoint]] = {}
        for j in joints:
            g.setdefault(j.parent, []).append(j)
        return g

    def _linearize_chain(self, base: str, end_link: Optional[str]) -> List[URDFJoint]:
        if end_link is None:
            # choose longest path by simple DFS
            best: List[URDFJoint] = []

            def dfs(link: str, path: List[URDFJoint]):
                nonlocal best
                if len(path) > len(best):
                    best = path.copy()
                for j in self._graph.get(link, []):
                    path.append(j)
                    dfs(j.child, path)
                    path.pop()

            dfs(base, [])
            return best
        # else find path to end_link
        res: List[URDFJoint] = []
        found = False

        def dfs2(link: str, path: List[URDFJoint]):
            nonlocal found, res
            if found:
                return
            if link == end_link:
                res = path.copy()
                found = True
                return
            for j in self._graph.get(link, []):
                path.append(j)
                dfs2(j.child, path)
                path.pop()

        dfs2(base, [])
        return res

    # ---------------- Public API -----------------
    def dof(self) -> int:
        """Return number of actuated joints.

        :return: dof.
        """
        return len(self._actuated)

    def joint_names(self) -> List[str]:
        """Actuated joint names.

        :return: names.
        """
        return [j.name for j in self._actuated]

    def name_to_index(self) -> Dict[str, int]:
        """Map joint name to index.

        :return: mapping.
        """
        return {j.name: j.index for j in self._actuated}

    # ------------- Kinematics ---------------------
    @staticmethod
    def _rpy_matrix(r, p, y):
        sr, cr = math.sin(r), math.cos(r)
        sp, cp = math.sin(p), math.cos(p)
        sy, cy = math.sin(y), math.cos(y)
        # R = Rz(y) * Ry(p) * Rx(r)
        return [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ]

    @staticmethod
    def _axis_rotation(axis, theta):
        ax, ay, az = axis
        norm = math.sqrt(ax * ax + ay * ay + az * az) or 1.0
        ax, ay, az = ax / norm, ay / norm, az / norm
        ct = math.cos(theta)
        st = math.sin(theta)
        vt = 1 - ct
        return [
            [ct + ax * ax * vt, ax * ay * vt - az * st, ax * az * vt + ay * st],
            [ay * ax * vt + az * st, ct + ay * ay * vt, ay * az * vt - ax * st],
            [az * ax * vt - ay * st, az * ay * vt + ax * st, ct + az * az * vt],
        ]

    @staticmethod
    def _axis_translation(axis, d):
        ax, ay, az = axis
        norm = math.sqrt(ax * ax + ay * ay + az * az) or 1.0
        ax, ay, az = ax / norm, ay / norm, az / norm
        return [ax * d, ay * d, az * d]

    def forward_kinematics(self, q: Sequence[float], return_numpy: bool = False):
        """Compute FK for chain.

        :param q: joint values with length dof().
        :param return_numpy: if True and NumPy is enabled, return NumPy arrays; else lists.
        :return: dict link->(4x4 pose matrix), end-effector pose under key 'end'.
        """
        if len(q) != self.dof():
            raise ValueError("Expected %d joint values" % self.dof())
        
        # Use NumPy-accelerated FK if available and enabled
        if self.use_numpy and _HAS_NUMPY:
            poses = forward_kinematics_numpy(
                self._chain_joints,
                self._actuated,
                self.base_link,
                self.end_link,
                q
            )
            if not return_numpy:
                # Convert NumPy arrays to lists for compatibility
                poses = {k: v.tolist() for k, v in poses.items()}
            return poses
        # Pure Python fallback
        q_map = {j.name: q[j.index] for j in self._actuated}
        # base pose
        poses: Dict[str, List[List[float]]] = {self.base_link: self._make_transform([[1, 0, 0], [0, 1, 0], [0, 0, 1]], [0, 0, 0])}
        # traverse chain joints
        for j in self._chain_joints:
            parent_pose = poses[j.parent]
            R_origin = self._rpy_matrix(*j.origin_rpy)
            t_origin = j.origin_xyz
            R_joint = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
            t_joint = [0, 0, 0]
            if j.joint_type == "revolute":
                R_joint = self._axis_rotation(j.axis, q_map.get(j.name, 0.0))
            elif j.joint_type == "prismatic":
                t_joint = self._axis_translation(j.axis, q_map.get(j.name, 0.0))
            # compose parent -> joint origin
            T_origin = self._make_transform(R_origin, t_origin)
            T_motion = self._make_transform(R_joint, t_joint)
            child_pose = self._matmul(parent_pose, self._matmul(T_origin, T_motion))
            poses[j.child] = child_pose
        poses["end"] = poses.get(self.end_link, list(poses.values())[-1])
        return poses

    # ------------- Small matrix helpers -------------
    @staticmethod
    def _make_transform(R, t):
        return [
            [R[0][0], R[0][1], R[0][2], t[0]],
            [R[1][0], R[1][1], R[1][2], t[1]],
            [R[2][0], R[2][1], R[2][2], t[2]],
            [0, 0, 0, 1],
        ]

    @staticmethod
    def _matmul(A, B):
        C = [[0.0] * 4 for _ in range(4)]
        for i in range(4):
            for j in range(4):
                C[i][j] = sum(A[i][k] * B[k][j] for k in range(4))
        return C


__all__ = ["RobotModel", "JointSpec"]
