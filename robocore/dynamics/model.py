"""Dynamics model: tree + inertia for use by id/fd algorithms.

Copyright (c) 2025 Synria Robotics Co., Ltd.

Licensed under the MIT License.

Author: Synria Robotics Team
Website: https://synriarobotics.ai
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def _rpy_to_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Rotation matrix from RPY (intrinsic XYZ)."""
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    R = np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ], dtype=np.float64)
    return R


def _make_transform(R: np.ndarray, p: np.ndarray) -> np.ndarray:
    """4x4 transform from R (3x3) and p (3,)."""
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(p, dtype=np.float64).ravel()[:3]
    return T


def _skew(v: np.ndarray) -> np.ndarray:
    """Skew-symmetric matrix of 3-vector."""
    v = np.asarray(v, dtype=np.float64).ravel()[:3]
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]], dtype=np.float64)


def _spatial_inertia_from_mci(mass: float, com: np.ndarray, I_com: np.ndarray) -> np.ndarray:
    """6x6 spatial inertia in body frame (origin at link origin). I_com at COM."""
    c = np.asarray(com, dtype=np.float64).ravel()[:3]
    I = np.asarray(I_com, dtype=np.float64)
    if I.shape != (3, 3):
        I = np.eye(3, dtype=np.float64) * 1e-6
    I_orig = I + mass * (np.dot(c, c) * np.eye(3) - np.outer(c, c))
    mc = mass * c
    top = np.hstack([I_orig, _skew(mc)])
    bot = np.hstack([_skew(mc).T, mass * np.eye(3)])
    return np.vstack([top, bot])


def _ad(T: np.ndarray) -> np.ndarray:
    """6x6 adjoint of 4x4 transform T (velocity from parent to child frame)."""
    R = T[:3, :3]
    p = T[:3, 3]
    P = _skew(p)
    top = np.hstack([R, np.zeros((3, 3))])
    bot = np.hstack([P @ R, R])
    return np.vstack([top, bot])


def _ad_T(T: np.ndarray) -> np.ndarray:
    """Transpose of adjoint (transform spatial force from child to parent)."""
    return _ad(T).T


def _spatial_inertia_to_mci(I6: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
    """Extract (mass, com_xyz, I_com) from 6x6 spatial inertia (in same frame)."""
    m = float(I6[3, 3])
    if m < 1e-12:
        return 0.0, np.zeros(3, dtype=np.float64), np.zeros((3, 3), dtype=np.float64)
    # Block [0:3, 3:6] is skew(m*c): skew(x,y,z) has (0,1)=-z,(0,2)=y,(1,0)=z,(1,2)=-x,(2,0)=-y,(2,1)=x.
    mc = np.array([-I6[1, 5], I6[0, 5], I6[1, 3]], dtype=np.float64)
    c = mc / m
    I_orig = I6[:3, :3].copy()
    I_com = I_orig - m * (np.dot(c, c) * np.eye(3) - np.outer(c, c))
    return m, c, I_com


@dataclass
class DynamicsBody:
    """Single body in the dynamics tree.

    :param link_name: Link name.
    :param parent: Parent body index (-1 for base).
    :param joint_type: revolute, prismatic, or fixed.
    :param joint_origin_xyz: Joint frame origin translation.
    :param joint_origin_rpy: Joint frame origin RPY.
    :param joint_axis: Joint axis (3,) unit vector.
    :param q_index: Index in q for this joint (-1 if fixed).
    :param mass: Mass (kg).
    :param com_xyz: Center of mass in link frame (3,).
    :param inertia: 3x3 inertia matrix at COM in link frame.
    """
    link_name: str
    parent: int
    joint_type: str
    joint_origin_xyz: np.ndarray
    joint_origin_rpy: np.ndarray
    joint_axis: np.ndarray
    q_index: int
    mass: float
    com_xyz: np.ndarray
    inertia: np.ndarray


@dataclass
class DynamicsModel:
    """Tree + inertia for dynamics (aligned with FK chain order).

    :param nq: Number of DOF.
    :param bodies: List of bodies in DFS order (body 0 = base).
    :param gravity: Gravity vector in world frame (3,), e.g. [0,0,-9.81].
    """
    nq: int
    bodies: List[DynamicsBody] = field(default_factory=list)
    gravity: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, -9.81], dtype=np.float64))

    def __post_init__(self):
        if self.gravity is None or len(self.gravity) != 3:
            self.gravity = np.array([0.0, 0.0, -9.81], dtype=np.float64)


def build_dynamics_model(robot_model: Any) -> DynamicsModel:
    """Build DynamicsModel from RobotModel (same chain as FK).

    :param robot_model: RobotModel instance.
    :return: DynamicsModel for use by id.py / fd.py.
    """
    chain_joints = getattr(robot_model, "_chain_joints", [])
    chain_dof_list = getattr(robot_model, "_chain_dof_list", [])
    base_link = robot_model.base_link

    joint_name_to_q_index: Dict[str, int] = {j.name: j.index for j in chain_dof_list}

    if hasattr(robot_model.parsed_model, "get_link_inertials"):
        link_inertials = robot_model.parsed_model.get_link_inertials()
    else:
        link_inertials = {}

    bodies: List[DynamicsBody] = []
    # Body 0: base link (no joint)
    mass0, com_xyz0, _com_rpy0, I0 = link_inertials.get(
        base_link, (0.0, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], np.zeros((3, 3), dtype=np.float64))
    )
    bodies.append(
        DynamicsBody(
            link_name=base_link,
            parent=-1,
            joint_type="fixed",
            joint_origin_xyz=np.zeros(3, dtype=np.float64),
            joint_origin_rpy=np.zeros(3, dtype=np.float64),
            joint_axis=np.array([0.0, 0.0, 1.0], dtype=np.float64),
            q_index=-1,
            mass=float(mass0),
            com_xyz=np.array(com_xyz0, dtype=np.float64),
            inertia=np.array(I0, dtype=np.float64),
        )
    )

    for j in chain_joints:
        child_link = j.child
        mass, com_xyz, _com_rpy, I = link_inertials.get(
            child_link, (0.0, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], np.zeros((3, 3), dtype=np.float64))
        )
        axis = np.array(j.axis, dtype=np.float64)
        if np.linalg.norm(axis) > 1e-10:
            axis = axis / np.linalg.norm(axis)
        else:
            axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        q_index = joint_name_to_q_index.get(j.name, -1)
        bodies.append(
            DynamicsBody(
                link_name=child_link,
                parent=len(bodies) - 1,
                joint_type=j.joint_type if j.joint_type in ("revolute", "prismatic") else "fixed",
                joint_origin_xyz=np.array(j.origin_xyz, dtype=np.float64),
                joint_origin_rpy=np.array(j.origin_rpy, dtype=np.float64),
                joint_axis=axis,
                q_index=q_index,
                mass=float(mass),
                com_xyz=np.array(com_xyz, dtype=np.float64),
                inertia=np.array(I, dtype=np.float64),
            )
        )

    # Merge into last revolute link all descendants connected by fixed or non-chain joints (match Pinocchio reduced model).
    chain_joint_names = set(j.name for j in chain_joints)
    all_joints = getattr(robot_model.parsed_model, "joints", [])
    if chain_joints and all_joints and hasattr(all_joints[0], "parent"):
        parent_to_children: Dict[str, List[Tuple[str, Any]]] = {}
        for j in all_joints:
            parent_to_children.setdefault(j.parent, []).append((j.child, j))
        # Last actuated link: child of the last revolute/prismatic joint in the chain (e.g. link6, not tool0).
        last_revolute_link = None
        for j in reversed(chain_joints):
            if j.joint_type in ("revolute", "prismatic"):
                last_revolute_link = j.child
                break
        if last_revolute_link is None:
            last_revolute_link = chain_joints[-1].child
        body_index_by_link = {b.link_name: i for i, b in enumerate(bodies)}
        last_rev_idx = body_index_by_link.get(last_revolute_link)
        if last_rev_idx is not None:
            # BFS from last_revolute_link: collect (child_link, T_from_last_rev_to_child) for fixed or non-chain joints.
            merge_list: List[Tuple[str, np.ndarray]] = []
            q: deque = deque([(last_revolute_link, np.eye(4, dtype=np.float64))])
            while q:
                link, T_6_to_link = q.popleft()
                for child_link, j in parent_to_children.get(link, []):
                    if j.joint_type != "fixed" and j.name in chain_joint_names:
                        continue
                    R = _rpy_to_matrix(
                        float(j.origin_rpy[0]), float(j.origin_rpy[1]), float(j.origin_rpy[2])
                    )
                    T_link_child = _make_transform(R, np.asarray(j.origin_xyz, dtype=np.float64))
                    T_6_to_child = T_6_to_link @ T_link_child
                    merge_list.append((child_link, T_6_to_child))
                    q.append((child_link, T_6_to_child))
            if merge_list:
                # Direct (mass, COM, I_com) merge: more transparent than 6x6 spatial inertia.
                # Collect all contributors: (mass, com_in_parent, I_com_in_parent).
                bp = bodies[last_rev_idx]
                contributors: List[Tuple[float, np.ndarray, np.ndarray]] = [
                    (bp.mass, bp.com_xyz.copy(), bp.inertia.copy())
                ]
                for child_link, T_p_c in merge_list:
                    if child_link in body_index_by_link:
                        bc = bodies[body_index_by_link[child_link]]
                        mc, cc, Ic = bc.mass, bc.com_xyz, bc.inertia
                    else:
                        mass_c, com_c, _cr, Ic_raw = link_inertials.get(
                            child_link,
                            (0.0, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], np.zeros((3, 3), dtype=np.float64)),
                        )
                        mc, cc = float(mass_c), np.array(com_c, dtype=np.float64)
                        Ic = np.asarray(Ic_raw, dtype=np.float64)
                    if mc < 1e-12:
                        continue
                    R_c = T_p_c[:3, :3]
                    p_c = T_p_c[:3, 3]
                    # COM of child in parent frame
                    com_in_parent = R_c @ np.asarray(cc, dtype=np.float64) + p_c
                    # Rotate inertia to parent frame
                    I_in_parent = R_c @ np.asarray(Ic, dtype=np.float64) @ R_c.T
                    contributors.append((mc, com_in_parent, I_in_parent))
                # Compute merged mass and COM
                m_total = sum(mi for mi, _, _ in contributors)
                if m_total > 1e-12:
                    c_total = sum(mi * ci for mi, ci, _ in contributors) / m_total
                else:
                    c_total = np.zeros(3, dtype=np.float64)
                # Compute merged inertia at c_total using parallel axis theorem
                I_total = np.zeros((3, 3), dtype=np.float64)
                for mi, ci, Ii in contributors:
                    d = ci - c_total
                    I_total += Ii + mi * (np.dot(d, d) * np.eye(3) - np.outer(d, d))
                bodies[last_rev_idx] = DynamicsBody(
                    link_name=bodies[last_rev_idx].link_name,
                    parent=bodies[last_rev_idx].parent,
                    joint_type=bodies[last_rev_idx].joint_type,
                    joint_origin_xyz=bodies[last_rev_idx].joint_origin_xyz,
                    joint_origin_rpy=bodies[last_rev_idx].joint_origin_rpy,
                    joint_axis=bodies[last_rev_idx].joint_axis,
                    q_index=bodies[last_rev_idx].q_index,
                    mass=m_total,
                    com_xyz=c_total,
                    inertia=I_total,
                )
                for child_link, _ in merge_list:
                    if child_link in body_index_by_link:
                        i0 = body_index_by_link[child_link]
                        b0 = bodies[i0]
                        bodies[i0] = DynamicsBody(
                            link_name=b0.link_name,
                            parent=b0.parent,
                            joint_type=b0.joint_type,
                            joint_origin_xyz=b0.joint_origin_xyz,
                            joint_origin_rpy=b0.joint_origin_rpy,
                            joint_axis=b0.joint_axis,
                            q_index=b0.q_index,
                            mass=0.0,
                            com_xyz=np.zeros(3, dtype=np.float64),
                            inertia=np.zeros((3, 3), dtype=np.float64),
                        )

    # Merge remaining fixed-descendant inertias (in-chain fixed bodies) into parent.
    for i in range(len(bodies) - 1, 0, -1):
        b = bodies[i]
        if b.joint_type != "fixed":
            continue
        if b.mass < 1e-12:
            continue  # already zeroed out
        p = b.parent
        R = _rpy_to_matrix(float(b.joint_origin_rpy[0]), float(b.joint_origin_rpy[1]), float(b.joint_origin_rpy[2]))
        p_vec = np.asarray(b.joint_origin_xyz, dtype=np.float64)
        # Transform child COM and inertia to parent frame
        com_child_in_parent = R @ b.com_xyz + p_vec
        I_child_in_parent = R @ b.inertia @ R.T
        # Merge into parent using direct (mass, com, I_com)
        m_p = bodies[p].mass
        m_c = b.mass
        m_total = m_p + m_c
        if m_total > 1e-12:
            c_total = (m_p * bodies[p].com_xyz + m_c * com_child_in_parent) / m_total
        else:
            c_total = np.zeros(3, dtype=np.float64)
        I_total = np.zeros((3, 3), dtype=np.float64)
        for mi, ci, Ii in [(m_p, bodies[p].com_xyz, bodies[p].inertia), (m_c, com_child_in_parent, I_child_in_parent)]:
            d = ci - c_total
            I_total += Ii + mi * (np.dot(d, d) * np.eye(3) - np.outer(d, d))
        bodies[p] = DynamicsBody(
            link_name=bodies[p].link_name,
            parent=bodies[p].parent,
            joint_type=bodies[p].joint_type,
            joint_origin_xyz=bodies[p].joint_origin_xyz,
            joint_origin_rpy=bodies[p].joint_origin_rpy,
            joint_axis=bodies[p].joint_axis,
            q_index=bodies[p].q_index,
            mass=m_total,
            com_xyz=c_total,
            inertia=I_total,
        )
        # Zero out fixed body so it no longer contributes in id/fd.
        bodies[i] = DynamicsBody(
            link_name=b.link_name,
            parent=b.parent,
            joint_type=b.joint_type,
            joint_origin_xyz=b.joint_origin_xyz,
            joint_origin_rpy=b.joint_origin_rpy,
            joint_axis=b.joint_axis,
            q_index=b.q_index,
            mass=0.0,
            com_xyz=np.zeros(3, dtype=np.float64),
            inertia=np.zeros((3, 3), dtype=np.float64),
        )

    nq = len(chain_dof_list)
    return DynamicsModel(nq=nq, bodies=bodies)
