"""Analytic (closed-form) Jacobian using NumPy.

Provides a 6×n Jacobian consistent with the current IK error definition:
  e = [ p_target - p_current ; axis_angle(R_current -> R_target) ]

The numeric Jacobian implementation approximates derivative columns via
finite differences on the pose (position) and axis-angle orientation error
relative to the CURRENT pose (i.e. R_ref = R_current). For small joint
perturbations δq the orientation error produced numerically is simply the
joint screw axis direction scaled by δq (for revolute joints). Thus the
closed-form orientation block equals the geometric angular velocity part.

This module computes the geometric Jacobian columns analytically and returns
them in the same convention used by the numeric version (derivative of the
CURRENT pose representation w.r.t joints):

For each actuated joint i (in world frame):
  Revolute:
	Jv_i = z_i × (p_end - p_i)
	Jw_i = z_i
  Prismatic:
	Jv_i = z_i
	Jw_i = 0

Where z_i is the (normalized) joint axis expressed in the world frame after
applying the parent link transform and the joint origin transform (URDF's
<origin rpy, xyz>). The axis direction is unaffected by the joint's own
rotation about itself, so we evaluate it at the joint origin frame.

NOTE:
- No target pose is required; this is purely configuration dependent.
- The orientation part matches the small-angle limit of the numeric
  axis-angle derivative, so IK convergence characteristics remain consistent.
- If later a true analytic mapping of axis-angle error (with Log map left
  Jacobian J_l^{-1}(phi)) is desired, an additional post-multiplication of
  the rotational block by J_l^{-1}(phi_current) can be added externally.

Returns:
  J (np.ndarray, shape (6, n))
"""

from __future__ import annotations

from typing import TYPE_CHECKING
import numpy as np
import math

if TYPE_CHECKING:
	from robocore.modeling.robot_model import RobotModel, JointSpec

__all__ = ["analytic_jacobian_numpy"]


def analytic_jacobian_numpy(model: "RobotModel", q) -> np.ndarray:
	"""Compute analytic (geometric) Jacobian 6×n for the current configuration.

	The returned matrix matches the numeric Jacobian's convention so it can
	be substituted directly in the IK solver.

	Parameters
	----------
	model : RobotModel
		Robot model instance (serial chain).
	q : array-like (n,)
		Joint configuration (actuated DoFs order).

	Returns
	-------
	J : np.ndarray (6, n)
		Analytic Jacobian (top 3 rows linear, bottom 3 rows orientation).
	"""
	if not hasattr(model, "_chain_joints"):
		raise AttributeError("RobotModel missing internal chain joints; incompatible instance")

	q = np.asarray(q, dtype=np.float64)
	n = model.dof()
	if q.shape[0] != n:
		raise ValueError(f"Configuration length {q.shape[0]} != dof {n}")

	# Build forward transforms collecting joint origins and axes in world frame.
	# We'll reconstruct lightweight FK to avoid list->np conversions inside model.forward_kinematics.
	T_parent = np.eye(4, dtype=np.float64)  # base link pose assumed identity
	# Maps needed for quick joint value lookup
	q_map = {js.name: q[js.index] for js in model._actuated}

	# Storage for each actuated joint: origin position p_i and axis z_i (world frame)
	p_list = [None] * n  # type: ignore
	z_list = [None] * n  # type: ignore

	# We also need end-effector pose; compute along the chain.
	end_T = T_parent

	# Helper lambdas (NumPy implementations)
	def rpy_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
		sr, cr = math.sin(roll), math.cos(roll)
		sp, cp = math.sin(pitch), math.cos(pitch)
		sy, cy = math.sin(yaw), math.cos(yaw)
		# R = Rz(yaw) * Ry(pitch) * Rx(roll)
		return np.array([
			[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
			[sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
			[-sp, cp * sr, cp * cr],
		], dtype=np.float64)

	def axis_rotation(axis, theta: float) -> np.ndarray:
		ax, ay, az = axis
		norm = math.sqrt(ax * ax + ay * ay + az * az) or 1.0
		ax, ay, az = ax / norm, ay / norm, az / norm
		ct = math.cos(theta)
		st = math.sin(theta)
		vt = 1.0 - ct
		return np.array([
			[ct + ax * ax * vt, ax * ay * vt - az * st, ax * az * vt + ay * st],
			[ay * ax * vt + az * st, ct + ay * ay * vt, ay * az * vt - ax * st],
			[az * ax * vt - ay * st, az * ay * vt + ax * st, ct + az * az * vt],
		], dtype=np.float64)

	def axis_translation(axis, d: float) -> np.ndarray:
		ax, ay, az = axis
		norm = math.sqrt(ax * ax + ay * ay + az * az) or 1.0
		ax, ay, az = ax / norm, ay / norm, az / norm
		return np.array([ax * d, ay * d, az * d], dtype=np.float64)

	# Iterate over chain joints (includes fixed/unactuated joints too)
	for urdf_joint in model._chain_joints:  # type: ignore[attr-defined]
		# Parent transform already in T_parent
		R_origin = rpy_matrix(*urdf_joint.origin_rpy)
		t_origin = np.array(urdf_joint.origin_xyz, dtype=np.float64)

		# Transform to joint origin
		T_origin = np.eye(4, dtype=np.float64)
		T_origin[:3, :3] = R_origin
		T_origin[:3, 3] = t_origin
		T_joint_origin = T_parent @ T_origin

		# If actuated, record axis & origin position BEFORE motion transform
		if urdf_joint.joint_type in ("revolute", "prismatic"):
			# Find its actuated spec to get index
			js: "JointSpec" = next(js for js in model._actuated if js.name == urdf_joint.name)  # type: ignore[attr-defined]
			axis_local = np.asarray(urdf_joint.axis, dtype=np.float64)
			# World axis after parent & origin rotation (joint self-rotation does not change axis direction)
			z_i = T_joint_origin[:3, :3] @ (axis_local / (np.linalg.norm(axis_local) or 1.0))
			p_i = T_joint_origin[:3, 3].copy()
			p_list[js.index] = p_i
			z_list[js.index] = z_i

		# Apply motion of this joint to get child pose
		R_motion = np.eye(3, dtype=np.float64)
		t_motion = np.zeros(3, dtype=np.float64)
		if urdf_joint.joint_type == "revolute":
			theta = q_map.get(urdf_joint.name, 0.0)
			R_motion = axis_rotation(urdf_joint.axis, theta)
		elif urdf_joint.joint_type == "prismatic":
			d = q_map.get(urdf_joint.name, 0.0)
			t_motion = axis_translation(urdf_joint.axis, d)

		T_motion = np.eye(4, dtype=np.float64)
		T_motion[:3, :3] = R_motion
		T_motion[:3, 3] = t_motion

		# Child link transform
		T_child = T_joint_origin @ T_motion
		T_parent = T_child  # Next iteration parent
		end_T = T_child  # Will end at end-effector

	p_end = end_T[:3, 3]

	# Assemble Jacobian (geometric world-frame)
	J_geo = np.zeros((6, n), dtype=np.float64)
	for i in range(n):
		z_i = z_list[i]; p_i = p_list[i]
		if z_i is None or p_i is None:
			raise RuntimeError("Internal error: missing joint axis or origin position while building Jacobian")
		js = model._actuated[i]
		if js.joint_type == "revolute":
			J_geo[:3, i] = np.cross(z_i, (p_end - p_i))
			J_geo[3:6, i] = z_i
		elif js.joint_type == "prismatic":
			J_geo[:3, i] = z_i

	# The numeric implementation's rotational rows correspond to the incremental
	# axis-angle vector aligned with END-EFFECTOR frame axes (observed empirically
	# via zero-config comparison). To match it, rotate angular part from world to
	# end-effector frame: omega_eef = R_end^T * omega_world.
	R_end = end_T[:3, :3]
	J = J_geo.copy()
	J[3:6, :] = R_end.T @ J_geo[3:6, :]
	return J

