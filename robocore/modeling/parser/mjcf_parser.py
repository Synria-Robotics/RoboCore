"""Minimal MJCF parser.

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

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from .urdf_parser import URDFJoint  # Re‑use the same dataclass expected by RobotModel


def _parse_floats(s: Optional[str], n: int, default: float = 0.0) -> List[float]:
	if not s:
		return [default] * n
	return [float(x) for x in s.strip().split()]  # type: ignore[arg-type]


def _quat_to_rpy(q):
	"""Convert quaternion (w, x, y, z) to RPY (XYZ intrinsic) radians."""
	w, x, y, z = q
	# Reference standard conversion
	# roll (x)
	sinr_cosp = 2 * (w * x + y * z)
	cosr_cosp = 1 - 2 * (x * x + y * y)
	roll = math.atan2(sinr_cosp, cosr_cosp)
	# pitch (y)
	sinp = 2 * (w * y - z * x)
	if abs(sinp) >= 1:
		pitch = math.copysign(math.pi / 2, sinp)
	else:
		pitch = math.asin(sinp)
	# yaw (z)
	siny_cosp = 2 * (w * z + x * y)
	cosy_cosp = 1 - 2 * (y * y + z * z)
	yaw = math.atan2(siny_cosp, cosy_cosp)
	return [roll, pitch, yaw]


def load_mjcf(path: str | Path) -> Dict[str, object]:
	"""Load MJCF (MuJoCo XML) and return a URDF‑like structure.

	:param path: file path to .xml.
	:return: dict with keys: 'name', 'joints', 'base_links'
	"""
	path = str(path)
	tree = ET.parse(path)
	root = tree.getroot()
	if root.tag != "mujoco":
		raise ValueError("Not a MJCF file: root tag != <mujoco>")
	model_name = root.attrib.get("model", Path(path).stem)
	worldbody = root.find("worldbody")
	if worldbody is None:
		raise ValueError("MJCF missing <worldbody> element")

	# Collect all bodies and build parent map recursively
	def body_name(elem: ET.Element) -> str:
		return elem.attrib.get("name", f"body_{id(elem)}")

	parent_map: Dict[ET.Element, Optional[str]] = {}
	all_bodies: List[ET.Element] = []

	# Recursive traversal to build parent_map and collect all bodies
	def build_tree(body_elem, parent_name):
		all_bodies.append(body_elem)
		parent_map[body_elem] = parent_name
		for child_body in body_elem.findall("body"):
			build_tree(child_body, body_name(body_elem))

	for b in worldbody.findall("body"):
		build_tree(b, None)  # top-level parent is None (world)

	if not all_bodies:
		raise ValueError("No bodies found in MJCF worldbody")

	# Extract joints from ALL bodies (not just the longest chain)
	# This allows us to capture branching structures like dual grippers
	joints: List[URDFJoint] = []
	parents = set()
	children = set()

	for body in all_bodies:
		bname = body_name(body)
		parent_name = parent_map[body] or "world"

		# Pose of body frame relative to parent
		pos = _parse_floats(body.attrib.get("pos"), 3)
		if "euler" in body.attrib:
			rpy = _parse_floats(body.attrib.get("euler"), 3)
		elif "quat" in body.attrib:
			q = _parse_floats(body.attrib.get("quat"), 4)
			rpy = _quat_to_rpy(q)
		else:
			rpy = [0.0, 0.0, 0.0]

		# Extract ALL supported joints in this body (not just the first one)
		for joint_elem in body.findall("joint"):
			jtype = joint_elem.attrib.get("type", "hinge")
			if jtype not in ("hinge", "slide"):
				continue  # Skip unsupported joint types

			jname = joint_elem.attrib.get("name", f"joint_{len(joints)}")
			if jtype == "hinge":
				joint_type = "revolute"
			elif jtype == "slide":
				joint_type = "prismatic"
			else:
				joint_type = "fixed"

			axis = _parse_floats(joint_elem.attrib.get("axis"), 3)

			# Extract limits
			range_attr = joint_elem.attrib.get("range")
			lower = upper = None
			if range_attr:
				vals = _parse_floats(range_attr, 2)
				if len(vals) == 2:
					lower, upper = vals

			joints.append(
				URDFJoint(
					name=jname,
					joint_type=joint_type,
					parent=parent_name,
					child=bname,
					axis=axis,
					origin_xyz=pos,
					origin_rpy=rpy,
					limit_lower=lower,
					limit_upper=upper,
				)
			)
			parents.add(parent_name)
			children.add(bname)

	base_links = list(parents - children) or ["world"]
	return {"name": model_name, "joints": joints, "base_links": base_links}


__all__ = ["load_mjcf"]
