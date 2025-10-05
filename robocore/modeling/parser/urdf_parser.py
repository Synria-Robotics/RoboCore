"""URDF parser minimal implementation.

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

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class URDFJoint:
    """Joint info.

    :param name: joint name.
    :param joint_type: 'revolute', 'prismatic', 'fixed'.
    :param parent: parent link.
    :param child: child link.
    :param axis: axis vector (3,).
    :param origin_xyz: translation (3,).
    :param origin_rpy: rpy (3, radians).
    :param limit_lower: lower limit or None.
    :param limit_upper: upper limit or None.
    """

    name: str
    joint_type: str
    parent: str
    child: str
    axis: List[float]
    origin_xyz: List[float]
    origin_rpy: List[float]
    limit_lower: Optional[float]
    limit_upper: Optional[float]


def _parse_float_list(s: str) -> List[float]:
    return [float(x) for x in s.strip().split()] if s else [0.0, 0.0, 0.0]


def load_urdf(path: str | Path) -> Dict[str, object]:
    """Load URDF and return structure.

    :param path: file path.
    :return: dict keys: 'name','joints','base_links'.
    """
    path = str(path)
    tree = ET.parse(path)
    root = tree.getroot()
    robot_name = root.attrib.get("name", "")
    joint_elems = root.findall("joint")
    joints: List[URDFJoint] = []
    parents = set()
    children = set()
    for je in joint_elems:
        jname = je.attrib["name"]
        jtype = je.attrib.get("type", "fixed")
        parent = je.find("parent").attrib["link"]
        child = je.find("child").attrib["link"]
        origin_elem = je.find("origin")
        if origin_elem is not None:
            xyz = _parse_float_list(origin_elem.attrib.get("xyz", "0 0 0"))
            rpy = _parse_float_list(origin_elem.attrib.get("rpy", "0 0 0"))
        else:
            xyz = [0.0, 0.0, 0.0]
            rpy = [0.0, 0.0, 0.0]
        axis_elem = je.find("axis")
        axis = _parse_float_list(axis_elem.attrib.get("xyz", "0 0 1")) if axis_elem is not None else [0.0, 0.0, 1.0]
        limit_elem = je.find("limit")
        lower = float(limit_elem.attrib["lower"]) if (limit_elem is not None and "lower" in limit_elem.attrib) else None
        upper = float(limit_elem.attrib["upper"]) if (limit_elem is not None and "upper" in limit_elem.attrib) else None
        parents.add(parent)
        children.add(child)
        joints.append(
            URDFJoint(
                name=jname,
                joint_type=jtype,
                parent=parent,
                child=child,
                axis=axis,
                origin_xyz=xyz,
                origin_rpy=rpy,
                limit_lower=lower,
                limit_upper=upper,
            )
        )
    base_links = list(parents - children)
    return {"name": robot_name, "joints": joints, "base_links": base_links}


__all__ = ["URDFJoint", "load_urdf"]
