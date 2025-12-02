"""URDF parser class implementation.

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

import os
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from robocore.modeling.parser.utils import JointSpec
from robocore.modeling.parser.urdf_parser.urdf import URDF as URDFRobot
from robocore.utils.path import create_dir, list_absl_path


def _parse_float_list(s: str) -> List[float]:
    """Parse a space-separated string of floats.

    :param s: Space-separated string of floats
    :return: List of floats
    """
    return [float(x) for x in s.strip().split()] if s else [0.0, 0.0, 0.0]


class URDFParser:
    """URDF parser class with unified interface matching MJCFParser."""

    def __init__(self, urdf_path: str | Path):
        """
        :param urdf_path: Path to the URDF file
        """
        self.urdf_path = Path(urdf_path)
        if not self.urdf_path.exists():
            raise FileNotFoundError(f"URDF file not found: {urdf_path}")

        # Parse joint information
        self.joints = self._parse_joints()
        self.link_mesh_map = {}
        self.link_virtual_map = {}

    def _parse_joints(self) -> List[JointSpec]:
        """
        :return: List of JointSpec objects
        """
        tree = ET.parse(str(self.urdf_path))
        root = tree.getroot()
        joint_elems = root.findall("joint")
        joints: List[JointSpec] = []
        parents = set()
        children = set()

        for idx, je in enumerate(joint_elems):
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
                JointSpec(
                    name=jname,
                    index=idx,
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
        return joints

    def get_joint_names(self) -> List[JointSpec]:
        """
        :return: List of JointSpec objects
        """
        return self.joints

    def get_joint_limits(self) -> np.ndarray:
        """
        :return: Array of joint limits with shape (n_joints, 2) where each row is [lower, upper]
        """
        return np.array([(j.limit_lower, j.limit_upper) for j in self.joints])

    def get_link_names(self) -> List[str]:
        """
        :return: List of link names
        """
        tree = ET.parse(str(self.urdf_path))
        root = tree.getroot()
        link_elems = root.findall("link")
        link_names = []
        for link_elem in link_elems:
            link_name = link_elem.attrib.get("name", "")
            if link_name:
                link_names.append(link_name)
        return link_names

    def get_link_virtual_map(self) -> Tuple[Dict[str, List[str]], Dict[str, str]]:
        """
        :return: Tuple of (link_virtual_map, inverse_link_virtual_map)
        """
        all_links = self.get_link_names()
        self.link_virtual_map = {}
        for link in all_links:
            if "world" in link.lower():
                continue
            else:
                self.link_virtual_map[link] = [link]

        self.inverse_link_virtual_map = {v: k for k, vs in self.link_virtual_map.items() for v in vs}
        return self.link_virtual_map, self.inverse_link_virtual_map

    def get_real_link_names(self) -> List[str]:
        """
        :return: List of real link names
        """
        if not self.link_virtual_map:
            self.get_link_virtual_map()
        return list(self.link_virtual_map.keys())

    def get_link_mesh_map(self) -> Dict[str, Dict[str, Dict]]:
        """
        Get the map of link and its corresponding geometries from the URDF file.

        :return: {link_body_name: {geom_name: geometry_info}}
        """
        robot = URDFRobot.from_xml_file(str(self.urdf_path))
        model_dir = os.path.dirname(str(self.urdf_path))

        # Process each link
        for link in robot.links:
            self.link_mesh_map[link.name] = {}

            # Process collision geometries
            if link.collision is not None:
                collisions = link.collision if isinstance(link.collision, list) else [link.collision]
                for idx, collision in enumerate(collisions):
                    if collision.geometry is None:
                        continue

                    geom = collision.geometry
                    geom_name = f"{link.name}_collision_{idx}"

                    # Get origin (position and orientation)
                    if collision.origin is not None:
                        geom_pos = collision.origin.xyz if collision.origin.xyz is not None else [0, 0, 0]
                        geom_rpy = collision.origin.rpy if collision.origin.rpy is not None else [0, 0, 0]
                    else:
                        geom_pos = [0, 0, 0]
                        geom_rpy = [0, 0, 0]

                    # Handle different geometry types
                    if hasattr(geom, 'mesh') and geom.mesh is not None:
                        # Mesh geometry
                        mesh_filename = geom.mesh.filename
                        if mesh_filename:
                            # Resolve mesh path
                            if not os.path.isabs(mesh_filename):
                                mesh_path = os.path.join(model_dir, mesh_filename)
                            else:
                                mesh_path = mesh_filename

                            # Check if mesh file exists
                            if os.path.exists(mesh_path):
                                mesh_scale = geom.mesh.scale if hasattr(geom.mesh, 'scale') and geom.mesh.scale else [1, 1, 1]
                                self.link_mesh_map[link.name][geom_name] = {
                                    'type': 'mesh',
                                    'params': {
                                        'mesh_path': mesh_path,
                                        'name': geom_name,
                                        'position': geom_pos,
                                        'quaternion': [1, 0, 0, 0],  # URDF uses RPY, not quaternion
                                        'rpy': geom_rpy,
                                        'scale': mesh_scale
                                    }
                                }

                    elif hasattr(geom, 'sphere') and geom.sphere is not None:
                        # Sphere geometry
                        radius = geom.sphere.radius
                        self.link_mesh_map[link.name][geom_name] = {
                            'type': 'sphere',
                            'params': {
                                'radius': radius,
                                'position': geom_pos,
                                'name': geom_name
                            }
                        }

                    elif hasattr(geom, 'cylinder') and geom.cylinder is not None:
                        # Cylinder geometry
                        radius = geom.cylinder.radius
                        length = geom.cylinder.length
                        self.link_mesh_map[link.name][geom_name] = {
                            'type': 'cylinder',
                            'params': {
                                'radius': radius,
                                'height': length,
                                'position': geom_pos,
                                'name': geom_name
                            }
                        }

                    elif hasattr(geom, 'box') and geom.box is not None:
                        # Box geometry
                        size = geom.box.size
                        self.link_mesh_map[link.name][geom_name] = {
                            'type': 'box',
                            'params': {
                                'extents': size,
                                'position': geom_pos,
                                'name': geom_name
                            }
                        }

        return self.link_mesh_map

    def get_link_meshname_map(self) -> Dict[str, List[str]]:
        """
        :return: {link_body_name: [mesh_name]}
        """
        link_meshname_map = {}
        for link, geoms in self.link_mesh_map.items():
            link_meshname_map[link] = []
            for geom in geoms:
                link_meshname_map[link].append(self.link_mesh_map[link][geom]['params']['name'])
        return link_meshname_map

    def num_joints(self) -> int:
        """
        :return: Number of joints
        """
        return len(self.joints)

    def to_dict(self) -> Dict[str, object]:
        """
        :return: Dictionary with 'name', 'joints', and 'base_links' keys
        """
        tree = ET.parse(str(self.urdf_path))
        root = tree.getroot()
        robot_name = root.attrib.get("name", "")

        # Find base links (parents that are never children)
        parents = set(j.parent for j in self.joints)
        children = set(j.child for j in self.joints)
        base_links = list(parents - children)

        return {
            "name": robot_name,
            "joints": self.joints,
            "base_links": base_links
        }


def load_urdf(path: str | Path) -> Dict[str, object]:
    """Load URDF and return structure (convenience function).

    :param path: file path.
    :return: dict keys: 'name','joints','base_links'.
    """
    parser = URDFParser(path)
    return parser.to_dict()


__all__ = ["URDFParser", "load_urdf"]

