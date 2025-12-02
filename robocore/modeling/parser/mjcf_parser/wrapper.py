"""MJCF parser using MuJoCo.

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
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np

import mujoco
import trimesh
from robocore.modeling.parser.utils import JointSpec
from robocore.transform.conversions import quaternion_to_rpy, quaternion_reorder
from robocore.modeling.parser.mjcf_parser.parser import from_path
from robocore.utils.path import create_dir, list_absl_path


class MJCFParser:
    def __init__(self, mjcf_path: str | Path):
        """
        :param mjcf_path: Path to the MJCF file
        
        Note: Initialization is slower than URDFParser (~200-250ms vs ~1ms) because
        MuJoCo needs to load and compile the complete physics model. This is necessary
        to extract joint information from the MJCF format. The MuJoCo model includes:
        - Full physics parameters (mass, inertia, collision geometries)
        - Compiled model for simulation
        - Complete kinematic and dynamic structure
        """
        self.mjcf_path = Path(mjcf_path)
        if not self.mjcf_path.exists():
            raise FileNotFoundError(f"MJCF file not found: {mjcf_path}")

        # Load the MuJoCo model (this is the main performance bottleneck)
        # MuJoCo needs to parse XML, build physics model, and compile it
        self.model = mujoco.MjModel.from_xml_path(str(self.mjcf_path))
        self.data = mujoco.MjData(self.model)

        # Parse joint information (requires self.model to be loaded)
        self.joints = self._parse_joints()
        self.link_mesh_map = {}

    def _get_joint_type(self, jnt_type: int) -> str:
        """
        :param jnt_type: MuJoCo joint type constant
        :return: Joint type string
        """
        type_map = {
            mujoco.mjtJoint.mjJNT_FREE: "floating",
            mujoco.mjtJoint.mjJNT_BALL: "ball",
            mujoco.mjtJoint.mjJNT_SLIDE: "prismatic",
            mujoco.mjtJoint.mjJNT_HINGE: "revolute",
        }
        return type_map.get(jnt_type, "fixed")

    def _parse_joints(self) -> List[JointSpec]:
        """
        :return: List of JointSpec objects
        """
        joints = []
        idx = 0

        for i in range(self.model.njnt):
            # Get joint name
            joint_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
            if joint_name is None:
                joint_name = f"joint_{i}"

            # Get joint type
            jnt_type = self.model.jnt_type[i]
            joint_type = self._get_joint_type(jnt_type)

            # Get body IDs
            body_id = self.model.jnt_bodyid[i]
            parent_body_id = self.model.body_parentid[body_id]

            # Get body names
            child = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body_id)
            if child is None:
                child = f"body_{body_id}"

            parent = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, parent_body_id)
            if parent is None:
                parent = f"body_{parent_body_id}"

            # Get joint axis (in body frame)
            axis = self.model.jnt_axis[i].copy()
            axis = axis.tolist()

            # Get body's position and orientation relative to parent
            # MuJoCo stores body_pos as position relative to parent body
            # and body_quat as quaternion relative to parent body
            pos = self.model.body_pos[body_id].copy()
            origin_xyz = pos.tolist()

            # Get body's relative quaternion and convert to RPY
            # MuJoCo uses [w, x, y, z] format
            quat = self.model.body_quat[body_id].copy()
            origin_rpy = quaternion_to_rpy(quaternion_reorder(quat))

            # Get joint limits
            limit_lower = None
            limit_upper = None
            if self.model.jnt_limited[i]:
                limit_lower = float(self.model.jnt_range[i, 0])
                limit_upper = float(self.model.jnt_range[i, 1])

            joint = JointSpec(
                name=joint_name,
                index=idx,
                joint_type=joint_type,
                parent=parent,
                child=child,
                axis=axis,
                origin_xyz=origin_xyz,
                origin_rpy=origin_rpy,
                limit_lower=limit_lower,
                limit_upper=limit_upper
            )

            joints.append(joint)
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
        :return: List of link names (bodies)
        """
        link_names = []
        for i in range(self.model.nbody):
            body_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, i)
            if body_name is None:
                body_name = f"body_{i}"
            link_names.append(body_name)
        return link_names

    def get_link_virtual_map(self):
        """
        :return: {link_body_name: [virtual_link_0, virtual_link_1, ...]}
        """
        all_links = self.get_link_names()
        self.link_virtual_map = {}
        for link in all_links:
            if "world" in link:
                continue
            # if "_0" in link or "_1" in link or "_2" in link:
            #     link_name = link.split("_")[:-1]
            #     link_name = "_".join(link_name)
            #     if link_name not in link_virtual_map:
            #         link_virtual_map[link_name] = []
            #     link_virtual_map[link_name].append(link)
            else:
                self.link_virtual_map[link] = [link]

        self.inverse_link_virtual_map = {v: k for k, vs in self.link_virtual_map.items() for v in vs}
        return self.link_virtual_map, self.inverse_link_virtual_map

    def get_real_link_names(self):
        """
        :return: [real_link_0, real_link_1, ...]
        """
        return list(self.link_virtual_map.keys())

    def get_link_mesh_map(self):
        """
        Get the map of link and its corresponding geometries from the MJCF file.

        :return: {link_body_name: {geom_name: mesh_path}}
        """
        robot = from_path(self.mjcf_path)
        bodies = robot.find_all("body")
        robot.compiler.meshdir = robot.compiler.meshdir or "meshes"
        mesh_dir = os.path.join(robot.namescope.model_dir, robot.compiler.meshdir)
        create_dir(mesh_dir)
        all_mesh_file_stl = list_absl_path(mesh_dir, recursive=True, suffix=".stl")
        all_mesh_file_STL = list_absl_path(mesh_dir, recursive=True, suffix=".STL")
        all_mesh_file_obj = list_absl_path(mesh_dir, recursive=True, suffix=".obj")
        all_mesh_file_OBJ = list_absl_path(mesh_dir, recursive=True, suffix=".OBJ")
        all_mesh_files = all_mesh_file_stl + all_mesh_file_STL + all_mesh_file_obj + all_mesh_file_OBJ

        mesh_map = robot.get_assets_map()
        mesh_name_path_map = {}
        for mesh_name, mesh_file in mesh_map.items():
            mesh_path = None
            for mesh_file_exist in all_mesh_files:
                # Use basename for exact filename match to avoid substring issues
                # e.g., "base_link.STL" should not match "right_arm_base_link.STL"
                if os.path.basename(mesh_file_exist) == mesh_file:
                    mesh_path = mesh_file_exist
                    break  # Stop at first match
            if mesh_path is not None:
                mesh_name_path_map[mesh_name] = mesh_path
            else:
                raise FileNotFoundError(f"Mesh file {mesh_file} not found in the mesh directory.")

        meshes = robot.find_all("mesh")

        # 遍历所有 bodies，处理几何体
        for body in bodies:
            geoms_this_body = body.geom
            self.link_mesh_map[body.name] = {}

            for geom in geoms_this_body:
                # Filter out visual geometries (contype="0" conaffinity="0")
                # Only keep collision geometries for distance field computation
                geom_contype = getattr(geom, 'contype', None)
                geom_conaffinity = getattr(geom, 'conaffinity', None)

                # Skip visual geometries (with contype="0" and conaffinity="0")
                # Note: attributes might be strings "0" or numbers 0
                is_visual = (str(geom_contype) == "0" and
                             str(geom_conaffinity) == "0") if (geom_contype is not None and geom_conaffinity is not None) else False

                if is_visual:
                    continue

                geom_type = geom.type or "capsule"  # 默认类型为胶囊
                geom_pos = geom.pos if geom.pos is not None else [0, 0, 0]
                geom_quat = geom.quat if geom.quat is not None else [1, 0, 0, 0]  # w, x, y, z

                # 处理不同的几何体类型
                if geom_type == "mesh":
                    geom_mesh_name = geom.mesh.name
                    geom_mesh_path = mesh_name_path_map[geom_mesh_name]
                    mesh_scale = [1, 1, 1]
                    for mesh in meshes:
                        if mesh.name == "wheelchair_mesh":
                            mesh_scale = mesh.scale
                    self.link_mesh_map[body.name][geom_mesh_name] = {
                        'type': 'mesh',
                        'params': {'mesh_path': geom_mesh_path, 'name': geom_mesh_name, 'position': geom_pos,
                                   'quaternion': geom_quat, 'scale': mesh_scale}
                    }

                elif geom_type == "sphere":
                    geom_mesh_size = geom.size[0]  # 球体的大小是半径
                    self.link_mesh_map[body.name][geom.name] = {
                        'type': 'sphere',
                        'params': {'radius': geom_mesh_size, 'position': geom_pos, 'name': geom.name}
                    }

                elif geom_type == "cylinder":
                    geom_mesh_size = geom.size  # 圆柱体的大小是 [半径, 高度]
                    self.link_mesh_map[body.name][geom.name] = {
                        'type': 'cylinder',
                        'params': {'radius': geom_mesh_size[0], 'height': geom_mesh_size[1], 'position': geom_pos,
                                   'name': geom.name}
                    }

                elif geom_type == "box":
                    geom_mesh_size = geom.size  # 盒子的大小是 [x, y, z] 维度
                    self.link_mesh_map[body.name][geom.name] = {
                        'type': 'box',
                        'params': {'extents': geom_mesh_size, 'position': geom_pos, 'name': geom.name}
                    }

                elif geom_type == "capsule":
                    geom_mesh_size = geom.size  # 胶囊的半径储存在 size[0]
                    geom_fromto = geom.fromto  # 从fromto属性获取胶囊两端的坐标
                    if geom_fromto is None:
                        # Skip capsules without fromto defined
                        continue
                    from_point = geom_fromto[:3]  # 胶囊起点
                    to_point = geom_fromto[3:]  # 胶囊终点
                    # 计算胶囊的高度（两点之间的距离）
                    height = ((to_point[0] - from_point[0]) ** 2 +

                              (to_point[1] - from_point[1]) ** 2 +

                              (to_point[2] - from_point[2]) ** 2) ** 0.5
                    # 胶囊的参数化描述
                    self.link_mesh_map[body.name][geom.name] = {
                        'type': 'capsule',
                        'params': {
                            'radius': geom_mesh_size[0],  # 胶囊的半径
                            'height': height,  # 胶囊的高度
                            'from': from_point,  # 起点坐标
                            'to': to_point,  # 终点坐标
                            'name': geom.name,
                            "position": geom_pos
                        }
                    }

                else:
                    raise ValueError(f"Unsupported geometry type {geom_type}.")
        return self.link_mesh_map

    def get_link_meshname_map(self):
        """
        :return: {link_body_name: [mesh_name]}
        """
        link_meshname_map = {}
        for link, geoms in self.link_mesh_map.items():
            link_meshname_map[link] = []
            for geom in geoms:
                link_meshname_map[link].append(self.link_mesh_map[link][geom]['params']['name'])
        return link_meshname_map

    def get_robot_mesh(self, vertices_list, faces):
        assert len(vertices_list) == len(faces), "The number of vertices and faces should be the same."
        robot_mesh = [trimesh.Trimesh(verts, face) for verts, face in zip(vertices_list, faces)]
        return robot_mesh

    def num_joints(self) -> int:
        return len(self.joints)

    def to_dict(self) -> Dict[str, object]:
        """
        :return: Dictionary with 'name', 'joints', and 'base_links' keys
        """
        # Get robot name from MuJoCo model
        robot_name = self.model.names.decode('utf-8').split('\x00')[0] if self.model.names else ""

        # Convert joints dict to list for compatibility
        joints_list = list(self.joints.values())

        # Find base links (parents that are never children)
        parents = set(j.parent for j in joints_list)
        children = set(j.child for j in joints_list)
        base_links = list(parents - children)

        return {
            "name": robot_name,
            "joints": joints_list,
            "base_links": base_links
        }


def load_mjcf(path: str | Path) -> Dict[str, object]:
    """
    :param path: Path to the MJCF file
    :return: Dictionary with keys 'name', 'joints', 'base_links'
    """
    parser = MJCFParser(path)
    return parser.to_dict()


__all__ = ["MJCFParser", "load_mjcf"]
