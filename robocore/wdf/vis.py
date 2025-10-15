"""
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

import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import trimesh
import skimage
from robocore.wdf import utils


def plot_2D_sdf(joint_value, rdf_bp, nbData, model, device, used_links=None):
    robot_mesh = rdf_bp.robot.get_forward_robot_mesh(joint_value)[0]

    if isinstance(robot_mesh, trimesh.Trimesh):
        # 获取机器人模型的网格
        robot_mesh = np.sum(robot_mesh)
        # 获取机器人网格的边界，用于定义矩形空间
        bbox_min = robot_mesh.bounds[0]  # 最小边界 (X_min, Y_min, Z_min)
        bbox_max = robot_mesh.bounds[1]  # 最大边界 (X_max, Y_max, Z_max)
    else:
        bbox_min = [-1, -1, 0]
        bbox_max = [1, 1, 2]

    # 在边界外扩展一些空间
    margin = 0.1
    x_range = (bbox_min[0] - margin, bbox_max[0] + margin)
    y_range = (bbox_min[1] - margin, bbox_max[1] + margin)
    z_range = (bbox_min[2] - margin, bbox_max[2] + margin)

    # 在矩形空间内生成均匀分布的点
    domain_x = torch.linspace(y_range[0], y_range[1], nbData).to(device)
    domain_y = torch.linspace(z_range[0], z_range[1], nbData).to(device)
    grid_x, grid_y = torch.meshgrid(domain_x, domain_y)

    # 在矩形空间内定义采样点
    p1 = torch.stack([grid_x.reshape(-1), grid_y.reshape(-1), torch.zeros_like(grid_x.reshape(-1))], dim=1)
    p2 = torch.stack([torch.zeros_like(grid_x.reshape(-1)), grid_x.reshape(-1), grid_y.reshape(-1)], dim=1)
    p3 = torch.stack([grid_x.reshape(-1), torch.zeros_like(grid_x.reshape(-1)), grid_y.reshape(-1)], dim=1)

    grid_x, grid_y = grid_x.detach().cpu().numpy(), grid_y.detach().cpu().numpy()

    # 绘制第一个图像 (YoZ 平面)
    plt.figure(figsize=(10, 10))
    plt.rc('font', size=25)
    p2_split = torch.split(p2, 1000, dim=0)
    sdf, ana_grad = [], []
    for p_2 in p2_split:
        sdf_split, ana_grad_split = rdf_bp.get_whole_body_sdf_batch(p_2, joint_value, model, use_derivative=True,
                                                                    used_links=used_links)
        sdf_split, ana_grad_split = sdf_split.squeeze(), ana_grad_split.squeeze()
        sdf.append(sdf_split)
        ana_grad.append(ana_grad_split)
    sdf = torch.cat(sdf, dim=0)
    ana_grad = torch.cat(ana_grad, dim=0)
    p2 = p2.detach().cpu().numpy()
    sdf = sdf.squeeze().reshape(nbData, nbData).detach().cpu().numpy()

    ct1 = plt.contour(grid_x, grid_y, sdf, levels=18)
    plt.clabel(ct1, inline=False, fontsize=10)
    ana_grad_2d = -torch.nn.functional.normalize(ana_grad[:, [1, 2]], dim=-1) * 0.01
    p2_3d = p2.reshape(nbData, nbData, 3)
    ana_grad_3d = ana_grad_2d.reshape(nbData, nbData, 2)

    plt.quiver(p2_3d[0:-1:4, 0:-1:4, 1], p2_3d[0:-1:4, 0:-1:4, 2],
               ana_grad_3d[0:-1:4, 0:-1:4, 0].detach().cpu().numpy(),
               ana_grad_3d[0:-1:4, 0:-1:4, 1].detach().cpu().numpy(), scale=0.5, color=[0.1, 0.1, 0.1])
    plt.xlim([y_range[0], y_range[1]])
    plt.ylim([z_range[0], z_range[1]])
    plt.title('YoZ')
    plt.show()

    # 绘制第二个图像 (XoZ 平面)
    plt.figure(figsize=(10, 10))
    plt.rc('font', size=25)
    p3_split = torch.split(p3, 1000, dim=0)
    sdf, ana_grad = [], []
    for p_3 in p3_split:
        sdf_split, ana_grad_split = rdf_bp.get_whole_body_sdf_batch(p_3, joint_value, model, use_derivative=True,
                                                                    used_links=used_links)
        sdf_split, ana_grad_split = sdf_split.squeeze(), ana_grad_split.squeeze()
        sdf.append(sdf_split)
        ana_grad.append(ana_grad_split)
    sdf = torch.cat(sdf, dim=0)
    ana_grad = torch.cat(ana_grad, dim=0)
    p3 = p3.detach().cpu().numpy()
    sdf = sdf.squeeze().reshape(nbData, nbData).detach().cpu().numpy()

    ct1 = plt.contour(grid_x, grid_y, sdf, levels=18)
    plt.clabel(ct1, inline=False, fontsize=10)

    ana_grad_2d = -torch.nn.functional.normalize(ana_grad[:, [0, 2]], dim=-1) * 0.01
    p3_3d = p3.reshape(nbData, nbData, 3)
    ana_grad_3d = ana_grad_2d.reshape(nbData, nbData, 2)

    plt.quiver(p3_3d[0:-1:4, 0:-1:4, 0], p3_3d[0:-1:4, 0:-1:4, 2],
               ana_grad_3d[0:-1:4, 0:-1:4, 0].detach().cpu().numpy(),
               ana_grad_3d[0:-1:4, 0:-1:4, 1].detach().cpu().numpy(), scale=0.5, color=[0.1, 0.1, 0.1])
    plt.title('XoZ')
    plt.show()


def plot_3D_sdf_with_gradient_surface_points(joint_value, rdf_bp, model, device, used_links=None):
    robot_mesh = rdf_bp.robot.get_forward_robot_mesh(joint_value)[0]
    robot_mesh = np.sum(robot_mesh)

    surface_points = robot_mesh.vertices
    scene = trimesh.Scene()
    # robot mesh
    scene.add_geometry(robot_mesh)
    choice = np.random.choice(len(surface_points), 1024, replace=False)
    surface_points = surface_points[choice]
    p = torch.from_numpy(surface_points).float().to(device)
    ball_query = trimesh.creation.uv_sphere(1).vertices
    choice_ball = np.random.choice(len(ball_query), 1024, replace=False)
    ball_query = ball_query[choice_ball]
    p = p + torch.from_numpy(ball_query).float().to(device) * 0.5
    sdf, ana_grad = rdf_bp.get_whole_body_sdf_batch(p, joint_value, model, use_derivative=True, used_links=used_links)
    sdf, ana_grad = sdf.squeeze().detach().cpu().numpy(), ana_grad.squeeze().detach().cpu().numpy()
    # points
    pts = p.detach().cpu().numpy()
    colors = np.zeros_like(pts, dtype=object)
    colors[:, 0] = np.abs(sdf) * 400
    # pc =trimesh.PointCloud(pts,colors)
    # scene.add_geometry(pc)

    # gradients
    for i in range(len(pts)):
        dg = ana_grad[i]
        if dg.sum() == 0:
            continue
        c = colors[i]
        if np.any(c > 255):
            c = [255, 0, 0]
        # print(c)
        m = utils.create_arrow(-dg, pts[i], vec_length=0.05, color=c)
        scene.add_geometry(m)
    scene.show()


def plot_3D_sdf_with_gradient(joint_value, rdf_bp, model, device, used_links=None):
    """
    Plot 3D SDF with gradient arrows.

    :param joint_value: joint values, numpy array
    :param rdf_bp: RDF instance
    :param model: trained RDF model
    :param device: torch device
    :param used_links: list of link names to use, optional
    """
    robot_mesh = rdf_bp.robot.get_forward_robot_mesh(joint_value)[0]

    if isinstance(robot_mesh, trimesh.Trimesh):
        # 获取机器人模型的网格
        robot_mesh = np.sum(robot_mesh)
        # 获取机器人网格的边界，用于定义矩形空间
        bbox_min = robot_mesh.bounds[0]  # 最小边界 (X_min, Y_min, Z_min)
        bbox_max = robot_mesh.bounds[1]  # 最大边界 (X_max, Y_max, Z_max)
    else:
        bbox_min = [-0.5, -0.5, -0.5]
        bbox_max = [0.5, 0.5, 0.5]

    # 在边界外扩展一些空间
    margin = 0.1
    x_range = (bbox_min[0] - margin, bbox_max[0] + margin)
    y_range = (bbox_min[1] - margin, bbox_max[1] + margin)
    z_range = (bbox_min[2] - margin, bbox_max[2] + margin)

    # 定义采样点的密度
    nbData = 5  # 控制每个维度的采样点数量

    # 在矩形空间内生成均匀分布的采样点
    domain_x = torch.linspace(x_range[0], x_range[1], nbData).to(device)
    domain_y = torch.linspace(y_range[0], y_range[1], nbData).to(device)
    domain_z = torch.linspace(z_range[0], z_range[1], nbData).to(device)
    grid_x, grid_y, grid_z = torch.meshgrid(domain_x, domain_y, domain_z, indexing='ij')

    # 将 3D 网格点展开成二维数组以便批量处理, [B, N, 3]
    points = torch.stack([grid_x.reshape(-1), grid_y.reshape(-1), grid_z.reshape(-1)], dim=1).unsqueeze(0)

    # 计算 SDF 和梯度
    sdf, ana_grad = rdf_bp.get_whole_body_sdf_batch(points, joint_value, model, use_derivative=True,
                                                    used_links=used_links)
    sdf, ana_grad = sdf.squeeze().detach().cpu().numpy(), ana_grad.squeeze().detach().cpu().numpy()

    # 转为 NumPy 以便绘制
    pts = points.detach().cpu().numpy()

    # 创建场景并添加机器人网格
    scene = trimesh.Scene()
    scene.add_geometry(robot_mesh)

    # 渲染点基于 SDF 值的颜色
    colors = np.zeros((pts.shape[1], 3))
    colors[:, 0] = np.clip(np.abs(sdf) * 400, 0, 255)  # 使用红色通道表示 SDF 值

    # 绘制梯度箭头
    for i in range(len(pts[0])):
        dg = ana_grad[i]
        if np.linalg.norm(dg) == 0:
            continue
        c = colors[i]
        if np.any(c > 255):
            c = [255, 0, 0]
        arrow = utils.create_arrow(-dg, pts[0][i], vec_length=0.05, color=c)
        scene.add_geometry(arrow)

    # 显示场景
    scene.show()


def plot_3D_sdf_envelope(joint_value, rdf_bp, model, device, used_links=None,
                         nbData=64, distance_levels=None, show_robot=True):
    """
    Plot 3D SDF using envelope-style visualization with multiple distance field isosurfaces.
    
    :param joint_value: joint values, numpy array
    :param rdf_bp: RDF instance
    :param model: trained RDF model
    :param device: torch device
    :param used_links: list of link names to use, optional
    :param nbData: int, grid resolution for sampling
    :param distance_levels: list of float, distance levels to visualize, e.g. [0.02, 0.05, 0.1, 0.15]
    :param show_robot: bool, whether to show the robot mesh
    """
    print("Generating envelope-style distance field visualization...")

    # 获取机器人网格
    robot_mesh_list = rdf_bp.robot.get_forward_robot_mesh(joint_value)

    if isinstance(robot_mesh_list, list) and len(robot_mesh_list) > 0:
        robot_mesh = robot_mesh_list[0]
        if hasattr(robot_mesh, '__iter__') and not isinstance(robot_mesh, trimesh.Trimesh):
            robot_mesh = np.sum(robot_mesh)
        bbox_min = robot_mesh.bounds[0]
        bbox_max = robot_mesh.bounds[1]
    else:
        robot_mesh = None
        bbox_min = np.array([-0.5, -0.5, -0.5])
        bbox_max = np.array([0.5, 0.5, 1.5])

    # 扩展边界以便看到距离场
    margin = 0.2
    x_range = (bbox_min[0] - margin, bbox_max[0] + margin)
    y_range = (bbox_min[1] - margin, bbox_max[1] + margin)
    z_range = (bbox_min[2] - margin, bbox_max[2] + margin)

    # 生成3D网格
    print(f"Sampling {nbData}^3 = {nbData**3} points...")
    domain_x = torch.linspace(x_range[0], x_range[1], nbData).to(device)
    domain_y = torch.linspace(y_range[0], y_range[1], nbData).to(device)
    domain_z = torch.linspace(z_range[0], z_range[1], nbData).to(device)
    grid_x, grid_y, grid_z = torch.meshgrid(domain_x, domain_y, domain_z, indexing='ij')

    # 展开网格点
    points = torch.stack([grid_x.reshape(-1), grid_y.reshape(-1), grid_z.reshape(-1)], dim=1).unsqueeze(0)

    # 批量计算SDF值，处理内存问题
    batch_size = 10000
    sdf_list = []
    print("Computing SDF values...")
    for i in range(0, points.shape[1], batch_size):
        end_idx = min(i + batch_size, points.shape[1])
        points_batch = points[:, i:end_idx, :]
        sdf_batch, _ = rdf_bp.get_whole_body_sdf_batch(points_batch, joint_value, model,
                                                       use_derivative=False, used_links=used_links)
        sdf_list.append(sdf_batch.squeeze())

    sdf = torch.cat(sdf_list, dim=0).detach().cpu().numpy()
    sdf = sdf.reshape(nbData, nbData, nbData)

    # 设置默认的距离层级
    if distance_levels is None:
        # 自动选择合适的距离层级
        max_dist = np.max(np.abs(sdf))
        distance_levels = [0.02, 0.05, 0.1, 0.15, 0.2]
        distance_levels = [d for d in distance_levels if d < max_dist]
        if len(distance_levels) == 0:
            distance_levels = [max_dist * 0.3, max_dist * 0.5, max_dist * 0.7]

    print(f"Distance levels: {distance_levels}")

    # 创建场景
    scene = trimesh.Scene()

    # 添加机器人网格（不透明，深灰色）
    if show_robot and robot_mesh is not None:
        robot_mesh.visual.face_colors = [100, 100, 100, 255]
        scene.add_geometry(robot_mesh)

    # 定义颜色映射（从内到外：蓝色->绿色->黄色->红色）
    colors = [
        [100, 150, 255, 180],   # 浅蓝色
        [100, 255, 150, 140],   # 浅绿色
        [255, 255, 100, 100],   # 黄色
        [255, 150, 100, 80],    # 橙色
        [255, 100, 100, 60],    # 红色
    ]

    # 计算网格间距
    spacing = np.array([
        (x_range[1] - x_range[0]) / nbData,
        (y_range[1] - y_range[0]) / nbData,
        (z_range[1] - z_range[0]) / nbData
    ])

    # 为每个距离层级生成等值面
    print("Generating isosurfaces...")
    for i, level in enumerate(distance_levels):
        try:
            # 使用marching cubes算法生成等值面
            verts, faces, normals, values = skimage.measure.marching_cubes(
                sdf, level=level, spacing=spacing
            )

            # 调整顶点位置到世界坐标
            verts = verts + np.array([x_range[0], y_range[0], z_range[0]])

            # 创建网格
            iso_mesh = trimesh.Trimesh(vertices=verts, faces=faces)

            # 设置颜色和透明度
            color_idx = min(i, len(colors) - 1)
            iso_mesh.visual.face_colors = colors[color_idx]

            scene.add_geometry(iso_mesh)
            print(f"  Level {level:.3f}: {len(verts)} vertices, {len(faces)} faces")
        except Exception as e:
            print(f"  Warning: Failed to generate isosurface for level {level}: {e}")
            continue

    print("Visualization complete!")
    scene.show()
