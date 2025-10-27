#!/usr/bin/env python3
"""Demo 1: Independent Dual-Arm Control (Pure RobotModel version).

This version uses RobotModel instead of BimanualRobotModel to demonstrate
that BimanualRobotModel is just a convenience wrapper.

Drag red and blue spheres independently to control left and right arms.

Copyright (c) 2025 Synria Robotics Co., Ltd.
"""

import sys
from pathlib import Path
from typing import Optional, Dict, Tuple
import numpy as np
import time

try:
    import mujoco
    import mujoco.viewer
    MUJOCO_AVAILABLE = True
except ImportError:
    MUJOCO_AVAILABLE = False
    print("⚠️  MuJoCo not available. Install with: pip install mujoco")

from robocore.modeling.robot_model import RobotModel
from robocore.utils.path import get_robocore_path


class InteractiveDualArmIK:
    """Interactive dual-arm IK visualization with draggable targets (RobotModel version).
    
    :param mjcf_path: Path to MuJoCo MJCF/XML file
    :param left_end_link: Left arm end-effector link name
    :param right_end_link: Right arm end-effector link name
    """
    
    def __init__(self, mjcf_path: str, left_end_link: str, right_end_link: str):
        if not MUJOCO_AVAILABLE:
            raise ImportError("MuJoCo is required. Install with: pip install mujoco")
        
        # Load MuJoCo model
        self.mjcf_path = Path(mjcf_path)
        if not self.mjcf_path.exists():
            raise FileNotFoundError(f"MJCF file not found: {mjcf_path}")
        
        print(f"Loading MuJoCo model from: {self.mjcf_path}")
        self.mj_model = mujoco.MjModel.from_xml_path(str(self.mjcf_path))
        self.mj_data = mujoco.MjData(self.mj_model)
        print(f"✓ MuJoCo model loaded: {self.mj_model.nq} DOF")
        
        # Gripper center offset (from link7 to gripper center, in link7 frame)
        self.gripper_offset = np.array([0.14128, 0.0, 0.00015, 1.0])
        
        # Load RoboCore model with RobotModel (not BimanualRobotModel)
        print("\n[Pure RobotModel] Initializing robot model...")
        self.robot = RobotModel(str(self.mjcf_path), base_link='base_link', end_link=None)
        
        # Create left and right arm groups manually
        print("[Pure RobotModel] Creating dual-arm groups...")
        self.robot.add_groups({
            'left_arm': left_end_link,
            'right_arm': right_end_link
        })
        
        # Access sub-models
        self.left_model = self.robot.groups()['left_arm']
        self.right_model = self.robot.groups()['right_arm']
        
        print(f"[Pure RobotModel] Left arm: {self.left_model.num_chain_dof} DOF")
        print(f"[Pure RobotModel] Right arm: {self.right_model.num_chain_dof} DOF")
        
        # Current joint configuration
        self.q_left = np.zeros(self.left_model.num_chain_dof)
        self.q_right = np.zeros(self.right_model.num_chain_dof)
        
        # Target poses
        self.T_left_target = None
        self.T_right_target = None
        self.T_center_target = None
        
        # Initial reference poses
        self.T_left_initial = None
        self.T_right_initial = None
        
        # Coordination mode
        self.mode = 'independent'
        
        # Mocap body IDs
        self.left_marker_id = None
        self.right_marker_id = None
        self.center_marker_id = None
        
        # Initialize
        self._initialize_mocap_ids()
        self._initialize_targets()
        self._initialize_relative_transform()
        
        self.is_running = False
        self.viewer = None
        self.reset_requested = False
    
    def _initialize_mocap_ids(self):
        """Find mocap body IDs by name."""
        for body_id in range(self.mj_model.nbody):
            mocap_id = self.mj_model.body_mocapid[body_id]
            if mocap_id >= 0:
                body_name = mujoco.mj_id2name(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, body_id)
                if body_name == 'left_target':
                    self.left_marker_id = mocap_id
                elif body_name == 'right_target':
                    self.right_marker_id = mocap_id
                elif body_name == 'center_target':
                    self.center_marker_id = mocap_id
        
        print(f"✓ Mocap markers found: left={self.left_marker_id}, right={self.right_marker_id}, center={self.center_marker_id}")
    
    def _initialize_targets(self):
        """Initialize target poses from current FK."""
        # Get initial FK poses using RobotModel's FK method
        result_left = self.left_model.fk(self.q_left, return_end=True)
        result_right = self.right_model.fk(self.q_right, return_end=True)
        
        T_left_link7 = result_left
        T_right_link7 = result_right
        
        # Convert to gripper center poses
        T_offset = np.eye(4)
        T_offset[0:3, 3] = self.gripper_offset[0:3]
        
        self.T_left_target = T_left_link7 @ T_offset
        self.T_right_target = T_right_link7 @ T_offset
        
        self.T_left_initial = self.T_left_target.copy()
        self.T_right_initial = self.T_right_target.copy()
        
        # Center target is midpoint
        self.T_center_target = np.eye(4)
        self.T_center_target[0:3, 3] = 0.5 * (self.T_left_target[0:3, 3] + self.T_right_target[0:3, 3])
        self.T_center_target[0:3, 0:3] = self.T_left_target[0:3, 0:3].copy()
        
        print(f"✓ Initial targets set from zero-config FK")
        print(f"  Left gripper center: {self.T_left_target[:3, 3]}")
        print(f"  Right gripper center: {self.T_right_target[:3, 3]}")
    
    def _initialize_relative_transform(self):
        """Initialize relative grasp transform."""
        T_left_inv = np.linalg.inv(self.T_left_initial)
        self.T_rel_grasp = T_left_inv @ self.T_right_initial
        print(f"✓ Initial relative grasp transform set")
    
    def _update_robot_pose(self):
        """Update MuJoCo robot joint positions."""
        nq_left = self.left_model.num_chain_dof
        nq_right = self.right_model.num_chain_dof
        
        # MuJoCo model has: right arm (0-6), then left arm (7-13)
        self.mj_data.qpos[0:nq_right] = self.q_right
        self.mj_data.qpos[nq_right:nq_right+nq_left] = self.q_left
        
        mujoco.mj_forward(self.mj_model, self.mj_data)
    
    def _solve_ik(self):
        """Solve IK for both arms independently."""
        from robocore.kinematics.ik import inverse_kinematics
        
        # Left arm IK
        res_left = inverse_kinematics(
            self.left_model,
            target_pose=self.T_left_target.tolist(),
            q_initial=self.q_left,
            backend='numpy',
            method='dls',
            max_iters=30,
            pos_tol=5e-4,
            ori_tol=5e-3,
        )
        
        if res_left['success']:
            self.q_left = np.array(res_left['q'])
        
        # Right arm IK
        res_right = inverse_kinematics(
            self.right_model,
            target_pose=self.T_right_target.tolist(),
            q_initial=self.q_right,
            backend='numpy',
            method='dls',
            max_iters=30,
            pos_tol=5e-4,
            ori_tol=5e-3,
        )
        
        if res_right['success']:
            self.q_right = np.array(res_right['q'])
    
    def _update_mocap_from_targets(self):
        """Update mocap body poses from target poses."""
        if self.left_marker_id is not None:
            self.mj_data.mocap_pos[self.left_marker_id] = self.T_left_target[:3, 3]
            self.mj_data.mocap_quat[self.left_marker_id] = self._R_to_quat(self.T_left_target[:3, :3])
        
        if self.right_marker_id is not None:
            self.mj_data.mocap_pos[self.right_marker_id] = self.T_right_target[:3, 3]
            self.mj_data.mocap_quat[self.right_marker_id] = self._R_to_quat(self.T_right_target[:3, :3])
        
        if self.center_marker_id is not None:
            self.mj_data.mocap_pos[self.center_marker_id] = self.T_center_target[:3, 3]
            self.mj_data.mocap_quat[self.center_marker_id] = self._R_to_quat(self.T_center_target[:3, :3])
    
    def _update_targets_from_mocap(self):
        """Update target poses from mocap body poses."""
        if self.left_marker_id is not None:
            pos = self.mj_data.mocap_pos[self.left_marker_id]
            quat = self.mj_data.mocap_quat[self.left_marker_id]
            self.T_left_target = self._pose_from_pos_quat(pos, quat)
        
        if self.right_marker_id is not None:
            pos = self.mj_data.mocap_pos[self.right_marker_id]
            quat = self.mj_data.mocap_quat[self.right_marker_id]
            self.T_right_target = self._pose_from_pos_quat(pos, quat)
        
        if self.center_marker_id is not None:
            pos = self.mj_data.mocap_pos[self.center_marker_id]
            quat = self.mj_data.mocap_quat[self.center_marker_id]
            self.T_center_target = self._pose_from_pos_quat(pos, quat)
    
    @staticmethod
    def _R_to_quat(R):
        """Convert rotation matrix to quaternion (w, x, y, z)."""
        from scipy.spatial.transform import Rotation
        rot = Rotation.from_matrix(R)
        quat = rot.as_quat()  # x, y, z, w
        return np.array([quat[3], quat[0], quat[1], quat[2]])  # w, x, y, z
    
    @staticmethod
    def _pose_from_pos_quat(pos, quat):
        """Build 4x4 pose from position and quaternion."""
        from scipy.spatial.transform import Rotation
        R = Rotation.from_quat([quat[1], quat[2], quat[3], quat[0]])  # x, y, z, w
        T = np.eye(4)
        T[:3, :3] = R.as_matrix()
        T[:3, 3] = pos
        return T
    
    def reset(self):
        """Reset robot and targets to initial state."""
        print("\n🔄 Resetting to initial state...")
        
        self.q_left = np.zeros(self.left_model.num_chain_dof)
        self.q_right = np.zeros(self.right_model.num_chain_dof)
        
        self.T_left_target = self.T_left_initial.copy()
        self.T_right_target = self.T_right_initial.copy()
        
        self.T_center_target[0:3, 3] = 0.5 * (self.T_left_target[0:3, 3] + self.T_right_target[0:3, 3])
        self.T_center_target[0:3, 0:3] = self.T_left_target[0:3, 0:3].copy()
        
        self.reset_requested = False
    
    def run(self, mode='independent'):
        """Run the interactive visualization."""
        self.mode = mode
        
        print("\n" + "=" * 60)
        print("  Interactive Dual-Arm IK - Mode: INDEPENDENT")
        print("=" * 60)
        print("\nControls:")
        print("  - Drag colored spheres to move targets")
        print("  - RED sphere: Left arm target")
        print("  - BLUE sphere: Right arm target")
        print("\n  - Press SPACE to reset")
        print("  - Press ESC to exit\n")
        
        with mujoco.viewer.launch_passive(self.mj_model, self.mj_data) as viewer:
            self.viewer = viewer
            self.is_running = True
            
            while viewer.is_running() and self.is_running:
                step_start = time.time()
                
                # Handle reset request
                if self.reset_requested:
                    self.reset()
                
                # Update targets from mocap
                self._update_targets_from_mocap()
                
                # Solve IK
                self._solve_ik()
                
                # Update robot pose
                self._update_robot_pose()
                
                # Update mocap targets (for visualization)
                self._update_mocap_from_targets()
                
                # Sync to viewer
                viewer.sync()
                
                # Force real-time playback
                time_until_next_step = self.mj_model.opt.timestep - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)
        
        print("✓ Visualization closed")


def main():
    """Demo: Independent dual-arm IK control (Pure RobotModel version)."""
    
    # Model paths
    mjcf_path = get_robocore_path("assets/robot_descriptions/mjcf/Bessica_D_v1_0/Bessica_D_Covered_Interactive.xml")
    
    # End-effector links
    left_end = "left_arm_link7"
    right_end = "right_arm_link7"
    
    # Create interactive IK controller
    controller = InteractiveDualArmIK(mjcf_path, left_end, right_end)
    
    # Run in independent mode
    controller.run(mode='independent')


if __name__ == '__main__':
    main()

