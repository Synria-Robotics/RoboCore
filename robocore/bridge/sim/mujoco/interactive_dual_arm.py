#!/usr/bin/env python3
"""Interactive dual-arm IK with MuJoCo visualization.

Copyright (c) 2025 Synria Robotics Co., Ltd.

Provides real-time dual-arm IK solving with interactive target manipulation.
"""

import numpy as np
from pathlib import Path
from typing import Optional, Dict, Tuple
import time

try:
    import mujoco
    import mujoco.viewer
    MUJOCO_AVAILABLE = True
except ImportError:
    MUJOCO_AVAILABLE = False
    print("⚠️  MuJoCo not available. Install with: pip install mujoco")

from robocore.modeling.robot_model import RobotModel
from robocore.kinematics.task import absolute_task, relative_task
from robocore.utils.path import get_robocore_path


class InteractiveDualArmIK:
    """Interactive dual-arm IK visualization with draggable targets.
    
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
        # Based on MJCF: gripper fingers at pos="0.14128 ±0.0505 0.00015"
        # Center is at 0.14128 m in X direction from link7
        self.gripper_offset = np.array([0.14128, 0.0, 0.00015, 1.0])  # Homogeneous coordinates
        
        # Load RoboCore models
        self.robot = RobotModel(str(self.mjcf_path))
        self.robot.add_groups({
            'left_arm': left_end_link,
            'right_arm': right_end_link
        })
        
        self.left_model = self.robot.groups()['left_arm']
        self.right_model = self.robot.groups()['right_arm']
        
        print(f"✓ Left arm: {self.left_model.num_dof()} DOF, end: {left_end_link}")
        print(f"✓ Right arm: {self.right_model.num_dof()} DOF, end: {right_end_link}")
        
        # Current joint configuration
        self.q_left = np.zeros(self.left_model.num_dof())
        self.q_right = np.zeros(self.right_model.num_dof())
        
        # Target poses
        self.T_left_target = None
        self.T_right_target = None
        self.T_center_target = None  # For cooperative mode
        
        # Coordination mode
        self.mode = 'independent'  # 'independent', 'cooperative', 'mirror'
        
        # Mocap body IDs (for interactive markers)
        self.left_marker_id = None
        self.right_marker_id = None
        self.center_marker_id = None
        
        # Initialize mocap IDs
        self._initialize_mocap_ids()
        
        # Initialize targets from current FK
        self._initialize_targets()
        
        # Relative grasp transform (for cooperative mode)
        # T_rel = T_left^-1 @ T_right
        self.T_rel_grasp = None
        self._initialize_relative_transform()
        
        self.is_running = False
        self.viewer = None
    
    def _initialize_mocap_ids(self):
        """Find mocap body IDs by name."""
        # Iterate through all bodies to find mocap bodies
        for body_id in range(self.mj_model.nbody):
            # Check if this body is a mocap body
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
        # Get initial FK poses (at zero config, should be zero-config FK)
        result_left = self.left_model.fk(self.q_left)
        result_right = self.right_model.fk(self.q_right)
        
        # Get link7 poses (FK returns dict with 'end' key)
        T_left_link7 = result_left['end']
        T_right_link7 = result_right['end']
        
        # Convert to gripper center poses (T_gripper = T_link7 @ T_offset)
        T_offset = np.eye(4)
        T_offset[0:3, 3] = self.gripper_offset[0:3]
        
        self.T_left_target = T_left_link7 @ T_offset
        self.T_right_target = T_right_link7 @ T_offset
        
        # Center target is midpoint of left and right
        self.T_center_target = np.eye(4)
        self.T_center_target[0:3, 3] = 0.5 * (self.T_left_target[0:3, 3] + self.T_right_target[0:3, 3])
        # Use left orientation for center
        self.T_center_target[0:3, 0:3] = self.T_left_target[0:3, 0:3].copy()
        
        print(f"✓ Initial targets set from zero-config FK")
        print(f"  Left gripper center: {self.T_left_target[:3, 3]}")
        print(f"  Right gripper center: {self.T_right_target[:3, 3]}")
    
    def _initialize_relative_transform(self):
        """Initialize relative grasp transform from current poses."""
        # Convert gripper targets to link7 targets
        T_left_link7 = self._gripper_to_link7(self.T_left_target)
        T_right_link7 = self._gripper_to_link7(self.T_right_target)
        
        # Compute relative transform
        self.T_rel_grasp = np.linalg.inv(T_left_link7) @ T_right_link7
        print(f"✓ Initial relative grasp transform set")
    
    def _update_markers(self):
        """Update mocap marker poses to match current targets."""
        if self.left_marker_id is not None:
            self.mj_data.mocap_pos[self.left_marker_id] = self.T_left_target[0:3, 3]
            self.mj_data.mocap_quat[self.left_marker_id] = self._mat2quat(self.T_left_target[0:3, 0:3])
        
        if self.right_marker_id is not None:
            self.mj_data.mocap_pos[self.right_marker_id] = self.T_right_target[0:3, 3]
            self.mj_data.mocap_quat[self.right_marker_id] = self._mat2quat(self.T_right_target[0:3, 0:3])
        
        if self.center_marker_id is not None:
            self.mj_data.mocap_pos[self.center_marker_id] = self.T_center_target[0:3, 3]
            self.mj_data.mocap_quat[self.center_marker_id] = self._mat2quat(self.T_center_target[0:3, 0:3])
    
    def _mat2quat(self, R: np.ndarray) -> np.ndarray:
        """Convert rotation matrix to quaternion [w, x, y, z]."""
        trace = np.trace(R)
        
        if trace > 0:
            s = 0.5 / np.sqrt(trace + 1.0)
            w = 0.25 / s
            x = (R[2, 1] - R[1, 2]) * s
            y = (R[0, 2] - R[2, 0]) * s
            z = (R[1, 0] - R[0, 1]) * s
        else:
            if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
                s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
                w = (R[2, 1] - R[1, 2]) / s
                x = 0.25 * s
                y = (R[0, 1] + R[1, 0]) / s
                z = (R[0, 2] + R[2, 0]) / s
            elif R[1, 1] > R[2, 2]:
                s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
                w = (R[0, 2] - R[2, 0]) / s
                x = (R[0, 1] + R[1, 0]) / s
                y = 0.25 * s
                z = (R[1, 2] + R[2, 1]) / s
            else:
                s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
                w = (R[1, 0] - R[0, 1]) / s
                x = (R[0, 2] + R[2, 0]) / s
                y = (R[1, 2] + R[2, 1]) / s
                z = 0.25 * s
        
        return np.array([w, x, y, z])
    
    def _quat2mat(self, quat: np.ndarray) -> np.ndarray:
        """Convert quaternion [w, x, y, z] to rotation matrix."""
        w, x, y, z = quat
        return np.array([
            [1 - 2*(y*y + z*z),     2*(x*y - w*z),     2*(x*z + w*y)],
            [    2*(x*y + w*z), 1 - 2*(x*x + z*z),     2*(y*z - w*x)],
            [    2*(x*z - w*y),     2*(y*z + w*x), 1 - 2*(x*x + y*y)]
        ])
    
    def _get_marker_poses(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get current marker poses from MuJoCo (user dragged positions and orientations).
        
        Now supports full 6-DOF control by reading both position and orientation from mocap.
        """
        T_left = np.eye(4)
        T_right = np.eye(4)
        T_center = np.eye(4)
        
        if self.left_marker_id is not None:
            # Read position
            T_left[0:3, 3] = self.mj_data.mocap_pos[self.left_marker_id].copy()
            # Read orientation from mocap
            quat = self.mj_data.mocap_quat[self.left_marker_id].copy()
            T_left[0:3, 0:3] = self._quat2mat(quat)
        
        if self.right_marker_id is not None:
            T_right[0:3, 3] = self.mj_data.mocap_pos[self.right_marker_id].copy()
            # Read orientation from mocap
            quat = self.mj_data.mocap_quat[self.right_marker_id].copy()
            T_right[0:3, 0:3] = self._quat2mat(quat)
        
        if self.center_marker_id is not None:
            T_center[0:3, 3] = self.mj_data.mocap_pos[self.center_marker_id].copy()
            # Read orientation from mocap
            quat = self.mj_data.mocap_quat[self.center_marker_id].copy()
            T_center[0:3, 0:3] = self._quat2mat(quat)
        
        return T_left, T_right, T_center
    
    def _gripper_to_link7(self, T_gripper: np.ndarray) -> np.ndarray:
        """Convert gripper center pose to link7 pose.
        
        T_link7 = T_gripper @ inv(T_offset)
        """
        T_offset = np.eye(4)
        T_offset[0:3, 3] = self.gripper_offset[0:3]
        return T_gripper @ np.linalg.inv(T_offset)
    
    def solve_ik_independent(self):
        """Solve IK for independent dual-arm control (Demo 1)."""
        # Convert gripper center targets to link7 targets
        T_left_link7 = self._gripper_to_link7(self.T_left_target)
        T_right_link7 = self._gripper_to_link7(self.T_right_target)
        
        # Full 6-DOF constraint (position + orientation)
        tasks = [
            absolute_task('left_arm', T_left_link7, weight=1.0, row_mask=[1,1,1,1,1,1]),
            absolute_task('right_arm', T_right_link7, weight=1.0, row_mask=[1,1,1,1,1,1])
        ]
        
        result = self.robot.ik_tasks(
            tasks,
            {'left_arm': self.q_left, 'right_arm': self.q_right},
            mode='weighted',
            max_iters=15,  # 减少迭代次数
            tol=1e-2,  # 增大容差到1cm，减少微小振荡
            verbose=False
        )
        
        if result['success']:
            self.q_left = np.array(result['q_by_group']['left_arm'])
            self.q_right = np.array(result['q_by_group']['right_arm'])
    
    def solve_ik_cooperative(self):
        """Solve IK for cooperative control with relative constraint (Demo 2)."""
        # Convert gripper center targets to link7 targets
        T_left_link7 = self._gripper_to_link7(self.T_left_target)
        T_right_link7 = self._gripper_to_link7(self.T_right_target)
        
        # Targets already updated in step(), just solve with relative constraint
        # Full 6-DOF constraint for absolute tasks
        tasks = [
            absolute_task('left_arm', T_left_link7, weight=1.0, row_mask=[1,1,1,1,1,1]),
            absolute_task('right_arm', T_right_link7, weight=1.0, row_mask=[1,1,1,1,1,1]),
            relative_task('left_arm', 'right_arm', self.T_rel_grasp, 
                         weight=2.0, row_mask=[1, 1, 1, 0, 0, 0])  # Only constrain relative position
        ]
        
        result = self.robot.ik_tasks(
            tasks,
            {'left_arm': self.q_left, 'right_arm': self.q_right},
            mode='weighted',
            max_iters=15,  # 减少迭代次数
            tol=1e-2,  # 增大容差
            verbose=False
        )
        
        if result['success']:
            self.q_left = np.array(result['q_by_group']['left_arm'])
            self.q_right = np.array(result['q_by_group']['right_arm'])
    
    def solve_ik_mirror(self):
        """Solve IK for mirror symmetric control (Demo 3).
        
        Mirrors left arm to right arm across YZ plane (X-axis reflection).
        Left and right arms should be symmetric at initialization.
        """
        # Mirror transformation matrix for reflection across YZ plane
        M_mirror = np.array([
            [-1,  0,  0,  0],
            [ 0,  1,  0,  0],
            [ 0,  0,  1,  0],
            [ 0,  0,  0,  1]
        ])
        
        # Mirror left target to get right target
        # This reflects position and orientation across YZ plane
        self.T_right_target = M_mirror @ self.T_left_target @ M_mirror.T
        
        # Update right mocap marker to show mirrored position
        if self.right_marker_id is not None:
            self.mj_data.mocap_pos[self.right_marker_id] = self.T_right_target[0:3, 3]
            # In mirror mode, use the mirrored orientation
            R_right_mirror = self.T_right_target[0:3, 0:3]
            self.mj_data.mocap_quat[self.right_marker_id] = self._mat2quat(R_right_mirror)
        
        # Convert gripper center targets to link7 targets
        T_left_link7 = self._gripper_to_link7(self.T_left_target)
        T_right_link7 = self._gripper_to_link7(self.T_right_target)
        
        # Full 6-DOF constraint (position + orientation)
        tasks = [
            absolute_task('left_arm', T_left_link7, weight=1.0, row_mask=[1,1,1,1,1,1]),
            absolute_task('right_arm', T_right_link7, weight=1.0, row_mask=[1,1,1,1,1,1])
        ]
        
        result = self.robot.ik_tasks(
            tasks,
            {'left_arm': self.q_left, 'right_arm': self.q_right},
            mode='weighted',
            max_iters=15,  # 减少迭代次数
            tol=1e-2,  # 增大容差
            verbose=False
        )
        
        if result['success']:
            self.q_left = np.array(result['q_by_group']['left_arm'])
            self.q_right = np.array(result['q_by_group']['right_arm'])
    
    def _update_robot_pose(self):
        """Update MuJoCo robot joint positions."""
        # Map RoboCore joint values to MuJoCo qpos
        # MuJoCo model has: right arm (0-6), then left arm (7-13)
        nq_left = self.left_model.num_dof()
        nq_right = self.right_model.num_dof()
        
        # Right arm first, left arm second
        self.mj_data.qpos[0:nq_right] = self.q_right
        self.mj_data.qpos[nq_right:nq_right+nq_left] = self.q_left
        
        # Forward kinematics in MuJoCo
        mujoco.mj_forward(self.mj_model, self.mj_data)
    
    def step(self):
        """Single simulation step."""
        # Read marker positions (updated by user dragging)
        T_left_dragged, T_right_dragged, T_center_dragged = self._get_marker_poses()
        
        # Update targets based on mode
        if self.mode == 'independent':
            # Only update if markers actually moved (user dragged them)
            # This prevents chasing the initial mocap position if it's far from the gripper
            left_moved = np.linalg.norm(T_left_dragged[0:3, 3] - self.T_left_target[0:3, 3]) > 0.001
            right_moved = np.linalg.norm(T_right_dragged[0:3, 3] - self.T_right_target[0:3, 3]) > 0.001
            
            if left_moved or right_moved:
                if left_moved:
                    self.T_left_target = T_left_dragged
                if right_moved:
                    self.T_right_target = T_right_dragged
                self.solve_ik_independent()
        
        elif self.mode == 'cooperative':
            # Check if center was dragged (cooperative move)
            center_moved = np.linalg.norm(T_center_dragged[0:3, 3] - self.T_center_target[0:3, 3]) > 0.001
            
            # Check if individual targets were dragged
            left_moved = np.linalg.norm(T_left_dragged[0:3, 3] - self.T_left_target[0:3, 3]) > 0.001
            right_moved = np.linalg.norm(T_right_dragged[0:3, 3] - self.T_right_target[0:3, 3]) > 0.001
            
            if center_moved:
                # Green sphere dragged: move both arms together maintaining relative pose
                # Support both translation and rotation
                delta_pos = T_center_dragged[0:3, 3] - self.T_center_target[0:3, 3]
                
                # Rotation delta (from center target to dragged)
                R_old_center = self.T_center_target[0:3, 0:3]
                R_new_center = T_center_dragged[0:3, 0:3]
                R_delta = R_new_center @ R_old_center.T
                
                # Update left and right targets
                center_pos_old = self.T_center_target[0:3, 3]
                
                # Rotate left relative to center
                self.T_left_target[0:3, 3] = center_pos_old + R_delta @ (self.T_left_target[0:3, 3] - center_pos_old) + delta_pos
                self.T_left_target[0:3, 0:3] = R_delta @ self.T_left_target[0:3, 0:3]
                
                # Rotate right relative to center  
                self.T_right_target[0:3, 3] = center_pos_old + R_delta @ (self.T_right_target[0:3, 3] - center_pos_old) + delta_pos
                self.T_right_target[0:3, 0:3] = R_delta @ self.T_right_target[0:3, 0:3]
                
                self.T_center_target = T_center_dragged
                
                # Update red/blue mocap markers to follow green marker
                if self.left_marker_id is not None:
                    self.mj_data.mocap_pos[self.left_marker_id] = self.T_left_target[0:3, 3]
                    self.mj_data.mocap_quat[self.left_marker_id] = self._mat2quat(self.T_left_target[0:3, 0:3])
                if self.right_marker_id is not None:
                    self.mj_data.mocap_pos[self.right_marker_id] = self.T_right_target[0:3, 3]
                    self.mj_data.mocap_quat[self.right_marker_id] = self._mat2quat(self.T_right_target[0:3, 0:3])
                
                self.solve_ik_cooperative()
                
            elif left_moved or right_moved:
                # Red/blue spheres dragged: update grasp, move green to midpoint
                self.T_left_target = T_left_dragged
                self.T_right_target = T_right_dragged
                
                # Update relative transform (in link7 frame)
                T_left_link7 = self._gripper_to_link7(self.T_left_target)
                T_right_link7 = self._gripper_to_link7(self.T_right_target)
                self.T_rel_grasp = np.linalg.inv(T_left_link7) @ T_right_link7
                
                # Update center to midpoint (position and average orientation)
                self.T_center_target[0:3, 3] = 0.5 * (self.T_left_target[0:3, 3] + self.T_right_target[0:3, 3])
                # Use left orientation for center (or could average quaternions)
                self.T_center_target[0:3, 0:3] = self.T_left_target[0:3, 0:3].copy()
                
                # Update green mocap marker to follow red/blue midpoint
                if self.center_marker_id is not None:
                    self.mj_data.mocap_pos[self.center_marker_id] = self.T_center_target[0:3, 3]
                    self.mj_data.mocap_quat[self.center_marker_id] = self._mat2quat(self.T_center_target[0:3, 0:3])
                
                self.solve_ik_independent()
        
        elif self.mode == 'mirror':
            self.T_left_target = T_left_dragged
            self.solve_ik_mirror()
        
        # Update robot pose in MuJoCo
        self._update_robot_pose()
        
        # NOTE: Do NOT update markers here! User is dragging them.
        # Only update them at initialization or when we programmatically move them (cooperative mode)
    
    def run(self, mode: str = 'independent'):
        """Run interactive visualization.
        
        :param mode: Control mode - 'independent', 'cooperative', or 'mirror'
        """
        self.mode = mode
        self.is_running = True
        
        print(f"\n{'='*60}")
        print(f"  Interactive Dual-Arm IK - Mode: {mode.upper()}")
        print(f"{'='*60}")
        print(f"\nControls:")
        print(f"  - Drag colored spheres to move targets")
        
        if mode == 'independent':
            print(f"  - RED sphere: Left arm target")
            print(f"  - BLUE sphere: Right arm target")
        elif mode == 'cooperative':
            print(f"  - RED sphere: Left arm target")
            print(f"  - BLUE sphere: Right arm target")
            print(f"  - GREEN sphere: Center (cooperative motion)")
            print(f"  - Drag RED/BLUE to set grasp")
            print(f"  - Drag GREEN to move cooperatively")
        elif mode == 'mirror':
            print(f"  - RED sphere: Left arm target")
            print(f"  - Right arm mirrors left automatically")
        
        print(f"\nPress ESC to exit\n")
        
        # Try passive viewer first, fallback on macOS mjpython error
        import platform
        
        try:
            # Launch viewer
            with mujoco.viewer.launch_passive(self.mj_model, self.mj_data) as viewer:
                self.viewer = viewer
                
                # CRITICAL: Initialize mocap markers to current end-effector positions
                # This prevents the initial huge displacement that causes shaking
                self._update_markers()
                mujoco.mj_forward(self.mj_model, self.mj_data)
                
                # Sync viewer AFTER setting mocap positions
                viewer.sync()
                
                # Small delay to ensure mocap positions are stable
                time.sleep(0.1)
                
                while viewer.is_running() and self.is_running:
                    step_start = time.time()
                    
                    # Update IK
                    self.step()
                    
                    # Sync viewer
                    viewer.sync()
                    
                    # Control update rate (100 Hz)
                    time_until_next_step = self.mj_model.opt.timestep - (time.time() - step_start)
                    if time_until_next_step > 0:
                        time.sleep(time_until_next_step)
        
        except RuntimeError as e:
            if 'mjpython' in str(e) and platform.system() == 'Darwin':
                print(f"\n[ERROR] macOS requires mjpython for interactive viewer")
                print(f"\nPlease run with:")
                print(f"  mjpython examples/bridge/demo_mujoco_{mode}.py")
                print(f"\nIf mjpython is not available, install with:")
                print(f"  pip install mujoco")
                raise
            else:
                raise
        
        print("\n✓ Visualization closed")
