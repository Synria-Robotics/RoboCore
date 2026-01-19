#!/usr/bin/env python3
"""Interactive humanoid IK with MuJoCo visualization.

Copyright (c) 2025 Synria Robotics Co., Ltd.

Provides real-time humanoid IK solving with interactive target manipulation
for thumbs and toes.
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


class InteractiveHumanoidIK:
    """Interactive humanoid IK visualization with draggable thumb and toe targets.
    
    :param mjcf_path: Path to MuJoCo MJCF/XML file
    :param left_thumb_end_link: Left thumb end-effector link name
    :param right_thumb_end_link: Right thumb end-effector link name
    :param left_toe_end_link: Left toe end-effector link name
    :param right_toe_end_link: Right toe end-effector link name
    """
    
    def __init__(
        self, 
        mjcf_path: str, 
        left_thumb_end_link: str,
        right_thumb_end_link: str,
        left_toe_end_link: str,
        right_toe_end_link: str
    ):
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
        
        # Load RoboCore robot model
        # NOTE: Do NOT specify base_link='pelvis' here, use default base_link
        # This way FK/IK will use world coordinates (like dual-arm robot), avoiding coordinate conversion issues
        self.robot = RobotModel(str(self.mjcf_path))
        self.left_thumb_end_link = left_thumb_end_link
        self.right_thumb_end_link = right_thumb_end_link
        self.left_toe_end_link = left_toe_end_link
        self.right_toe_end_link = right_toe_end_link
        
        # Current joint configuration
        # Initialize with slightly bent knees to avoid IK flipping for legs
        self.q_full = np.zeros(self.robot.num_dof)
        self._set_initial_leg_config()
        
        # Target poses
        self.T_left_thumb_target = None
        self.T_right_thumb_target = None
        self.T_left_toe_target = None
        self.T_right_toe_target = None
        
        # Initial reference poses
        self.T_left_thumb_initial = None
        self.T_right_thumb_initial = None
        self.T_left_toe_initial = None
        self.T_right_toe_initial = None
        
        # Mocap body IDs (for interactive markers)
        self.left_thumb_marker_id = None
        self.right_thumb_marker_id = None
        self.left_toe_marker_id = None
        self.right_toe_marker_id = None
        
        # Initialize mocap IDs
        self._initialize_mocap_ids()
        
        # Initialize targets from current FK
        self._initialize_targets()
        
        # Initialize mocap markers to match targets
        self._update_markers()
        
        self.is_running = False
        self.viewer = None
        self.reset_requested = False
    
    def _set_initial_leg_config(self):
        """Set initial leg configuration with slightly bent knees.
        
        This prevents IK from finding flipped solutions when dragging feet.
        """
        # Map of joint names to initial values (in radians)
        # Slightly bent knees for stable IK starting point
        leg_config = {
            'left_hip_pitch_joint': -0.1,     # -5.7 deg
            'left_knee_joint': 0.2,           # 11.5 deg bent
            'left_ankle_pitch_joint': -0.1,   # -5.7 deg
            'right_hip_pitch_joint': -0.1,
            'right_knee_joint': 0.2,
            'right_ankle_pitch_joint': -0.1,
        }
        
        # Apply to q_full based on dof_list
        for i, js in enumerate(self.robot.dof_list):
            if js.name in leg_config:
                self.q_full[i] = leg_config[js.name]
        
        # Also update MuJoCo qpos
        nq = min(len(self.q_full), self.mj_model.nq)
        self.mj_data.qpos[0:nq] = self.q_full[0:nq]
        mujoco.mj_forward(self.mj_model, self.mj_data)
        
        print("✓ Initial leg config set with bent knees")
    
    def _initialize_mocap_ids(self):
        """Find mocap body IDs by name."""
        # Iterate through all bodies to find mocap bodies
        for body_id in range(self.mj_model.nbody):
            # Check if this body is a mocap body
            mocap_id = self.mj_model.body_mocapid[body_id]
            if mocap_id >= 0:
                body_name = mujoco.mj_id2name(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, body_id)
                if body_name == 'left_thumb_target':
                    self.left_thumb_marker_id = mocap_id
                elif body_name == 'right_thumb_target':
                    self.right_thumb_marker_id = mocap_id
                elif body_name == 'left_toe_target':
                    self.left_toe_marker_id = mocap_id
                elif body_name == 'right_toe_target':
                    self.right_toe_marker_id = mocap_id
        
        print(f"✓ Mocap markers found: left_thumb={self.left_thumb_marker_id}, "
              f"right_thumb={self.right_thumb_marker_id}, "
              f"left_toe={self.left_toe_marker_id}, "
              f"right_toe={self.right_toe_marker_id}")
    
    # Coordinate conversion functions removed - no longer needed
    # We now use world coordinates directly (like dual-arm robot)
    # This avoids coordinate conversion issues that caused reversed dragging directions
    
    def _initialize_targets(self):
        """Initialize target poses from current FK."""
        # Get initial FK poses (at zero config) - use default base_link (world coordinates)
        # No need to specify base_link, FK will use world coordinates like dual-arm robot
        T_left_thumb = self.robot.fk(self.q_full, end_link=self.left_thumb_end_link, return_end=True)
        T_right_thumb = self.robot.fk(self.q_full, end_link=self.right_thumb_end_link, return_end=True)
        T_left_toe = self.robot.fk(self.q_full, end_link=self.left_toe_end_link, return_end=True)
        T_right_toe = self.robot.fk(self.q_full, end_link=self.right_toe_end_link, return_end=True)
        
        # FK with return_end=True returns 4x4 matrix directly
        if isinstance(T_left_thumb, dict):
            T_left_thumb = T_left_thumb.get('end', T_left_thumb)
        if isinstance(T_right_thumb, dict):
            T_right_thumb = T_right_thumb.get('end', T_right_thumb)
        if isinstance(T_left_toe, dict):
            T_left_toe = T_left_toe.get('end', T_left_toe)
        if isinstance(T_right_toe, dict):
            T_right_toe = T_right_toe.get('end', T_right_toe)
        
        # FK already returns world coordinates, use directly for mocap bodies
        self.T_left_thumb_target = T_left_thumb.copy()
        self.T_right_thumb_target = T_right_thumb.copy()
        self.T_left_toe_target = T_left_toe.copy()
        self.T_right_toe_target = T_right_toe.copy()
        
        # Save initial poses
        self.T_left_thumb_initial = self.T_left_thumb_target.copy()
        self.T_right_thumb_initial = self.T_right_thumb_target.copy()
        self.T_left_toe_initial = self.T_left_toe_target.copy()
        self.T_right_toe_initial = self.T_right_toe_target.copy()
        
        print(f"✓ Initial targets set from zero-config FK")
        print(f"  Left thumb: {self.T_left_thumb_target[:3, 3]}")
        print(f"  Right thumb: {self.T_right_thumb_target[:3, 3]}")
        print(f"  Left toe: {self.T_left_toe_target[:3, 3]}")
        print(f"  Right toe: {self.T_right_toe_target[:3, 3]}")
    
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
    
    def _get_marker_poses(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Get current marker poses from MuJoCo (user dragged positions and orientations).
        
        Returns poses in world coordinates (mocap bodies are in world frame).
        """
        T_left_thumb = np.eye(4)
        T_right_thumb = np.eye(4)
        T_left_toe = np.eye(4)
        T_right_toe = np.eye(4)
        
        if self.left_thumb_marker_id is not None:
            T_left_thumb[0:3, 3] = self.mj_data.mocap_pos[self.left_thumb_marker_id].copy()
            quat = self.mj_data.mocap_quat[self.left_thumb_marker_id].copy()
            T_left_thumb[0:3, 0:3] = self._quat2mat(quat)
        
        if self.right_thumb_marker_id is not None:
            T_right_thumb[0:3, 3] = self.mj_data.mocap_pos[self.right_thumb_marker_id].copy()
            quat = self.mj_data.mocap_quat[self.right_thumb_marker_id].copy()
            T_right_thumb[0:3, 0:3] = self._quat2mat(quat)
        
        if self.left_toe_marker_id is not None:
            T_left_toe[0:3, 3] = self.mj_data.mocap_pos[self.left_toe_marker_id].copy()
            quat = self.mj_data.mocap_quat[self.left_toe_marker_id].copy()
            T_left_toe[0:3, 0:3] = self._quat2mat(quat)
        
        if self.right_toe_marker_id is not None:
            T_right_toe[0:3, 3] = self.mj_data.mocap_pos[self.right_toe_marker_id].copy()
            quat = self.mj_data.mocap_quat[self.right_toe_marker_id].copy()
            T_right_toe[0:3, 0:3] = self._quat2mat(quat)
        
        # Markers are already in world coordinates, return as-is
        return T_left_thumb, T_right_thumb, T_left_toe, T_right_toe
    
    def solve_ik(self):
        """Solve IK for all four targets simultaneously."""
        # Use world coordinates directly (like dual-arm robot)
        # No need to convert to pelvis-relative since FK/IK use world coordinates
        
        # Read current joint angles from MuJoCo as initial guess for IK
        # This ensures we always use the actual current pose, even if physics simulation
        # or other factors have modified the joint angles
        nq = min(self.robot.num_dof, self.mj_model.nq)
        q_current = self.mj_data.qpos[0:nq].copy()
        
        # Use multi-chain IK with unified configuration space
        # For interactive dragging, use fast/loose parameters (like dual-arm demo)
        # Current joint angles as initial guess usually works well for continuous dragging
        res = self.robot.ik(
            targets={
                self.left_thumb_end_link: self.T_left_thumb_target,
                self.right_thumb_end_link: self.T_right_thumb_target,
                self.left_toe_end_link: self.T_left_toe_target,
                self.right_toe_end_link: self.T_right_toe_target,
            },
            end_links=[
                self.left_thumb_end_link, 
                self.right_thumb_end_link,
                self.left_toe_end_link,
                self.right_toe_end_link
            ],
            q_initial=q_current,  # Use current MuJoCo joint angles for interactive continuity
            method='dls',
            max_iters=30,  # Reduced for interactive performance (was 200)
            pos_tol=1e-2,  # Looser tolerance for faster convergence (was 1e-3)
            ori_tol=1e-2,  # Looser tolerance for faster convergence (was 1e-3)
            # No num_initial_guesses - single attempt with current config is usually sufficient
            # No base_link specified - uses default (world coordinates)
        )

        self.q_full = np.array(res['q'])
    
    def _update_robot_pose(self):
        """Update MuJoCo robot joint positions from unified config space."""
        nq = min(len(self.q_full), self.mj_model.nq)
        self.mj_data.qpos[0:nq] = self.q_full[0:nq]
        
        # Forward kinematics in MuJoCo
        mujoco.mj_forward(self.mj_model, self.mj_data)
    
    def _update_markers(self):
        """Update mocap marker poses to match current targets."""
        if self.left_thumb_marker_id is not None:
            self.mj_data.mocap_pos[self.left_thumb_marker_id] = self.T_left_thumb_target[0:3, 3]
            self.mj_data.mocap_quat[self.left_thumb_marker_id] = self._mat2quat(self.T_left_thumb_target[0:3, 0:3])
        
        if self.right_thumb_marker_id is not None:
            self.mj_data.mocap_pos[self.right_thumb_marker_id] = self.T_right_thumb_target[0:3, 3]
            self.mj_data.mocap_quat[self.right_thumb_marker_id] = self._mat2quat(self.T_right_thumb_target[0:3, 0:3])
        
        if self.left_toe_marker_id is not None:
            self.mj_data.mocap_pos[self.left_toe_marker_id] = self.T_left_toe_target[0:3, 3]
            self.mj_data.mocap_quat[self.left_toe_marker_id] = self._mat2quat(self.T_left_toe_target[0:3, 0:3])
        
        if self.right_toe_marker_id is not None:
            self.mj_data.mocap_pos[self.right_toe_marker_id] = self.T_right_toe_target[0:3, 3]
            self.mj_data.mocap_quat[self.right_toe_marker_id] = self._mat2quat(self.T_right_toe_target[0:3, 0:3])
    
    def step(self):
        """Single simulation step."""
        # Read marker positions (updated by user dragging)
        T_left_thumb_dragged, T_right_thumb_dragged, T_left_toe_dragged, T_right_toe_dragged = self._get_marker_poses()
        
        # Check if any marker moved (position or rotation)
        left_thumb_pos_diff = np.linalg.norm(T_left_thumb_dragged[0:3, 3] - self.T_left_thumb_target[0:3, 3])
        left_thumb_rot_diff = np.linalg.norm(T_left_thumb_dragged[0:3, 0:3] - self.T_left_thumb_target[0:3, 0:3])
        left_thumb_moved = left_thumb_pos_diff > 0.001 or left_thumb_rot_diff > 0.01
        
        right_thumb_pos_diff = np.linalg.norm(T_right_thumb_dragged[0:3, 3] - self.T_right_thumb_target[0:3, 3])
        right_thumb_rot_diff = np.linalg.norm(T_right_thumb_dragged[0:3, 0:3] - self.T_right_thumb_target[0:3, 0:3])
        right_thumb_moved = right_thumb_pos_diff > 0.001 or right_thumb_rot_diff > 0.01
        
        left_toe_pos_diff = np.linalg.norm(T_left_toe_dragged[0:3, 3] - self.T_left_toe_target[0:3, 3])
        left_toe_rot_diff = np.linalg.norm(T_left_toe_dragged[0:3, 0:3] - self.T_left_toe_target[0:3, 0:3])
        left_toe_moved = left_toe_pos_diff > 0.001 or left_toe_rot_diff > 0.01
        
        right_toe_pos_diff = np.linalg.norm(T_right_toe_dragged[0:3, 3] - self.T_right_toe_target[0:3, 3])
        right_toe_rot_diff = np.linalg.norm(T_right_toe_dragged[0:3, 0:3] - self.T_right_toe_target[0:3, 0:3])
        right_toe_moved = right_toe_pos_diff > 0.001 or right_toe_rot_diff > 0.01
        
        # Update targets if markers moved (update entire transform matrix)
        if left_thumb_moved:
            self.T_left_thumb_target = T_left_thumb_dragged.copy()
        if right_thumb_moved:
            self.T_right_thumb_target = T_right_thumb_dragged.copy()
        if left_toe_moved:
            self.T_left_toe_target = T_left_toe_dragged.copy()
        if right_toe_moved:
            self.T_right_toe_target = T_right_toe_dragged.copy()
        
        # Solve IK if any target changed
        if left_thumb_moved or right_thumb_moved or left_toe_moved or right_toe_moved:
            self.solve_ik()
        
        # Update robot pose in MuJoCo
        self._update_robot_pose()

    def reset(self):
        """Reset robot and mocap markers to initial state."""
        print("\n🔄 Resetting to initial state...")

        # Reset joint angles to zero
        self.q_full = np.zeros(self.robot.num_dof)

        # Reset target poses to initial values (already in world coordinates)
        self.T_left_thumb_target = self.T_left_thumb_initial.copy()
        self.T_right_thumb_target = self.T_right_thumb_initial.copy()
        self.T_left_toe_target = self.T_left_toe_initial.copy()
        self.T_right_toe_target = self.T_right_toe_initial.copy()

        # Update MuJoCo state
        self._update_robot_pose()
        # Recompute initial targets from FK (in case pelvis moved)
        self._initialize_targets()
        self._update_markers()
        mujoco.mj_forward(self.mj_model, self.mj_data)

        print("✅ Reset complete!")

    def run(self):
        """Run interactive visualization."""
        self.is_running = True
        
        print(f"\n{'='*60}")
        print(f"  Interactive Humanoid IK Control")
        print(f"{'='*60}")
        print(f"\nControls:")
        print(f"  - Drag GREEN spheres to move thumb targets")
        print(f"  - Drag RED spheres to move toe targets")
        print(f"  - Press SPACE to reset")
        print(f"  - Press ESC to exit\n")
        
        import platform
        
        try:
            # Launch viewer
            with mujoco.viewer.launch_passive(self.mj_model, self.mj_data) as viewer:
                self.viewer = viewer
                
                # Enable shadows
                viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 1

                # Initialize mocap markers to current end-effector positions
                self._update_markers()
                mujoco.mj_forward(self.mj_model, self.mj_data)
                
                # Sync viewer AFTER setting mocap positions
                viewer.sync()
                
                # Small delay to ensure mocap positions are stable
                time.sleep(0.1)
                
                # Track last reset time to detect reset events
                last_reset_time = self.mj_data.time
                last_qpos = self.mj_data.qpos.copy()

                while viewer.is_running() and self.is_running:
                    step_start = time.time()
                    
                    # Check for reset: either time went backwards OR qpos was reset to zero
                    current_time = self.mj_data.time
                    current_qpos = self.mj_data.qpos.copy()

                    # Detect if simulation was reset
                    if current_time < last_reset_time or np.allclose(current_qpos[:self.robot.num_dof], 0.0, atol=1e-4):
                        # Only reset if we actually had non-zero joint angles before
                        if not np.allclose(last_qpos[:self.robot.num_dof], 0.0, atol=1e-4):
                            print("\n🔄 Simulation reset detected!")
                            self.reset()

                    # Also support keyboard-triggered reset
                    if getattr(self, 'reset_requested', False):
                        print("\n🔄 Reset requested via keyboard (SPACE)")
                        self.reset()
                        self.reset_requested = False

                    last_reset_time = current_time
                    last_qpos = current_qpos.copy()

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
                print(f"  mjpython examples/bridge/demo_mujoco_humanoid.py")
                print(f"\nIf mjpython is not available, install with:")
                print(f"  pip install mujoco")
                raise
            else:
                raise
        
        print("\n✓ Visualization closed")


__all__ = ["InteractiveHumanoidIK"]
