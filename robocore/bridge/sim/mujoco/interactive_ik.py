#!/usr/bin/env python3
"""Generic interactive IK with MuJoCo visualization.

Copyright (c) 2025 Synria Robotics Co., Ltd.

Provides a unified interactive IK interface that automatically discovers
interaction targets from MJCF files based on naming conventions.

Naming Convention:
    - Mocap bodies: `ik_target_<name>` (e.g., ik_target_left_gripper)
    - Offset sites: `ik_offset_<name>` (optional, for gripper center offsets)
    - Center target: `ik_target_center` (optional, for relative/coordinated control)
"""

import numpy as np
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any
from dataclasses import dataclass, field
import time

try:
    import mujoco
    import mujoco.viewer
    MUJOCO_AVAILABLE = True
except ImportError:
    MUJOCO_AVAILABLE = False
    print("⚠️  MuJoCo not available. Install with: pip install mujoco")

from robocore.modeling.robot_model import RobotModel


# Naming convention constants
IK_TARGET_PREFIX = "ik_target_"
IK_OFFSET_PREFIX = "ik_offset_"
IK_CENTER_TARGET = "ik_target_center"


@dataclass
class InteractionTarget:
    """Configuration for a single interaction target.
    
    :param name: Unique identifier for this target (extracted from mocap body name)
    :param end_link: Robot end-effector link name for IK
    :param mocap_id: MuJoCo mocap body ID
    :param color: Display color for the marker [r, g, b, a]
    :param offset: Optional offset from end_link to target point (homogeneous coords)
    :param T_target: Current target pose (4x4 matrix)
    :param T_initial: Initial target pose for reset
    """
    name: str
    end_link: str
    mocap_id: int
    color: np.ndarray = field(default_factory=lambda: np.array([1, 0, 0, 0.5]))
    offset: Optional[np.ndarray] = None
    T_target: np.ndarray = field(default_factory=lambda: np.eye(4))
    T_initial: np.ndarray = field(default_factory=lambda: np.eye(4))


class InteractiveIK:
    """Generic interactive IK visualization with auto-discovered targets.
    
    Automatically discovers interaction targets from MJCF based on naming convention:
    - Mocap bodies named `ik_target_<name>` are treated as draggable targets
    - Sites named `ik_offset_<name>` provide optional offsets (e.g., gripper center)
    - Mocap body `ik_target_center` enables coordinated control modes
    
    :param mjcf_path: Path to MuJoCo MJCF/XML file
    :param target_end_links: Dict mapping target name to end_link name.
                             If None, will try to auto-match by name.
    :param initial_config: Optional dict of joint_name -> initial_value
    """
    
    def __init__(
        self, 
        mjcf_path: str, 
        target_end_links: Optional[Dict[str, str]] = None,
        initial_config: Optional[Dict[str, float]] = None
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
        self.robot = RobotModel(str(self.mjcf_path))
        
        # Current joint configuration
        self.q_full = np.zeros(self.robot.num_dof)
        
        # Apply initial configuration if provided
        if initial_config:
            self._apply_initial_config(initial_config)
        
        # Auto-discover interaction targets
        self.targets: Dict[str, InteractionTarget] = {}
        self.center_marker_id: Optional[int] = None
        self.T_center_target: Optional[np.ndarray] = None
        self.T_center_initial: Optional[np.ndarray] = None
        
        self._discover_targets(target_end_links or {})
        
        # Initialize targets from current FK
        self._initialize_targets()
        
        # Coordination mode
        self.mode = 'independent'  # 'independent', 'relative', 'mirror'
        
        # For relative mode: store relative transforms between targets
        self.T_rel_transforms: Dict[str, np.ndarray] = {}
        self._initialize_relative_transforms()
        
        self.is_running = False
        self.viewer = None
        self.reset_requested = False
    
    def _apply_initial_config(self, config: Dict[str, float]):
        """Apply initial joint configuration.
        
        :param config: Dict mapping joint name to initial value (radians)
        """
        for i, js in enumerate(self.robot.dof_list):
            if js.name in config:
                self.q_full[i] = config[js.name]
        
        # Update MuJoCo qpos
        nq = min(len(self.q_full), self.mj_model.nq)
        self.mj_data.qpos[0:nq] = self.q_full[0:nq]
        mujoco.mj_forward(self.mj_model, self.mj_data)
        print(f"✓ Initial joint config applied ({len(config)} joints)")
    
    def _discover_targets(self, target_end_links: Dict[str, str]):
        """Discover interaction targets from MJCF.
        
        :param target_end_links: User-provided mapping of target name to end_link
        """
        # First, collect all offsets
        offsets: Dict[str, np.ndarray] = {}
        for site_id in range(self.mj_model.nsite):
            site_name = mujoco.mj_id2name(self.mj_model, mujoco.mjtObj.mjOBJ_SITE, site_id)
            if site_name and site_name.startswith(IK_OFFSET_PREFIX):
                name = site_name[len(IK_OFFSET_PREFIX):]
                site_pos = self.mj_model.site_pos[site_id].copy()
                offsets[name] = np.array([site_pos[0], site_pos[1], site_pos[2], 1.0])
                print(f"  Found offset site: {site_name} -> {site_pos}")
        
        # Find mocap bodies
        for body_id in range(self.mj_model.nbody):
            mocap_id = self.mj_model.body_mocapid[body_id]
            if mocap_id < 0:
                continue
            
            body_name = mujoco.mj_id2name(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, body_id)
            if not body_name:
                continue
            
            # Check for center target
            if body_name == IK_CENTER_TARGET:
                self.center_marker_id = mocap_id
                print(f"  Found center target: {body_name} (mocap_id={mocap_id})")
                continue
            
            # Check for ik_target_ prefix
            if not body_name.startswith(IK_TARGET_PREFIX):
                continue
            
            name = body_name[len(IK_TARGET_PREFIX):]
            
            # Get end_link - from user mapping or try to auto-match
            end_link = target_end_links.get(name)
            if not end_link:
                # Try to find a link with similar name
                end_link = self._auto_match_end_link(name)
            
            if not end_link:
                print(f"  ⚠️  No end_link mapping for target '{name}', skipping")
                continue
            
            # Get optional offset
            offset = offsets.get(name)
            
            # Create target
            target = InteractionTarget(
                name=name,
                end_link=end_link,
                mocap_id=mocap_id,
                offset=offset,
            )
            self.targets[name] = target
            
            offset_str = f", offset={offset[:3]}" if offset is not None else ""
            print(f"  Found target: {body_name} -> end_link='{end_link}'{offset_str}")
        
        print(f"✓ Discovered {len(self.targets)} interaction targets")
        if self.center_marker_id is not None:
            print(f"✓ Center target available for coordinated control")
    
    def _auto_match_end_link(self, target_name: str) -> Optional[str]:
        """Try to auto-match target name to an end_link in the robot model.
        
        :param target_name: Target name (e.g., 'left_gripper', 'right_toe')
        :return: Matching end_link name or None
        """
        # Get all link names from robot model
        link_names = [link.name for link in self.robot.links]
        
        # Try exact match first
        if target_name in link_names:
            return target_name
        
        # Try common patterns
        patterns = [
            target_name,
            f"{target_name}_link",
            target_name.replace('_gripper', '_link7'),
            target_name.replace('_hand', '_link7'),
        ]
        
        for pattern in patterns:
            if pattern in link_names:
                return pattern
        
        return None
    
    def _initialize_targets(self):
        """Initialize target poses from current FK."""
        for name, target in self.targets.items():
            # Get FK pose
            T_link = self.robot.fk(self.q_full, end_link=target.end_link, return_end=True)
            if isinstance(T_link, dict):
                T_link = T_link.get('end', T_link)
            
            # Apply offset if present
            if target.offset is not None:
                T_offset = np.eye(4)
                T_offset[0:3, 3] = target.offset[0:3]
                T_target = T_link @ T_offset
            else:
                T_target = T_link.copy()
            
            target.T_target = T_target
            target.T_initial = T_target.copy()
            print(f"  {name}: pos={T_target[:3, 3]}")
        
        # Initialize center target if present
        if self.center_marker_id is not None and len(self.targets) >= 2:
            positions = [t.T_target[:3, 3] for t in self.targets.values()]
            center_pos = np.mean(positions, axis=0)
            self.T_center_target = np.eye(4)
            self.T_center_target[:3, 3] = center_pos
            # Use first target's orientation
            first_target = next(iter(self.targets.values()))
            self.T_center_target[:3, :3] = first_target.T_target[:3, :3].copy()
            self.T_center_initial = self.T_center_target.copy()
        
        print(f"✓ Initial targets set from FK")
    
    def _initialize_relative_transforms(self):
        """Initialize relative transforms between targets (for relative mode)."""
        if len(self.targets) < 2:
            return
        
        target_list = list(self.targets.values())
        base_target = target_list[0]
        T_base = self._target_to_link_pose(base_target)
        
        for target in target_list[1:]:
            T_other = self._target_to_link_pose(target)
            T_rel = np.linalg.inv(T_base) @ T_other
            self.T_rel_transforms[target.name] = T_rel
        
        print(f"✓ Relative transforms initialized")
    
    def _target_to_link_pose(self, target: InteractionTarget) -> np.ndarray:
        """Convert target pose to link pose (remove offset).
        
        :param target: Interaction target
        :return: Link pose (4x4 matrix)
        """
        if target.offset is None:
            return target.T_target.copy()
        
        T_offset = np.eye(4)
        T_offset[0:3, 3] = target.offset[0:3]
        return target.T_target @ np.linalg.inv(T_offset)
    
    def _link_to_target_pose(self, target: InteractionTarget, T_link: np.ndarray) -> np.ndarray:
        """Convert link pose to target pose (apply offset).
        
        :param target: Interaction target
        :param T_link: Link pose (4x4 matrix)
        :return: Target pose (4x4 matrix)
        """
        if target.offset is None:
            return T_link.copy()
        
        T_offset = np.eye(4)
        T_offset[0:3, 3] = target.offset[0:3]
        return T_link @ T_offset
    
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
    
    def _update_markers(self):
        """Update mocap marker poses to match current targets."""
        for target in self.targets.values():
            self.mj_data.mocap_pos[target.mocap_id] = target.T_target[0:3, 3]
            self.mj_data.mocap_quat[target.mocap_id] = self._mat2quat(target.T_target[0:3, 0:3])
        
        if self.center_marker_id is not None and self.T_center_target is not None:
            self.mj_data.mocap_pos[self.center_marker_id] = self.T_center_target[0:3, 3]
            self.mj_data.mocap_quat[self.center_marker_id] = self._mat2quat(self.T_center_target[0:3, 0:3])
    
    def _get_marker_poses(self) -> Dict[str, np.ndarray]:
        """Get current marker poses from MuJoCo (user dragged positions).
        
        :return: Dict mapping target name to pose (4x4 matrix)
        """
        poses = {}
        for name, target in self.targets.items():
            T = np.eye(4)
            T[0:3, 3] = self.mj_data.mocap_pos[target.mocap_id].copy()
            quat = self.mj_data.mocap_quat[target.mocap_id].copy()
            T[0:3, 0:3] = self._quat2mat(quat)
            poses[name] = T
        return poses
    
    def _get_center_pose(self) -> Optional[np.ndarray]:
        """Get current center marker pose from MuJoCo.
        
        :return: Center pose (4x4 matrix) or None
        """
        if self.center_marker_id is None:
            return None
        
        T = np.eye(4)
        T[0:3, 3] = self.mj_data.mocap_pos[self.center_marker_id].copy()
        quat = self.mj_data.mocap_quat[self.center_marker_id].copy()
        T[0:3, 0:3] = self._quat2mat(quat)
        return T
    
    def solve_ik(self):
        """Solve IK for all targets simultaneously."""
        # Build targets dict for multi-chain IK
        ik_targets = {}
        end_links = []
        
        for target in self.targets.values():
            T_link = self._target_to_link_pose(target)
            ik_targets[target.end_link] = T_link
            end_links.append(target.end_link)
        
        # Solve multi-chain IK
        res = self.robot.ik(
            targets=ik_targets,
            end_links=end_links,
            q_initial=self.q_full,
            method='dls',
            max_iters=15,
            pos_tol=1e-2,
            ori_tol=1e-2,
        )
        
        self.q_full = np.array(res['q'])
    
    def _update_robot_pose(self):
        """Update MuJoCo robot joint positions."""
        nq = min(len(self.q_full), self.mj_model.nq)
        self.mj_data.qpos[0:nq] = self.q_full[0:nq]
        mujoco.mj_forward(self.mj_model, self.mj_data)
    
    def step(self):
        """Single simulation step."""
        # Read marker positions (updated by user dragging)
        marker_poses = self._get_marker_poses()
        center_pose = self._get_center_pose()
        
        # Check which markers moved
        moved_targets = {}
        for name, T_dragged in marker_poses.items():
            target = self.targets[name]
            pos_diff = np.linalg.norm(T_dragged[0:3, 3] - target.T_target[0:3, 3])
            rot_diff = np.linalg.norm(T_dragged[0:3, 0:3] - target.T_target[0:3, 0:3])
            if pos_diff > 0.001 or rot_diff > 0.01:
                moved_targets[name] = T_dragged
        
        # Check if center moved
        center_moved = False
        if center_pose is not None and self.T_center_target is not None:
            pos_diff = np.linalg.norm(center_pose[0:3, 3] - self.T_center_target[0:3, 3])
            rot_diff = np.linalg.norm(center_pose[0:3, 0:3] - self.T_center_target[0:3, 0:3])
            center_moved = pos_diff > 0.001 or rot_diff > 0.01
        
        # Handle based on mode
        if self.mode == 'independent':
            self._step_independent(moved_targets)
        elif self.mode == 'relative':
            self._step_relative(moved_targets, center_pose, center_moved)
        elif self.mode == 'mirror':
            self._step_mirror(moved_targets)
        
        # Update robot pose in MuJoCo
        self._update_robot_pose()
    
    def _step_independent(self, moved_targets: Dict[str, np.ndarray]):
        """Handle independent mode step.
        
        :param moved_targets: Dict of moved target name to new pose
        """
        if not moved_targets:
            return
        
        # Update moved targets
        for name, T_dragged in moved_targets.items():
            self.targets[name].T_target = T_dragged
        
        # Solve IK
        self.solve_ik()
    
    def _step_relative(
        self, 
        moved_targets: Dict[str, np.ndarray],
        center_pose: Optional[np.ndarray],
        center_moved: bool
    ):
        """Handle relative mode step.
        
        :param moved_targets: Dict of moved target name to new pose
        :param center_pose: Current center marker pose
        :param center_moved: Whether center marker was moved
        """
        if center_moved and center_pose is not None:
            # Move all targets together maintaining relative poses
            delta_pos = center_pose[0:3, 3] - self.T_center_target[0:3, 3]
            R_old = self.T_center_target[0:3, 0:3]
            R_new = center_pose[0:3, 0:3]
            R_delta = R_new @ R_old.T
            
            center_old = self.T_center_target[0:3, 3]
            
            for target in self.targets.values():
                # Rotate around center
                rel_pos = target.T_target[0:3, 3] - center_old
                new_pos = center_old + R_delta @ rel_pos + delta_pos
                target.T_target[0:3, 3] = new_pos
                target.T_target[0:3, 0:3] = R_delta @ target.T_target[0:3, 0:3]
                
                # Update mocap marker
                self.mj_data.mocap_pos[target.mocap_id] = target.T_target[0:3, 3]
                self.mj_data.mocap_quat[target.mocap_id] = self._mat2quat(target.T_target[0:3, 0:3])
            
            self.T_center_target = center_pose.copy()
            self.solve_ik()
        
        elif moved_targets:
            # Individual targets moved - update relative transforms
            for name, T_dragged in moved_targets.items():
                self.targets[name].T_target = T_dragged
            
            # Update relative transforms
            self._initialize_relative_transforms()
            
            # Update center
            if self.T_center_target is not None:
                positions = [t.T_target[:3, 3] for t in self.targets.values()]
                self.T_center_target[:3, 3] = np.mean(positions, axis=0)
                if self.center_marker_id is not None:
                    self.mj_data.mocap_pos[self.center_marker_id] = self.T_center_target[:3, 3]
            
            self.solve_ik()
    
    def _step_mirror(self, moved_targets: Dict[str, np.ndarray]):
        """Handle mirror mode step (mirrors across XZ plane).
        
        :param moved_targets: Dict of moved target name to new pose
        """
        if not moved_targets:
            return
        
        # Get target names sorted for pairing (e.g., left_gripper, right_gripper)
        target_names = list(self.targets.keys())
        
        # Find pairs (left/right)
        pairs = self._find_mirror_pairs()
        
        for moved_name, T_dragged in moved_targets.items():
            self.targets[moved_name].T_target = T_dragged
            
            # Find mirror pair
            mirror_name = pairs.get(moved_name)
            if mirror_name and mirror_name not in moved_targets:
                # Mirror the pose
                T_mirrored = self._mirror_pose(T_dragged, moved_name)
                self.targets[mirror_name].T_target = T_mirrored
                
                # Update mocap marker
                target = self.targets[mirror_name]
                self.mj_data.mocap_pos[target.mocap_id] = T_mirrored[0:3, 3]
                self.mj_data.mocap_quat[target.mocap_id] = self._mat2quat(T_mirrored[0:3, 0:3])
        
        self.solve_ik()
    
    def _find_mirror_pairs(self) -> Dict[str, str]:
        """Find left/right mirror pairs among targets.
        
        :return: Dict mapping target name to its mirror pair name
        """
        pairs = {}
        names = list(self.targets.keys())
        
        for name in names:
            if 'left' in name.lower():
                mirror = name.lower().replace('left', 'right')
                for other in names:
                    if other.lower() == mirror:
                        pairs[name] = other
                        pairs[other] = name
                        break
        
        return pairs
    
    def _mirror_pose(self, T: np.ndarray, source_name: str) -> np.ndarray:
        """Mirror a pose across XZ plane (Y=0).
        
        :param T: Source pose (4x4 matrix)
        :param source_name: Name of source target (to determine direction)
        :return: Mirrored pose (4x4 matrix)
        """
        T_mirrored = T.copy()
        
        # Mirror position (y -> -y)
        T_mirrored[1, 3] = -T[1, 3]
        
        # Mirror rotation
        M = np.diag([1, -1, 1])
        R = T[0:3, 0:3]
        T_mirrored[0:3, 0:3] = M @ R @ M.T
        
        return T_mirrored
    
    def reset(self):
        """Reset robot and mocap markers to initial state."""
        print("\n🔄 Resetting to initial state...")
        
        # Reset joint angles to zero
        self.q_full = np.zeros(self.robot.num_dof)
        
        # Reset targets
        for target in self.targets.values():
            target.T_target = target.T_initial.copy()
        
        # Reset center
        if self.T_center_initial is not None:
            self.T_center_target = self.T_center_initial.copy()
        
        # Reset relative transforms
        self._initialize_relative_transforms()
        
        # Update MuJoCo state
        self._update_robot_pose()
        self._update_markers()
        mujoco.mj_forward(self.mj_model, self.mj_data)
        
        print("✅ Reset complete!")
    
    def run(self, mode: str = 'independent'):
        """Run interactive visualization.
        
        :param mode: Control mode - 'independent', 'relative', or 'mirror'
        """
        self.mode = mode
        self.is_running = True
        
        print(f"\n{'='*60}")
        print(f"  Interactive IK Control - Mode: {mode.upper()}")
        print(f"{'='*60}")
        print(f"\nDiscovered targets:")
        for name, target in self.targets.items():
            print(f"  - {name} -> {target.end_link}")
        
        print(f"\nControls:")
        print(f"  - Drag spheres to move targets")
        if mode == 'relative' and self.center_marker_id is not None:
            print(f"  - Drag CENTER sphere to move all targets together")
        print(f"  - Press SPACE to reset")
        print(f"  - Press ESC to exit\n")
        
        import platform
        
        try:
            with mujoco.viewer.launch_passive(self.mj_model, self.mj_data) as viewer:
                self.viewer = viewer
                
                viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 1
                
                # Initialize mocap markers
                self._update_markers()
                mujoco.mj_forward(self.mj_model, self.mj_data)
                viewer.sync()
                
                time.sleep(0.1)
                
                last_reset_time = self.mj_data.time
                last_qpos = self.mj_data.qpos.copy()
                
                while viewer.is_running() and self.is_running:
                    step_start = time.time()
                    
                    # Check for reset
                    current_time = self.mj_data.time
                    current_qpos = self.mj_data.qpos.copy()
                    
                    nq = min(self.robot.num_dof, len(current_qpos))
                    if current_time < last_reset_time or np.allclose(current_qpos[:nq], 0.0, atol=1e-4):
                        if not np.allclose(last_qpos[:nq], 0.0, atol=1e-4):
                            print("\n🔄 Simulation reset detected!")
                            self.reset()
                    
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
                print(f"\nPlease run with: mjpython <script>.py")
                raise
            else:
                raise
        
        print("\n✓ Visualization closed")

__all__ = ["InteractiveIK", "InteractionTarget"]
