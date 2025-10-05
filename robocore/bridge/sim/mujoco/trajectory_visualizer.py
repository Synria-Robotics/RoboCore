#!/usr/bin/env python3
"""RoboCore Module

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

import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, List
import time

try:
    import mujoco
    import mujoco.viewer
    MUJOCO_AVAILABLE = True
except ImportError:
    MUJOCO_AVAILABLE = False
    print("⚠️  MuJoCo not available. Install with: pip install mujoco")


class TrajectoryVisualizer:
    """
    MuJoCo-based trajectory visualizer for robot motion.
    
    Parameters
    ----------
    urdf_path : str or Path
        Path to robot URDF file
    mjcf_path : str or Path, optional
        Path to MuJoCo MJCF/XML file. If None, will try to convert URDF.
    """
    
    def __init__(self, urdf_path: Optional[str] = None, mjcf_path: Optional[str] = None):
        if not MUJOCO_AVAILABLE:
            raise ImportError("MuJoCo is required. Install with: pip install mujoco")
        
        self.urdf_path = urdf_path
        self.mjcf_path = mjcf_path
        
        # MuJoCo model and data
        self.model = None
        self.data = None
        
        # Trajectory data
        self.trajectory = None
        self.trajectory_time = None
        self.current_index = 0
        self.is_playing = False
        self.playback_speed = 1.0
        self.loop = True
        
        # Timing
        self.last_update_time = 0
        self.sim_time = 0
        
        # Load model
        if mjcf_path:
            self._load_mjcf(mjcf_path)
        elif urdf_path:
            self._load_from_urdf(urdf_path)
        else:
            raise ValueError("Either urdf_path or mjcf_path must be provided")
    
    def _load_mjcf(self, mjcf_path: str):
        """Load MuJoCo model from MJCF file."""
        mjcf_path = Path(mjcf_path)
        if not mjcf_path.exists():
            raise FileNotFoundError(f"MJCF file not found: {mjcf_path}")
        
        print(f"Loading MuJoCo model from: {mjcf_path}")
        self.model = mujoco.MjModel.from_xml_path(str(mjcf_path))
        self.data = mujoco.MjData(self.model)
        print(f"✓ Model loaded: {self.model.nq} DOF")
    
    def _load_from_urdf(self, urdf_path: str):
        """
        Load model from URDF (requires MJCF conversion).
        
        Note: MuJoCo can load URDF directly, but MJCF is preferred.
        For best results, convert URDF to MJCF manually.
        """
        urdf_path = Path(urdf_path)
        if not urdf_path.exists():
            raise FileNotFoundError(f"URDF file not found: {urdf_path}")
        
        # Try to load URDF directly (MuJoCo 2.3.0+)
        print(f"Loading URDF: {urdf_path}")
        try:
            self.model = mujoco.MjModel.from_xml_path(str(urdf_path))
            self.data = mujoco.MjData(self.model)
            print(f"✓ Model loaded: {self.model.nq} DOF")
        except Exception as e:
            print(f"✗ Failed to load URDF directly: {e}")
            print("\nℹ️  Consider converting URDF to MJCF format:")
            print("   1. Use: mujoco.MjModel.from_xml_path() with proper assets")
            print("   2. Or manually convert URDF to MJCF")
            raise
    
    def load_trajectory(
        self,
        q_trajectory: np.ndarray,
        time: Optional[np.ndarray] = None,
        qd_trajectory: Optional[np.ndarray] = None,
        qdd_trajectory: Optional[np.ndarray] = None
    ):
        """
        Load trajectory data for playback.
        
        Parameters
        ----------
        q_trajectory : np.ndarray
            Joint positions (N, nq)
        time : np.ndarray, optional
            Time stamps (N,). If None, assumes uniform time steps.
        qd_trajectory : np.ndarray, optional
            Joint velocities (N, nq)
        qdd_trajectory : np.ndarray, optional
            Joint accelerations (N, nq)
        """
        if q_trajectory.ndim != 2:
            raise ValueError(f"q_trajectory must be 2D, got shape {q_trajectory.shape}")
        
        if q_trajectory.shape[1] != self.model.nq:
            raise ValueError(
                f"Trajectory DOF ({q_trajectory.shape[1]}) != model DOF ({self.model.nq})"
            )
        
        self.trajectory = q_trajectory
        
        if time is None:
            # Assume 0.01s time step
            time = np.arange(len(q_trajectory)) * 0.01
        
        self.trajectory_time = time
        self.qd_trajectory = qd_trajectory
        self.qdd_trajectory = qdd_trajectory
        
        self.current_index = 0
        self.sim_time = 0
        
        print(f"\n✓ Trajectory loaded:")
        print(f"  Points: {len(q_trajectory)}")
        print(f"  Duration: {time[-1]:.2f}s")
        print(f"  DOF: {q_trajectory.shape[1]}")
    
    def set_joint_positions(self, q: np.ndarray):
        """Set current joint positions."""
        if len(q) != self.model.nq:
            raise ValueError(f"Joint position size mismatch: {len(q)} != {self.model.nq}")
        
        self.data.qpos[:len(q)] = q
        mujoco.mj_forward(self.model, self.data)
    
    def play(self):
        """Start trajectory playback."""
        self.is_playing = True
        self.last_update_time = time.time()
    
    def pause(self):
        """Pause trajectory playback."""
        self.is_playing = False
    
    def reset(self):
        """Reset to trajectory start."""
        self.current_index = 0
        self.sim_time = 0
        if self.trajectory is not None:
            self.set_joint_positions(self.trajectory[0])
    
    def set_speed(self, speed: float):
        """Set playback speed (1.0 = real-time)."""
        self.playback_speed = max(0.1, min(10.0, speed))
        print(f"Playback speed: {self.playback_speed:.1f}x")
    
    def step_forward(self):
        """Step one frame forward."""
        if self.trajectory is None:
            return
        
        self.current_index = min(self.current_index + 1, len(self.trajectory) - 1)
        self.sim_time = self.trajectory_time[self.current_index]
        self.set_joint_positions(self.trajectory[self.current_index])
    
    def step_backward(self):
        """Step one frame backward."""
        if self.trajectory is None:
            return
        
        self.current_index = max(self.current_index - 1, 0)
        self.sim_time = self.trajectory_time[self.current_index]
        self.set_joint_positions(self.trajectory[self.current_index])
    
    def update(self):
        """Update simulation state (call in render loop)."""
        if not self.is_playing or self.trajectory is None:
            return
        
        current_time = time.time()
        dt = (current_time - self.last_update_time) * self.playback_speed
        self.last_update_time = current_time
        
        self.sim_time += dt
        
        # Find corresponding trajectory index
        if self.sim_time >= self.trajectory_time[-1]:
            if self.loop:
                # Loop back to start
                self.sim_time = 0
                self.current_index = 0
            else:
                # Stop at end
                self.is_playing = False
                self.current_index = len(self.trajectory) - 1
                return
        
        # Binary search for current index
        idx = np.searchsorted(self.trajectory_time, self.sim_time)
        idx = min(idx, len(self.trajectory) - 1)
        self.current_index = idx
        
        # Update joint positions
        self.set_joint_positions(self.trajectory[self.current_index])
    
    def visualize(self, title: str = "Trajectory Visualization"):
        """
        Open interactive MuJoCo viewer with trajectory playback.
        
        Parameters
        ----------
        title : str
            Window title
        
        Controls
        --------
        Space : Play/Pause
        R : Reset to start
        [ : Decrease speed
        ] : Increase speed
        , : Step backward
        . : Step forward
        L : Toggle loop
        ESC : Close viewer
        """
        if self.trajectory is None:
            raise ValueError("No trajectory loaded. Call load_trajectory() first.")
        
        print(f"\n{'='*70}")
        print(f"MuJoCo Trajectory Visualization")
        print(f"{'='*70}")
        print(f"\nControls:")
        print(f"  Space  : Play/Pause")
        print(f"  R      : Reset to start")
        print(f"  [  ]   : Decrease/Increase speed")
        print(f"  ,  .   : Step backward/forward")
        print(f"  L      : Toggle loop")
        print(f"  ESC    : Close viewer")
        print(f"\nTrajectory Info:")
        print(f"  Duration: {self.trajectory_time[-1]:.2f}s")
        print(f"  Points: {len(self.trajectory)}")
        print(f"  Speed: {self.playback_speed:.1f}x")
        print(f"  Loop: {'ON' if self.loop else 'OFF'}")
        print(f"\n{'='*70}\n")
        
        # Reset to start
        self.reset()
        
        # Try passive viewer first, fallback to active on macOS
        import platform
        
        try:
            # Launch passive viewer
            with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
                # Configure viewer
                viewer.cam.distance = 2.0
                viewer.cam.azimuth = 45
                viewer.cam.elevation = -20
                
                while viewer.is_running():
                    # Update trajectory
                    self.update()
                    
                    # Sync visualization
                    viewer.sync()
                    
                    # Small sleep to control frame rate
                    time.sleep(0.01)
        
        except Exception as e:
            if 'mjpython' in str(e) and platform.system() == 'Darwin':
                print(f"\n[WARNING] macOS requires mjpython for interactive viewer")
                print(f"\nOptions:")
                print(f"  1. mjpython examples/demo_mujoco_trajectory.py [args]")
                print(f"  2. Using fallback viewer (auto-play mode)\n")
                
                # Fallback: use simple animation loop
                print(f"Using simplified visualization mode...")
                self._visualize_simple()
            else:
                raise
    
    def _visualize_simple(self):
        """Simple visualization with mouse controls (for macOS compatibility)."""
        import glfw
        
        # Initialize GLFW
        if not glfw.init():
            raise RuntimeError("Failed to initialize GLFW")
        
        # Create window
        window = glfw.create_window(1200, 900, "MuJoCo Trajectory Viewer", None, None)
        if not window:
            glfw.terminate()
            raise RuntimeError("Failed to create GLFW window")
        
        glfw.make_context_current(window)
        glfw.swap_interval(1)
        
        # Create camera and scene
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.distance = 2.5
        cam.azimuth = 90
        cam.elevation = -15
        cam.lookat = np.array([0.0, 0.0, 0.8])
        
        opt = mujoco.MjvOption()
        scene = mujoco.MjvScene(self.model, maxgeom=10000)
        context = mujoco.MjrContext(self.model, mujoco.mjtFontScale.mjFONTSCALE_150)
        
        # Mouse interaction state
        button_left = False
        button_right = False
        button_middle = False
        last_x = 0
        last_y = 0
        
        def mouse_button_callback(window, button, action, mods):
            nonlocal button_left, button_right, button_middle
            button_left = glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS
            button_right = glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS
            button_middle = glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_MIDDLE) == glfw.PRESS
        
        def cursor_pos_callback(window, xpos, ypos):
            nonlocal last_x, last_y
            
            dx = xpos - last_x
            dy = ypos - last_y
            last_x = xpos
            last_y = ypos
            
            width, height = glfw.get_window_size(window)
            
            # Left drag: rotate
            if button_left:
                cam.azimuth += dx * 0.5
                cam.elevation += dy * 0.5
                cam.elevation = np.clip(cam.elevation, -89, 89)
            
            # Right drag: zoom
            elif button_right:
                cam.distance += dy * 0.01
                cam.distance = max(0.1, min(10.0, cam.distance))
            
            # Middle drag: translate
            elif button_middle:
                # Get camera right and up vectors
                forward = np.array([
                    np.cos(np.radians(cam.elevation)) * np.cos(np.radians(cam.azimuth)),
                    np.cos(np.radians(cam.elevation)) * np.sin(np.radians(cam.azimuth)),
                    np.sin(np.radians(cam.elevation))
                ])
                right = np.cross(forward, [0, 0, 1])
                right = right / np.linalg.norm(right)
                up = np.cross(right, forward)
                
                # Translate lookat point
                cam.lookat -= right * dx * 0.001 * cam.distance
                cam.lookat += up * dy * 0.001 * cam.distance
        
        def scroll_callback(window, xoffset, yoffset):
            cam.distance -= yoffset * 0.1
            cam.distance = max(0.1, min(10.0, cam.distance))
        
        def key_callback(window, key, scancode, action, mods):
            if action != glfw.PRESS:
                return
            
            # Space: play/pause
            if key == glfw.KEY_SPACE:
                if self.is_playing:
                    self.pause()
                    print("|| PAUSED")
                else:
                    self.play()
                    print("> PLAYING")
            
            # R: reset
            elif key == glfw.KEY_R:
                self.reset()
                print("RESET to start")
            
            # [ ]: speed control
            elif key == glfw.KEY_LEFT_BRACKET:
                self.set_speed(self.playback_speed * 0.5)
            elif key == glfw.KEY_RIGHT_BRACKET:
                self.set_speed(self.playback_speed * 2.0)
            
            # , .: step
            elif key == glfw.KEY_COMMA:
                self.step_backward()
                print(f"< BACKWARD: {self.current_index}/{len(self.trajectory)}")
            elif key == glfw.KEY_PERIOD:
                self.step_forward()
                print(f"> FORWARD: {self.current_index}/{len(self.trajectory)}")
            
            # L: loop
            elif key == glfw.KEY_L:
                self.loop = not self.loop
                print(f"LOOP: {'ON' if self.loop else 'OFF'}")
            
            # H: help
            elif key == glfw.KEY_H:
                print("\nControls:")
                print("  Space     : Play/Pause")
                print("  R         : Reset")
                print("  [ ]       : Decrease/Increase speed")
                print("  , .       : Step backward/forward")
                print("  L         : Toggle loop")
                print("  H         : Show help")
                print("  ESC       : Exit")
                print("\nMouse:")
                print("  Left drag   : Rotate")
                print("  Right drag  : Zoom")
                print("  Middle drag : Pan")
                print("  Scroll      : Zoom")
        
        # Set callbacks
        glfw.set_mouse_button_callback(window, mouse_button_callback)
        glfw.set_cursor_pos_callback(window, cursor_pos_callback)
        glfw.set_scroll_callback(window, scroll_callback)
        glfw.set_key_callback(window, key_callback)
        
        # Initialize mouse position
        last_x, last_y = glfw.get_cursor_pos(window)
        
        # Auto-play
        self.is_playing = True
        self.last_update_time = time.time()
        
        print(f"\nPlaying trajectory...")
        print(f"Press H for help, ESC to exit\n")
        
        # Main loop
        frame_count = 0
        last_info_time = time.time()
        
        while not glfw.window_should_close(window):
            # Update trajectory
            self.update()
            
            # Render
            viewport_width, viewport_height = glfw.get_framebuffer_size(window)
            viewport = mujoco.MjrRect(0, 0, viewport_width, viewport_height)
            
            mujoco.mjv_updateScene(
                self.model, self.data, opt, None, cam,
                mujoco.mjtCatBit.mjCAT_ALL, scene
            )
            mujoco.mjr_render(viewport, scene, context)
            
            # Overlay text info
            status = "> PLAY" if self.is_playing else "|| PAUSE"
            info_text = (
                f"{status} | "
                f"Time: {self.sim_time:.2f}/{self.trajectory_time[-1]:.2f}s | "
                f"Frame: {self.current_index+1}/{len(self.trajectory)} | "
                f"Speed: {self.playback_speed:.1f}x"
            )
            
            mujoco.mjr_overlay(
                mujoco.mjtFont.mjFONT_NORMAL,
                mujoco.mjtGridPos.mjGRID_TOPLEFT,
                viewport,
                info_text,
                "",
                context
            )
            
            # Show help hint
            mujoco.mjr_overlay(
                mujoco.mjtFont.mjFONT_NORMAL,
                mujoco.mjtGridPos.mjGRID_BOTTOMLEFT,
                viewport,
                "Press H for help | ESC to exit",
                "",
                context
            )
            
            # Swap buffers
            glfw.swap_buffers(window)
            glfw.poll_events()
            
            # Print periodic info
            frame_count += 1
            current_time = time.time()
            if current_time - last_info_time > 2.0:
                fps = frame_count / (current_time - last_info_time)
                print(f"[INFO] {status} | Progress: {self.sim_time/self.trajectory_time[-1]*100:.1f}% | FPS: {fps:.1f}")
                frame_count = 0
                last_info_time = current_time
        
        glfw.terminate()
        print(f"\nVisualization ended")
    
    def get_info(self) -> Dict[str, Any]:
        """Get current playback information."""
        if self.trajectory is None:
            return {"status": "No trajectory loaded"}
        
        return {
            "status": "Playing" if self.is_playing else "Paused",
            "time": self.sim_time,
            "duration": self.trajectory_time[-1],
            "progress": f"{self.sim_time / self.trajectory_time[-1] * 100:.1f}%",
            "frame": f"{self.current_index + 1}/{len(self.trajectory)}",
            "speed": f"{self.playback_speed:.1f}x",
            "loop": self.loop
        }
    
    def export_video(
        self,
        output_path: str,
        fps: int = 30,
        width: int = 1280,
        height: int = 720,
        camera_id: Optional[int] = None
    ):
        """
        Export trajectory to video file.
        
        Parameters
        ----------
        output_path : str
            Output video file path (e.g., 'trajectory.mp4')
        fps : int
            Frames per second
        width : int
            Video width
        height : int
            Video height
        camera_id : int, optional
            Camera ID to use. If None, uses free camera.
        
        Requires
        --------
        imageio : pip install imageio imageio-ffmpeg
        """
        try:
            import imageio
        except ImportError:
            raise ImportError(
                "imageio required for video export. "
                "Install with: pip install imageio imageio-ffmpeg"
            )
        
        if self.trajectory is None:
            raise ValueError("No trajectory loaded")
        
        print(f"\nExporting video to: {output_path}")
        print(f"  Resolution: {width}x{height}")
        print(f"  FPS: {fps}")
        
        # Create renderer
        renderer = mujoco.Renderer(self.model, width=width, height=height)
        
        # Calculate frames needed
        total_time = self.trajectory_time[-1]
        num_frames = int(total_time * fps)
        
        frames = []
        
        for i in range(num_frames):
            # Get trajectory position at this time
            t = i / fps
            idx = np.searchsorted(self.trajectory_time, t)
            idx = min(idx, len(self.trajectory) - 1)
            
            # Set joint positions
            self.set_joint_positions(self.trajectory[idx])
            
            # Render frame
            renderer.update_scene(self.data, camera=camera_id)
            frame = renderer.render()
            frames.append(frame)
            
            if (i + 1) % 30 == 0:
                print(f"  Progress: {(i+1)/num_frames*100:.1f}%")
        
        # Write video
        imageio.mimsave(output_path, frames, fps=fps)
        print(f"✓ Video exported successfully!")
        print(f"  Frames: {num_frames}")
        print(f"  Duration: {total_time:.2f}s")


def visualize_trajectory(
    trajectory_data: Dict[str, Any],
    mjcf_path: str,
    title: str = "Trajectory Visualization",
    auto_play: bool = True,
    playback_speed: float = 1.0,
    loop: bool = True
):
    """
    Convenience function to visualize trajectory.
    
    Parameters
    ----------
    trajectory_data : dict
        Dictionary with keys: 'q', 'time', optionally 'qd', 'qdd'
    mjcf_path : str
        Path to MuJoCo MJCF model file
    title : str
        Window title
    auto_play : bool
        Start playing automatically
    playback_speed : float
        Playback speed multiplier
    loop : bool
        Loop trajectory playback
    
    Example
    -------
    >>> trajectory_data = {
    ...     'q': q_trajectory,
    ...     'time': time_array,
    ...     'qd': qd_trajectory  # optional
    ... }
    >>> visualize_trajectory(trajectory_data, 'robot.xml')
    """
    # Create visualizer
    viz = TrajectoryVisualizer(mjcf_path=mjcf_path)
    
    # Load trajectory
    viz.load_trajectory(
        q_trajectory=trajectory_data['q'],
        time=trajectory_data.get('time'),
        qd_trajectory=trajectory_data.get('qd'),
        qdd_trajectory=trajectory_data.get('qdd')
    )
    
    # Configure playback
    viz.set_speed(playback_speed)
    viz.loop = loop
    
    if auto_play:
        viz.play()
    
    # Visualize
    viz.visualize(title=title)


__all__ = ['TrajectoryVisualizer', 'visualize_trajectory']
