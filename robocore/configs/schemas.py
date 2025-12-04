"""Configuration schemas for RoboCore using dataclasses.

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

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class RobotConfig:
    """Robot model configuration."""
    urdf_path: str = "robocore/assets/robot/urdf/Alicia-D_v5_4/alicia_duo_with_gripper.urdf"
    end_link: str = "tool0"
    base_link: Optional[str] = None
    

@dataclass
class SolverConfig:
    """Solver-specific configuration."""
    # IK Solver parameters
    max_iterations: int = 100
    position_tolerance: float = 1e-4
    orientation_tolerance: float = 1e-4
    damping: float = 0.01  # For DLS solver
    
    # Jacobian parameters
    step_size: float = 1e-6  # For numerical Jacobian
    

@dataclass
class IKConfig:
    """Inverse Kinematics configuration.
    
    Valid values:
    - method: "dls", "jacobian_transpose", "pseudoinverse"
    - backend: "numpy", "torch"
    """
    method: str = "dls"  
    backend: str = "numpy"
    solver: SolverConfig = field(default_factory=SolverConfig)
    

@dataclass
class ComputeConfig:
    """Computation backend configuration.
    
    Valid values:
    - backend: "numpy", "torch"
    - device: "cpu", "cuda", "cuda:0", "cuda:1", "cuda:2"  # MPS removed
    - dtype: "float32", "float64"
    """
    backend: str = "numpy"
    device: str = "cpu"
    dtype: str = "float32"
    batch_size: Optional[int] = None
    use_jit: bool = False  # For PyTorch JIT compilation


@dataclass
class KinematicsConfig:
    """Kinematics computation configuration.
    
    Valid values:
    - fk_backend: "numpy", "torch"
    - jacobian_method: "analytic", "numeric", "autograd"
    - jacobian_backend: "numpy", "torch"
    """
    # FK configuration
    fk_backend: str = "numpy"
    return_end_only: bool = True
    
    # IK configuration
    ik: IKConfig = field(default_factory=IKConfig)
    
    # Jacobian configuration
    jacobian_method: str = "analytic"
    jacobian_backend: str = "numpy"
    
    # Compute configuration
    compute: ComputeConfig = field(default_factory=ComputeConfig)


@dataclass
class RoboCoreConfig:
    """Top-level RoboCore configuration.
    
    Valid values:
    - log_level: "DEBUG", "INFO", "WARNING", "ERROR"
    """
    robot: RobotConfig = field(default_factory=RobotConfig)
    kinematics: KinematicsConfig = field(default_factory=KinematicsConfig)
    
    # Logging and debugging
    verbose: bool = False
    log_level: str = "INFO"
    
    # Random seed for reproducibility
    seed: Optional[int] = None
