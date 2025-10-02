"""RoboCore: Robot Kinematics and Control Library.

A lightweight robotics library providing:
- URDF/MJCF parsing
- Forward/Inverse Kinematics
- Singularity analysis
- Real and simulated robot bridges
- Pure Python and NumPy-accelerated implementations
"""

__version__ = "0.1.0"

# Core modules
from . import backend
from . import modeling
from . import kinematics
from . import analysis

# Main classes / unified APIs for convenience
from .modeling.robot_model import RobotModel
from .kinematics import (
    forward_kinematics,
    inverse_kinematics,
    jacobian,
    IKSolverNumPy as IKSolver,  # keep alias for backward style usage (numpy solver)
    analytic_jacobian_numpy,
    numeric_jacobian_numpy,
)

__all__ = [
    "__version__",
    "backend",
    "modeling",
    "kinematics",
    "analysis",
    "RobotModel",
    "IKSolver",
    "forward_kinematics",
    "inverse_kinematics",
    "jacobian",
    "numeric_jacobian_numpy",
    "analytic_jacobian_numpy",
]

