"""Independent bimanual Jacobian assembler (NumPy).

Assembles per-arm Jacobians (6xn) into a combined block-diagonal Jacobian
for dual-arm tasks.
"""
from __future__ import annotations
from typing import Sequence
import numpy as np

from robocore.modeling.robot_model import RobotModel
from robocore.kinematics.jacobian import jacobian as single_jacobian


class BiIndependentJacobianSolverNumpy:
    """Assemble a block-diagonal Jacobian for two independent arms."""

    def __init__(self, left_model: RobotModel, right_model: RobotModel):
        self.left = left_model
        self.right = right_model

    def compute(self, q_left: Sequence[float], q_right: Sequence[float], *, backend: str = 'numpy') -> np.ndarray:
        J_L = single_jacobian(self.left, q_left, backend=backend)
        J_R = single_jacobian(self.right, q_right, backend=backend)

        if hasattr(J_L, 'detach'):
            J_L = J_L.detach().cpu().numpy()
        else:
            J_L = np.array(J_L)
        if hasattr(J_R, 'detach'):
            J_R = J_R.detach().cpu().numpy()
        else:
            J_R = np.array(J_R)

        nL = J_L.shape[1]
        nR = J_R.shape[1]
        J = np.zeros((12, nL + nR))
        J[0:6, 0:nL] = J_L
        J[6:12, nL:] = J_R
        return J
