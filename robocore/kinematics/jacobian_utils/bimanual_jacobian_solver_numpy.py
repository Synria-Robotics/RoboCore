"""Independent bimanual Jacobian assembler (NumPy).

Assembles per-arm Jacobians (6xn) into a combined block-diagonal Jacobian
for dual-arm tasks.
"""
from __future__ import annotations
from typing import Sequence, Dict
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


class BiRelativeJacobianSolverNumpy:
    """Relative bimanual Jacobian (NumPy).

    Assemble Jacobian for relative constraints between right and left arm.
    Uses numerical differentiation for correctness (adjoint transform is complex).
    """

    def __init__(self, left_model: RobotModel, right_model: RobotModel):
        self.left = left_model
        self.right = right_model

    def compute(self, q_left: Sequence[float], q_right: Sequence[float], *, backend: str = 'numpy', constraint_type: str = 'pose') -> np.ndarray:
        """
        :param q_left: Left joint configuration
        :param q_right: Right joint configuration
        :param backend: Backend to use
        :param constraint_type: 'pose'|'position'|'orientation'
        :return: Relative constraint Jacobian
        """
        # Use numerical relative Jacobian from utils (correct adjoint handling)
        from robocore.kinematics.utils import relative_jacobian
        
        J_rel_full = relative_jacobian(self.left, self.right, q_left, q_right, backend='numpy')
        
        # Apply constraint type filtering
        if constraint_type == 'pose':
            return J_rel_full
        elif constraint_type == 'position':
            return J_rel_full[:3, :]
        elif constraint_type == 'orientation':
            return J_rel_full[3:, :]
        else:
            raise ValueError("Unknown constraint_type")


class BiMultiLinkJacobianSolverNumpy:
    """Multi-link Jacobian solver for arbitrary number of groups (NumPy)."""

    def __init__(self, groups: Dict[str, RobotModel]):
        self.groups = groups

    def block_jacobian(self, q_by_group: Dict[str, Sequence[float]], *, backend: str = 'auto') -> np.ndarray:
        """
        :param q_by_group: Mapping name -> joint vector
        :param backend: 'auto'|'numpy'|'torch'
        :return: Block-diagonal Jacobian for all groups stacked as 6*k rows
        """
        if not self.groups:
            raise ValueError("No groups defined.")
        # Order by insertion
        names = list(self.groups.keys())
        J_blocks = []
        cols_total = 0
        for name in names:
            model = self.groups[name]
            q = q_by_group[name]
            J = single_jacobian(model, q, backend=backend)
            J = J.detach().cpu().numpy() if hasattr(J, 'detach') else np.array(J)
            J_blocks.append(J)
            cols_total += J.shape[1]
        rows_total = 6 * len(J_blocks)
        J_whole = np.zeros((rows_total, cols_total))
        col_offset = 0
        for i, J in enumerate(J_blocks):
            r0 = 6 * i
            r1 = r0 + 6
            c1 = col_offset + J.shape[1]
            J_whole[r0:r1, col_offset:c1] = J
            col_offset = c1
        return J_whole

    def relative_jacobian_between(self, group_a: str, group_b: str,
                                  q_a: Sequence[float], q_b: Sequence[float], *,
                                  backend: str = 'auto') -> np.ndarray:
        """
        :param group_a: First group name
        :param group_b: Second group name
        :param q_a: Joint vector of group_a
        :param q_b: Joint vector of group_b
        :param backend: Backend for computation
        :return: 6 x (n_a + nR) relative Jacobian (pose)
        """
        if not self.groups:
            raise ValueError("No groups defined.")
        a = self.groups[group_a]
        b = self.groups[group_b]
        from robocore.kinematics.utils import relative_jacobian
        J_rel = relative_jacobian(a, b, q_a, q_b, backend=backend)
        return J_rel.detach().cpu().numpy() if hasattr(J_rel, 'detach') else np.array(J_rel)
