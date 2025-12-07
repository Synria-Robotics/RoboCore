"""Independent bimanual Jacobian assembler (Torch).

Assembles per-arm Jacobians (6xn) into a combined block-diagonal Jacobian
for dual-arm tasks.
"""
from __future__ import annotations
from typing import Sequence, Dict
import torch
import numpy as np

from robocore.modeling.robot_model import RobotModel
from robocore.kinematics.jacobian import jacobian as single_jacobian


class BiIndependentJacobianSolverTorch:
    """Assemble a block-diagonal Jacobian for two independent arms."""

    def __init__(self, left_model: RobotModel, right_model: RobotModel):
        self.left = left_model
        self.right = right_model

    def compute(self, q_left: Sequence[float] | np.ndarray | torch.Tensor,
                q_right: Sequence[float] | np.ndarray | torch.Tensor) -> torch.Tensor:
        """Compute bimanual independent Jacobian.
        
        :param q_left: Left joint configuration(s) - [n] or [B, n]
        :param q_right: Right joint configuration(s) - [n] or [B, n]
        :return: Jacobian matrix - [12, nL+nR] or [B, 12, nL+nR]
        """
        J_L = single_jacobian(self.left, q_left)
        J_R = single_jacobian(self.right, q_right)

        if hasattr(J_L, 'detach'):
            J_L = J_L.detach()
        else:
            J_L = torch.tensor(J_L, dtype=torch.float64)
        if hasattr(J_R, 'detach'):
            J_R = J_R.detach()
        else:
            J_R = torch.tensor(J_R, dtype=torch.float64)

        # Handle batch mode
        is_batch = J_L.ndim == 3
        if is_batch:
            batch_size = J_L.shape[0]
            nL = J_L.shape[2]
            nR = J_R.shape[2]
            J = torch.zeros((batch_size, 12, nL + nR), dtype=J_L.dtype, device=J_L.device)
            J[:, 0:6, 0:nL] = J_L
            J[:, 6:12, nL:] = J_R
        else:
            nL = J_L.shape[1]
            nR = J_R.shape[1]
            J = torch.zeros((12, nL + nR), dtype=J_L.dtype, device=J_L.device)
            J[0:6, 0:nL] = J_L
            J[6:12, nL:] = J_R
        return J


class BiRelativeJacobianSolverTorch:
    """Relative bimanual Jacobian (Torch).

    Assemble Jacobian for relative constraints between right and left arm.
    Uses numerical differentiation for correctness (adjoint transform is complex).
    """

    def __init__(self, left_model: RobotModel, right_model: RobotModel):
        self.left = left_model
        self.right = right_model

    def compute(self, q_left: Sequence[float] | np.ndarray | torch.Tensor, 
                q_right: Sequence[float] | np.ndarray | torch.Tensor, 
                *, constraint_type: str = 'pose') -> torch.Tensor:
        """Compute bimanual relative Jacobian.
        
        :param q_left: Left joint configuration(s) - [n] or [B, n]
        :param q_right: Right joint configuration(s) - [n] or [B, n]
        :param constraint_type: 'pose'|'position'|'orientation'
        :return: Relative constraint Jacobian - [6, nL+nR] or [B, 6, nL+nR] (or [3, nL+nR] / [B, 3, nL+nR] for position/orientation)
        """
        import numpy as np
        # Convert to numpy for batch detection
        if torch.is_tensor(q_left):
            q_left_arr = q_left.detach().cpu().numpy()
        else:
            q_left_arr = np.asarray(q_left)
        if torch.is_tensor(q_right):
            q_right_arr = q_right.detach().cpu().numpy()
        else:
            q_right_arr = np.asarray(q_right)
        
        # Detect batch mode
        is_batch = q_left_arr.ndim == 2 and q_right_arr.ndim == 2
        if is_batch:
            batch_size = q_left_arr.shape[0]
            if q_right_arr.shape[0] != batch_size:
                raise ValueError(f"Batch size mismatch: left={batch_size}, right={q_right_arr.shape[0]}")
        
        # Use numerical relative Jacobian from utils (correct adjoint handling)
        # NOTE: This returns numpy array, so convert to torch
        from robocore.kinematics.utils import relative_jacobian
        
        if is_batch:
            # Process batch
            J_rel_list = []
            for i in range(batch_size):
                J_rel = relative_jacobian(self.left, self.right, q_left_arr[i], q_right_arr[i])
                J_rel_list.append(J_rel)
            J_rel_full = np.stack(J_rel_list, axis=0)  # [B, 6, nL+nR]
            J_rel_full = torch.tensor(J_rel_full, dtype=torch.float64)
        else:
            J_rel_full = relative_jacobian(self.left, self.right, q_left, q_right)
            J_rel_full = torch.tensor(J_rel_full, dtype=torch.float64)  # [6, nL+nR]
        
        # Apply constraint type filtering
        if constraint_type == 'pose':
            return J_rel_full
        elif constraint_type == 'position':
            if is_batch:
                return J_rel_full[:, :3, :]  # [B, 3, nL+nR]
            else:
                return J_rel_full[:3, :]  # [3, nL+nR]
        elif constraint_type == 'orientation':
            if is_batch:
                return J_rel_full[:, 3:, :]  # [B, 3, nL+nR]
            else:
                return J_rel_full[3:, :]  # [3, nL+nR]
        else:
            raise ValueError("Unknown constraint_type")


class BiMultiLinkJacobianSolverTorch:
    """Multi-link Jacobian solver for arbitrary number of groups (Torch)."""

    def __init__(self, groups: Dict[str, RobotModel]):
        self.groups = groups

    def block_jacobian(self, q_by_group: Dict[str, Sequence[float]]) -> torch.Tensor:
        """
        :param q_by_group: Mapping name -> joint vector
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
            J = single_jacobian(model, q)
            J = J.detach() if hasattr(J, 'detach') else torch.tensor(J, dtype=torch.float32)
            J_blocks.append(J)
            cols_total += J.shape[1]
        rows_total = 6 * len(J_blocks)
        J_whole = torch.zeros((rows_total, cols_total), dtype=torch.float32)
        col_offset = 0
        for i, J in enumerate(J_blocks):
            r0 = 6 * i
            r1 = r0 + 6
            c1 = col_offset + J.shape[1]
            J_whole[r0:r1, col_offset:c1] = J
            col_offset = c1
        return J_whole

    def relative_jacobian_between(self, group_a: str, group_b: str,
                                  q_a: Sequence[float], q_b: Sequence[float]) -> torch.Tensor:
        """
        :param group_a: First group name
        :param group_b: Second group name
        :param q_a: Joint vector of group_a
        :param q_b: Joint vector of group_b
        :return: 6 x (n_a + nR) relative Jacobian (pose)
        """
        if not self.groups:
            raise ValueError("No groups defined.")
        a = self.groups[group_a]
        b = self.groups[group_b]
        from robocore.kinematics.utils import relative_jacobian
        J_rel = relative_jacobian(a, b, q_a, q_b)
        return J_rel.detach() if hasattr(J_rel, 'detach') else torch.tensor(J_rel, dtype=torch.float32)
