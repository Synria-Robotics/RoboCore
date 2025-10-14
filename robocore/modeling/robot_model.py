"""Tree-based robot model.

Copyright (c) 2025 Synria Robotics Co., Ltd.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from robocore.modeling.tree_model import TreeModel, SerialChain
from robocore.kinematics.tree_kinematics import TreeKinematics
from robocore.utils.beauty_logger import beauty_print


class RobotModel:
    """Tree-based robot model with depth-first indexing."""

    def __init__(self, model_path: Union[str, Path], base_link: Optional[str] = None,
                 end_link: Optional[str] = None, dtype=torch.float32, device="cpu"):
        """Initialize robot model.
        
        :param model_path: Path to URDF/MJCF file
        :param base_link: Base link name (auto-detect if None)
        :param end_link: End effector link name (auto-detect if None)
        :param dtype: Data type
        :param device: Computing device
        """
        self.model_path = str(model_path)
        self.name = os.path.basename(model_path)
        self.dtype = dtype
        self.device = device

        # Create tree model
        self.tree_model = TreeModel(model_path, dtype=dtype, device=device)

        # Create kinematics computer
        self.kinematics = TreeKinematics(self.tree_model)

        # Set base and end links
        self.base_link = base_link or self.tree_model.root_frame.name
        self.end_link = end_link or self._auto_detect_end_link()

        # Extract chain information
        self._extract_chain_info()

        # Multi-chain management
        self._chains: Dict[str, SerialChain] = {}

        beauty_print(f"📦 Loading robot model from: {self.model_path}")
        beauty_print(f"✓ Robot loaded: {self.num_dof} DOF, end_link={self.end_link}", type="success")

    def _auto_detect_end_link(self) -> str:
        """Auto-detect end effector link."""
        leaf_links = self.tree_model._find_leaf_links()

        # Prefer links with specific keywords
        priority_keywords = ['gripper', 'end', 'tip', 'hand', 'finger']

        for keyword in priority_keywords:
            for link in leaf_links:
                if keyword.lower() in link.lower():
                    return link

        # If not found, choose first leaf link
        if leaf_links:
            return leaf_links[0]
        else:
            return self.tree_model.root_frame.name

    def _extract_chain_info(self):
        """Extract chain information."""
        # Get joint information
        self.joint_list = []
        self.joint_limit = []

        for frame_idx in self.tree_model.joint_indices:
            frame_name = self.tree_model.idx_to_frame[frame_idx.item()]
            frame = self.tree_model._find_frame_by_name(frame_name)
            if frame and frame.joint.joint_type in ('revolute', 'prismatic'):
                self.joint_list.append(frame.joint.name)
                limits = frame.joint.limits or [-np.pi, np.pi]
                self.joint_limit.append(limits)

        self.num_dof = len(self.joint_list)
        if self.joint_limit:
            self.joint_limit = np.array(self.joint_limit)
        else:
            self.joint_limit = np.zeros((0, 2))

        # Get link information
        self.real_link = list(self.tree_model.frame_to_idx.keys())
        self.all_link = self.real_link.copy()

    def forward_kinematics(self, q: Union[Dict[str, float], List[float], np.ndarray, torch.Tensor],
                           *, backend: str = 'auto', return_end: bool = False,
                           device: Optional[Any] = None, dtype: Optional[Any] = None) -> Union[Dict[str, Any], Any]:
        """Compute forward kinematics.
        
        :param q: Joint configuration
        :param backend: Computation backend
        :param return_end: Return only end effector pose
        :param device: Computing device
        :param dtype: Data type
        :return: Dictionary mapping link names to transforms, or single transform
        """
        # Convert data type and device
        if dtype is not None:
            self.dtype = dtype
        if device is not None:
            self.device = device

        # Compute forward kinematics
        result = self.kinematics.forward_kinematics(q)

        if return_end:
            return result[self.end_link]
        else:
            return result

    def inverse_kinematics(self, target_pose: List[List[float]],
                           q_initial: Optional[Sequence[float]] = None,
                           backend: str = 'auto', method: str = 'pinv',
                           max_iters: int = 120, pos_tol: float = 1e-4,
                           ori_tol: float = 1e-4, **kwargs) -> Dict[str, Any]:
        """Compute inverse kinematics.
        
        :param target_pose: Target pose
        :param q_initial: Initial joint configuration
        :param backend: Computation backend
        :param method: IK method
        :param max_iters: Maximum iterations
        :param pos_tol: Position tolerance
        :param ori_tol: Orientation tolerance
        :return: IK result dictionary
        """
        try:
            from robocore.kinematics.ik import inverse_kinematics

            if q_initial is None:
                q_initial = [0.0] * self.num_dof

            return inverse_kinematics(
                self, target_pose, q_initial, backend=backend, method=method,
                max_iters=max_iters, pos_tol=pos_tol, ori_tol=ori_tol, **kwargs
            )
        except ImportError:
            beauty_print("⚠️ IK module not available", type="warning")
            return {'q': q_initial or [0.0] * self.num_dof, 'success': False}

    def jacobian(self, q: Sequence[float], *, backend: str = 'auto',
                 method: str = 'analytic', **kwargs) -> Any:
        """Compute Jacobian matrix.
        
        :param q: Joint configuration
        :param backend: Computation backend
        :param method: Jacobian method
        :return: Jacobian matrix
        """
        try:
            from robocore.kinematics.jacobian import jacobian
            return jacobian(self, q, backend=backend, method=method, **kwargs)
        except ImportError:
            beauty_print("⚠️ Jacobian module not available", type="warning")
            return torch.zeros(6, self.num_dof)

    def random_q(self, rng=None, scale: float = 0.5):
        """Generate random joint configuration.
        
        :param rng: Random number generator
        :param scale: Scaling factor for joint range
        :return: Random joint configuration
        """
        if rng is None:
            rng = np.random.default_rng()

        q = [0.0] * self.num_dof
        for i, (joint_name, limits) in enumerate(zip(self.joint_list, self.joint_limit)):
            lo, hi = limits
            mid = 0.5 * (lo + hi)
            span = 0.5 * (hi - lo) * scale
            q[i] = float(rng.uniform(mid - span, mid + span))

        return q

    def extract_chain(self, end_link: str, root_link: str = None) -> SerialChain:
        """Extract serial chain from tree.
        
        :param end_link: End link name
        :param root_link: Root link name
        :return: Serial chain object
        """
        if root_link is None:
            root_link = self.base_link

        key = f"{root_link}_{end_link}"
        if key not in self._chains:
            self._chains[key] = SerialChain(self.tree_model, end_link, root_link)

        return self._chains[key]

    def auto_discover_chains(self) -> Dict[str, str]:
        """Auto-discover available chains.
        
        :return: Dictionary mapping chain names to end links
        """
        return self.tree_model.auto_discover_chains()

    def batch_forward_kinematics(self, joint_configs: Dict[str, Union[List[float], np.ndarray]]) -> Dict[str, Dict[str, torch.Tensor]]:
        """Compute batch forward kinematics for multiple chains.
        
        :param joint_configs: Joint configurations for each chain
        :return: Dictionary mapping chain names to FK results
        """
        results = {}

        for chain_name, q in joint_configs.items():
            # Extract corresponding chain
            if chain_name in self.auto_discover_chains():
                end_link = self.auto_discover_chains()[chain_name]
                chain = self.extract_chain(end_link)
                results[chain_name] = chain.forward_kinematics(q)
            else:
                # Use full robot
                results[chain_name] = self.forward_kinematics(q)

        return results

    def to(self, dtype=None, device=None):
        """Convert to different dtype/device.
        
        :param dtype: Data type
        :param device: Computing device
        :return: Self
        """
        if dtype is not None:
            self.dtype = dtype
        if device is not None:
            self.device = device

        self.tree_model = self.tree_model.to(dtype=self.dtype, device=self.device)
        self.kinematics = TreeKinematics(self.tree_model)

        return self

    def print_tree(self, show_joints: bool = True, show_fixed: bool = False):
        """Print tree structure.
        
        :param show_joints: Show joint information
        :param show_fixed: Show fixed joints
        """
        self.tree_model.print_tree()

    def summary(self, show_chain: bool = False, title: str = "Robot Model Summary"):
        """Print robot model summary.
        
        :param show_chain: Show chain details
        :param title: Summary title
        """
        beauty_print(title, type="module", centered=True)
        beauty_print(f"File: {self.model_path}")
        beauty_print(f"DOF: {self.num_dof}  |  End Link: {self.end_link}")
        beauty_print(f"Base Link: {self.base_link}")
        beauty_print(f"Actuated Joints: {self.joint_list}")

        if show_chain:
            beauty_print("Chain Details:")
            for i, joint_name in enumerate(self.joint_list):
                limits = self.joint_limit[i] if i < len(self.joint_limit) else [-np.pi, np.pi]
                beauty_print(f"  [{i}] {joint_name}  limits: [{limits[0]:.3f}, {limits[1]:.3f}]")

    # Compatibility methods
    def fk(self, q: Sequence[float], *, backend: str = 'auto', return_end: bool = False,
           device: Optional[Any] = None, dtype: Optional[Any] = None) -> Union[Dict[str, Any], Any]:
        """Forward kinematics (compatibility method)."""
        return self.forward_kinematics(q, backend=backend, return_end=return_end, device=device, dtype=dtype)

    def ik(self, target_pose: List[List[float]], q_initial: Optional[Sequence[float]] = None,
           backend: str = 'auto', method: str = 'pinv', max_iters: int = 120,
           pos_tol: float = 1e-4, ori_tol: float = 1e-4, multi_start: int = 0,
           multi_noise: float = 0.3, random_seed: Optional[int] = None,
           torch_device: Optional[str] = None, torch_dtype: Optional[Any] = None,
           **solver_kwargs) -> Dict[str, Any]:
        """Inverse kinematics (compatibility method)."""
        return self.inverse_kinematics(target_pose, q_initial, backend, method, max_iters, pos_tol, ori_tol, **solver_kwargs)

    def spawn_chain(self, end_link: str) -> "RobotModel":
        """Create chain-specific view (compatibility method)."""
        return RobotModel(self.model_path, end_link=end_link, dtype=self.dtype, device=self.device)

    def available_leaf_links(self) -> List[str]:
        """Return leaf links (compatibility method)."""
        return self.tree_model._find_leaf_links()

    def available_end_links(self) -> List[str]:
        """Return end links (compatibility method)."""
        return self.available_leaf_links()

    def get_joint_parameter_names(self) -> List[str]:
        """Get joint parameter names (compatibility method)."""
        return self.joint_list.copy()

    def get_frame_names(self, exclude_fixed: bool = True) -> List[str]:
        """Get frame names (compatibility method)."""
        return self.real_link.copy()

    def get_link_names(self) -> List[str]:
        """Get link names (compatibility method)."""
        return self.all_link.copy()

    def get_joint_limits(self) -> Tuple[List[float], List[float]]:
        """Get joint limits (compatibility method)."""
        if len(self.joint_limit) > 0:
            return self.joint_limit[:, 0].tolist(), self.joint_limit[:, 1].tolist()
        else:
            return [], []

    def get_joint_velocity_limits(self) -> Tuple[List[float], List[float]]:
        """Get joint velocity limits (compatibility method)."""
        return [], []

    def get_joint_effort_limits(self) -> Tuple[List[float], List[float]]:
        """Get joint effort limits (compatibility method)."""
        return [], []

    def clamp(self, th):
        """Clamp joint configuration (compatibility method)."""
        th = torch.tensor(th, device=self.device, dtype=self.dtype)
        limits = torch.tensor(self.joint_limit, device=self.device, dtype=self.dtype)
        return torch.clamp(th, limits[:, 0], limits[:, 1])

    def get_all_frame_indices(self):
        """Get all frame indices (compatibility method)."""
        return torch.tensor(list(range(len(self.frame_to_idx))), dtype=torch.long, device=self.device)

    def get_frame_indices(self, *frame_names):
        """Get frame indices (compatibility method)."""
        indices = []
        for name in frame_names:
            if name in self.frame_to_idx:
                indices.append(self.frame_to_idx[name])
        return torch.tensor(indices, dtype=torch.long, device=self.device)

    def find_frame(self, name: str):
        """Find frame (compatibility method)."""
        return self.tree_model._find_frame_by_name(name)

    def find_link(self, name: str):
        """Find link (compatibility method)."""
        frame = self.find_frame(name)
        return frame.link if frame else None

    def find_joint(self, name: str):
        """Find joint (compatibility method)."""
        frame = self.find_frame(name)
        return frame.joint if frame else None


class BimanualRobotModel(RobotModel):
    """Bimanual robot model (compatibility class)."""

    def __init__(self, model_path: Union[str, Path], left_end_link: str, right_end_link: str):
        """Initialize bimanual robot model.
        
        :param model_path: Path to model file
        :param left_end_link: Left arm end link
        :param right_end_link: Right arm end link
        """
        super().__init__(model_path)

        # Create left and right arm chains
        self.left_chain = self.extract_chain(left_end_link)
        self.right_chain = self.extract_chain(right_end_link)

        self.left_end_link = left_end_link
        self.right_end_link = right_end_link

        beauty_print(f"✓ Bimanual robot initialized:", type="success")
        beauty_print(f"  Left arm: {self.left_chain.num_dof} DOF, end: {left_end_link}")
        beauty_print(f"  Right arm: {self.right_chain.num_dof} DOF, end: {right_end_link}")

    def fk(self, q_left: Sequence[float], q_right: Sequence[float],
           *, backend: str = 'auto', mode: str = 'indep', **kwargs) -> Dict[str, Any]:
        """Compute bimanual forward kinematics.
        
        :param q_left: Left arm joint configuration
        :param q_right: Right arm joint configuration
        :param backend: Computation backend
        :param mode: Computation mode
        :return: FK results for both arms
        """
        # Use batch computation
        joint_configs = {
            'left_arm': q_left,
            'right_arm': q_right
        }

        results = self.batch_forward_kinematics(joint_configs)

        return {
            'left': results['left_arm'],
            'right': results['right_arm']
        }


# Export classes
__all__ = ["RobotModel", "BimanualRobotModel"]
