"""Tree-based robot model with depth-first indexing.

Copyright (c) 2025 Synria Robotics Co., Ltd.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Union, Any
from pathlib import Path

from robocore.modeling.frame import Frame, Link, Joint, Transform3d
from robocore.modeling.parser.mjcf_parser import MJCFParser
from robocore.modeling.parser.urdf_parser import load_urdf
from robocore.utils.beauty_logger import beauty_print


class TreeModel:
    """Tree-based robot model with depth-first indexing like pytorch_kinematics."""
    
    def __init__(self, model_path: str, dtype=torch.float32, device="cpu"):
        """Initialize tree model.
        
        :param model_path: Path to model file (URDF/MJCF)
        :param dtype: Data type
        :param device: Computing device
        """
        self.model_path = str(model_path)
        self.dtype = dtype
        self.device = device
        
        # Parse model
        self.parsed_model = self._parse_model()
        
        # Build tree structure
        self.root_frame = None
        self._build_tree()
        
        # Build indexing system (depth-first)
        self.frame_to_idx = {}
        self.idx_to_frame = {}
        self.parents_indices = []
        self.joint_indices = []
        self.joint_type_indices = []
        self.axes = None
        self.link_offsets = []
        self.joint_offsets = []
        
        self._build_indexing()
    
    def _parse_model(self):
        """Parse MJCF/URDF file."""
        if self.model_path.endswith('.xml'):
            return MJCFParser(self.model_path)
        elif self.model_path.endswith('.urdf'):
            return load_urdf(self.model_path)
        else:
            raise ValueError("Unsupported model file format. Only URDF and MJCF are supported.")
    
    def _build_tree(self):
        """Build tree structure from parsed model."""
        # Find root link
        root_link_name = self._find_root_link()
        
        # Create root frame
        root_link = self._create_link(root_link_name)
        root_joint = Joint("root_joint", joint_type="fixed", dtype=self.dtype, device=self.device)
        self.root_frame = Frame(root_link_name, root_link, root_joint)
        
        # Recursively build subtree
        self._build_subtree(self.root_frame, root_link_name)
    
    def _find_root_link(self):
        """Find root link (not a child of any joint)."""
        all_links = set()
        child_links = set()
        
        for joint in self.parsed_model.joints:
            all_links.add(joint.parent)
            all_links.add(joint.child)
            child_links.add(joint.child)
        
        # Root link is not a child of any joint
        root_links = all_links - child_links
        
        # Prefer base_link as root
        if 'base_link' in root_links:
            return 'base_link'
        elif len(root_links) == 1:
            return list(root_links)[0]
        else:
            return list(root_links)[0]
    
    def _create_link(self, link_name: str) -> Link:
        """Create link object from parsed model."""
        # Get link info from parsed model
        link_info = self._get_link_info(link_name)
        
        # Create transformation
        offset = None
        if link_info and 'origin' in link_info:
            origin = link_info['origin']
            offset = Transform3d(
                pos=origin.get('xyz', [0, 0, 0]),
                rot=origin.get('rpy', [0, 0, 0])
            )
        
        # Create visual geometries
        visuals = self._create_visuals(link_name)
        
        return Link(link_name, offset=offset, visuals=visuals)
    
    def _get_link_info(self, link_name: str) -> Optional[Dict]:
        """Get link information from parsed model."""
        for link in getattr(self.parsed_model, 'links', []):
            if hasattr(link, 'name') and link.name == link_name:
                return link.__dict__
        return None
    
    def _create_visuals(self, link_name: str) -> List:
        """Create visual geometries for link."""
        visuals = []
        # TODO: Implement visual geometry creation
        return visuals
    
    def _create_joint(self, joint_name: str, joint_info=None) -> Joint:
        """Create joint object from parsed model."""
        if joint_info is None:
            joint_info = self._get_joint_info(joint_name)
        
        if joint_info is None:
            return Joint(joint_name, joint_type="fixed", dtype=self.dtype, device=self.device)
        
        # Create transformation
        offset = None
        if 'origin' in joint_info:
            origin = joint_info['origin']
            offset = Transform3d(
                pos=origin.get('xyz', [0, 0, 0]),
                rot=origin.get('rpy', [0, 0, 0])
            )
        
        # Get joint type and axis
        joint_type = joint_info.get('joint_type', 'fixed')
        axis = joint_info.get('axis', [0, 0, 1])
        
        # Get limits
        limits = joint_info.get('limits')
        velocity_limits = joint_info.get('velocity_limits')
        effort_limits = joint_info.get('effort_limits')
        
        return Joint(
            joint_name, offset=offset, joint_type=joint_type,
            axis=axis, limits=limits, velocity_limits=velocity_limits,
            effort_limits=effort_limits, dtype=self.dtype, device=self.device
        )
    
    def _get_joint_info(self, joint_name: str) -> Optional[Dict]:
        """Get joint information from parsed model."""
        for joint in self.parsed_model.joints:
            if joint.name == joint_name:
                return joint.__dict__
        return None
    
    def _build_subtree(self, parent_frame: Frame, parent_link: str):
        """Recursively build subtree."""
        # Find all joints with parent_link as parent
        child_joints = [j for j in self.parsed_model.joints if j.parent == parent_link]
        
        for joint in child_joints:
            # Create child frame
            child_link = self._create_link(joint.child)
            child_joint_obj = self._create_joint(joint.name, joint.__dict__)
            child_frame = Frame(joint.child, child_link, child_joint_obj)
            
            # Add to parent frame
            parent_frame.add_child(child_frame)
            
            # Recursively build subtree
            self._build_subtree(child_frame, joint.child)
    
    def _build_indexing(self):
        """Build depth-first indexing system."""
        # Depth-first traversal to build indices
        queue = [(self.root_frame, -1, 0)]  # (frame, parent_idx, depth)
        idx = 0
        
        # Collect joint information
        joint_names = []
        joint_axes = []
        
        while queue:
            frame, parent_idx, depth = queue.pop(0)
            
            # Build mappings
            self.frame_to_idx[frame.name] = idx
            self.idx_to_frame[idx] = frame.name
            
            # Build parent path
            if parent_idx == -1:
                self.parents_indices.append([idx])
            else:
                self.parents_indices.append(self.parents_indices[parent_idx] + [idx])
            
            # Handle joint
            if frame.joint.joint_type in ('revolute', 'prismatic'):
                joint_names.append(frame.joint.name)
                joint_axes.append(frame.joint.axis)
                self.joint_indices.append(idx)
            
            # Handle joint type indices
            joint_type_idx = Joint.TYPES.index(frame.joint.joint_type)
            self.joint_type_indices.append(joint_type_idx)
            
            # Handle offsets
            self.link_offsets.append(
                frame.link.offset.get_matrix() if frame.link.offset else None
            )
            self.joint_offsets.append(
                frame.joint.offset.get_matrix() if frame.joint.offset else None
            )
            
            # Add children to queue
            for child in frame.children:
                queue.append((child, idx, depth + 1))
            
            idx += 1
        
        # Create axes tensor
        if joint_axes:
            self.axes = torch.stack(joint_axes)
        else:
            self.axes = torch.zeros([0, 3], dtype=self.dtype, device=self.device)
        
        # Convert to tensors
        self.joint_type_indices = torch.tensor(self.joint_type_indices, dtype=torch.long, device=self.device)
        self.joint_indices = torch.tensor(self.joint_indices, dtype=torch.long, device=self.device)
        self.parents_indices = [torch.tensor(p, dtype=torch.long, device=self.device) for p in self.parents_indices]
    
    def extract_serial_chain(self, end_link: str, root_link: str = None) -> 'SerialChain':
        """Extract serial chain from tree."""
        if root_link is None:
            root_link = self.root_frame.name
        
        # Find path
        path = self._find_path(root_link, end_link)
        
        # Create serial chain
        return SerialChain(self, path)
    
    def _find_path(self, root_link: str, end_link: str) -> List[Frame]:
        """Find path from root_link to end_link."""
        def dfs(frame: Frame, target: str, path: List[Frame]) -> bool:
            if frame.name == target:
                return True
            
            for child in frame.children:
                path.append(child)
                if dfs(child, target, path):
                    return True
                path.pop()
            
            return False
        
        # Start from root
        if root_link == self.root_frame.name:
            path = [self.root_frame]
        else:
            path = []
            # Find root link frame
            root_frame = self._find_frame_by_name(root_link)
            if root_frame:
                path = [root_frame]
        
        if dfs(self.root_frame, end_link, path):
            return path
        else:
            raise ValueError(f"Path from {root_link} to {end_link} not found")
    
    def _find_frame_by_name(self, name: str) -> Optional[Frame]:
        """Find frame by name."""
        def search(frame: Frame) -> Optional[Frame]:
            if frame.name == name:
                return frame
            for child in frame.children:
                result = search(child)
                if result:
                    return result
            return None
        
        return search(self.root_frame)
    
    def auto_discover_chains(self) -> Dict[str, str]:
        """Auto-discover available serial chains."""
        leaf_links = self._find_leaf_links()
        chains = {}
        
        for leaf in leaf_links:
            chain_name = self._suggest_chain_name(leaf)
            chains[chain_name] = leaf
        
        return chains
    
    def _find_leaf_links(self) -> List[str]:
        """Find leaf links (no children)."""
        all_links = set()
        parent_links = set()
        
        for joint in self.parsed_model.joints:
            all_links.add(joint.parent)
            all_links.add(joint.child)
            parent_links.add(joint.parent)
        
        # Leaf links are not parents of any joint
        return sorted(list(all_links - parent_links))
    
    def _suggest_chain_name(self, end_link: str) -> str:
        """Suggest chain name based on end link."""
        if 'left' in end_link.lower():
            return 'left_arm'
        elif 'right' in end_link.lower():
            return 'right_arm'
        elif 'gripper' in end_link.lower():
            return 'gripper'
        elif 'hand' in end_link.lower():
            return 'hand'
        elif 'finger' in end_link.lower():
            return 'finger'
        else:
            return f'chain_{end_link}'
    
    def to(self, dtype=None, device=None):
        """Convert to different dtype/device."""
        if dtype is not None:
            self.dtype = dtype
        if device is not None:
            self.device = device
        
        self.root_frame = self.root_frame.to(dtype=self.dtype, device=self.device)
        
        # Convert tensors
        if self.axes is not None:
            self.axes = self.axes.to(dtype=self.dtype, device=self.device)
        
        self.joint_type_indices = self.joint_type_indices.to(dtype=torch.long, device=self.device)
        self.joint_indices = self.joint_indices.to(dtype=torch.long, device=self.device)
        self.parents_indices = [p.to(dtype=torch.long, device=self.device) for p in self.parents_indices]
        
        # Convert offsets
        self.link_offsets = [l if l is None else l.to(dtype=self.dtype, device=self.device) for l in self.link_offsets]
        self.joint_offsets = [j if j is None else j.to(dtype=self.dtype, device=self.device) for j in self.joint_offsets]
        
        return self
    
    def print_tree(self, do_print=True):
        """Print tree structure."""
        tree = str(self.root_frame)
        if do_print:
            print(tree)
        return tree


class SerialChain:
    """Serial chain extracted from tree model."""
    
    def __init__(self, tree_model: TreeModel, end_link: str, root_link: str = None):
        """Initialize serial chain.
        
        :param tree_model: Tree model
        :param end_link: End link name
        :param root_link: Root link name
        """
        self.tree_model = tree_model
        self.end_link = end_link
        self.root_link = root_link or tree_model.root_frame.name
        
        # Extract chain path
        self.path = tree_model._find_path(root_link, end_link)
        
        # Extract chain joint information
        self._extract_chain_joints()
    
    def _extract_chain_joints(self):
        """Extract joint information for chain."""
        self.joint_list = []
        self.joint_limit = []
        
        for frame in self.path:
            if frame.joint.joint_type in ('revolute', 'prismatic'):
                self.joint_list.append(frame.joint.name)
                limits = frame.joint.limits or [-np.pi, np.pi]
                self.joint_limit.append(limits)
        
        self.num_dof = len(self.joint_list)
        if self.joint_limit:
            self.joint_limit = np.array(self.joint_limit)
        else:
            self.joint_limit = np.zeros((0, 2))
    
    def forward_kinematics(self, q: Union[List[float], np.ndarray, torch.Tensor], 
                          end_only: bool = True):
        """Compute forward kinematics for chain."""
        # TODO: Implement chain-specific FK
        if end_only:
            return torch.eye(4).unsqueeze(0)
        else:
            result = {}
            for frame in self.path:
                result[frame.name] = torch.eye(4).unsqueeze(0)
            return result
    
    def jacobian(self, q: Union[List[float], np.ndarray, torch.Tensor], 
                 locations=None, **kwargs):
        """Compute Jacobian matrix."""
        # TODO: Implement chain-specific Jacobian
        return torch.zeros(6, self.num_dof)