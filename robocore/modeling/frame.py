"""Robot basic data structures.

Copyright (c) 2025 Synria Robotics Co., Ltd.
"""

import torch
import numpy as np
from typing import List, Optional, Dict, Any, Union


class Transform3d:
    """3D transformation representation."""
    
    def __init__(self, pos=None, rot=None, matrix=None):
        """Initialize transformation.
        
        :param pos: Position vector [x, y, z]
        :param rot: Rotation (RPY angles or quaternion)
        :param matrix: 4x4 transformation matrix
        """
        if matrix is not None:
            self.matrix = torch.tensor(matrix, dtype=torch.float32)
        else:
            self.matrix = torch.eye(4, dtype=torch.float32)
            if pos is not None:
                self.matrix[:3, 3] = torch.tensor(pos, dtype=torch.float32)
            # TODO: Handle rotation properly
    
    def get_matrix(self):
        """Get transformation matrix."""
        return self.matrix
    
    def to(self, dtype=None, device=None):
        """Convert to different dtype/device."""
        if dtype is not None:
            self.matrix = self.matrix.to(dtype=dtype)
        if device is not None:
            self.matrix = self.matrix.to(device=device)
        return self


class Visual:
    """Visual geometry representation."""
    
    TYPES = ['box', 'cylinder', 'sphere', 'capsule', 'mesh']
    
    def __init__(self, offset: Optional[Transform3d] = None, 
                 geom_type: str = None, geom_param: Dict = None):
        """Initialize visual geometry.
        
        :param offset: Geometry offset transformation
        :param geom_type: Geometry type
        :param geom_param: Geometry parameters
        """
        self.offset = offset
        self.geom_type = geom_type
        self.geom_param = geom_param or {}
    
    def to(self, dtype=None, device=None):
        """Convert to different dtype/device."""
        if self.offset is not None:
            self.offset = self.offset.to(dtype=dtype, device=device)
        return self


class Link:
    """Robot link representation."""
    
    def __init__(self, name: str, offset: Optional[Transform3d] = None, 
                 visuals: List[Visual] = None):
        """Initialize robot link.
        
        :param name: Link name
        :param offset: Link offset transformation
        :param visuals: List of visual geometries
        """
        self.name = name
        self.offset = offset
        self.visuals = visuals or []
    
    def to(self, dtype=None, device=None):
        """Convert to different dtype/device."""
        if self.offset is not None:
            self.offset = self.offset.to(dtype=dtype, device=device)
        for visual in self.visuals:
            visual.to(dtype=dtype, device=device)
        return self


class Joint:
    """Robot joint representation."""
    
    TYPES = ['fixed', 'revolute', 'prismatic']
    
    def __init__(self, name: str, offset: Optional[Transform3d] = None,
                 joint_type: str = 'fixed', axis: Union[List[float], np.ndarray, torch.Tensor] = None,
                 limits: Optional[List[float]] = None,
                 velocity_limits: Optional[List[float]] = None,
                 effort_limits: Optional[List[float]] = None,
                 dtype=torch.float32, device="cpu"):
        """Initialize robot joint.
        
        :param name: Joint name
        :param offset: Joint offset transformation
        :param joint_type: Joint type ('fixed', 'revolute', 'prismatic')
        :param axis: Joint axis vector
        :param limits: Joint limits [lower, upper]
        :param velocity_limits: Velocity limits
        :param effort_limits: Effort limits
        :param dtype: Data type
        :param device: Computing device
        """
        self.name = name
        self.offset = offset
        self.joint_type = joint_type
        
        if axis is None:
            self.axis = torch.tensor([0.0, 0.0, 1.0], dtype=dtype, device=device)
        else:
            if isinstance(axis, (list, np.ndarray)):
                self.axis = torch.tensor(axis, dtype=dtype, device=device)
            else:
                self.axis = axis.to(dtype=dtype, device=device)
        
        # Normalize axis vector
        self.axis = self.axis / self.axis.norm()
        
        self.limits = limits
        self.velocity_limits = velocity_limits
        self.effort_limits = effort_limits
    
    def to(self, dtype=None, device=None):
        """Convert to different dtype/device."""
        if dtype is not None:
            self.axis = self.axis.to(dtype=dtype)
        if device is not None:
            self.axis = self.axis.to(device=device)
        
        if self.offset is not None:
            self.offset = self.offset.to(dtype=dtype, device=device)
        return self
    
    def clamp(self, joint_position):
        """Clamp joint position to limits."""
        if self.limits is None:
            return joint_position
        else:
            return torch.clamp(joint_position, self.limits[0], self.limits[1])


class Frame:
    """Robot frame node."""
    
    def __init__(self, name: str, link: Link, joint: Joint, 
                 children: List['Frame'] = None):
        """Initialize robot frame node.
        
        :param name: Frame name
        :param link: Link object
        :param joint: Joint object
        :param children: List of child frames
        """
        self.name = name
        self.link = link
        self.joint = joint
        self.children = children or []
        self.parent = None  # Back reference
    
    def add_child(self, child: 'Frame'):
        """Add child frame."""
        child.parent = self
        self.children.append(child)
    
    def to(self, dtype=None, device=None):
        """Convert to different dtype/device."""
        self.joint = self.joint.to(dtype=dtype, device=device)
        self.link = self.link.to(dtype=dtype, device=device)
        self.children = [c.to(dtype=dtype, device=device) for c in self.children]
        return self
    
    def print_tree(self, prefix: str = "", is_last: bool = True, show_joints: bool = True):
        """Print tree structure."""
        connector = "└── " if is_last else "├── "
        extension = "    " if is_last else "│   "
        
        print(f"{prefix}{connector}{self.name}")
        
        if show_joints and self.joint.joint_type != 'fixed':
            joint_symbol = "⚙" if self.joint.joint_type in ("revolute", "prismatic") else "⊗"
            joint_line = f"{prefix}{extension}  {joint_symbol} {self.joint.name} ({self.joint.joint_type})"
            print(joint_line)
        
        for i, child in enumerate(self.children):
            is_last_child = (i == len(self.children) - 1)
            child_prefix = prefix + extension
            child.print_tree(child_prefix, is_last_child, show_joints)
    
    def __str__(self):
        """Return string representation of tree structure."""
        lines = []
        
        def _str_helper(frame, prefix="", is_last=True):
            connector = "└── " if is_last else "├── "
            extension = "    " if is_last else "│   "
            
            lines.append(f"{prefix}{connector}{frame.name}")
            
            for i, child in enumerate(frame.children):
                is_last_child = (i == len(frame.children) - 1)
                child_prefix = prefix + extension
                _str_helper(child, child_prefix, is_last_child)
        
        _str_helper(self)
        return "\n".join(lines)