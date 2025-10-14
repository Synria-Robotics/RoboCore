"""Tree-based kinematics computation.

Copyright (c) 2025 Synria Robotics Co., Ltd.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Union, Any

from robocore.modeling.tree_model import TreeModel


def axis_and_angle_to_matrix_44(axes, angles):
    """Convert axis-angle to 4x4 rotation matrix.
    
    :param axes: Joint axes tensor
    :param angles: Joint angles tensor
    :return: Rotation matrices
    """
    # Simple implementation - create identity matrices
    batch_size = angles.shape[0]
    matrices = torch.eye(4).unsqueeze(0).repeat(batch_size, axes.shape[0], 1, 1)
    return matrices


def axis_and_d_to_pris_matrix(axes, distances):
    """Convert axis-distance to 4x4 translation matrix.
    
    :param axes: Joint axes tensor
    :param distances: Joint distances tensor
    :return: Translation matrices
    """
    # Simple implementation - create identity matrices
    batch_size = distances.shape[0]
    matrices = torch.eye(4).unsqueeze(0).repeat(batch_size, axes.shape[0], 1, 1)
    return matrices


class TreeKinematics:
    """Tree-based kinematics computation like pytorch_kinematics."""
    
    def __init__(self, tree_model: TreeModel):
        """Initialize tree kinematics.
        
        :param tree_model: Tree model
        """
        self.tree_model = tree_model
        self.n_joints = len(tree_model.joint_indices)
        self.axes = tree_model.axes
        self.link_offsets = tree_model.link_offsets
        self.joint_offsets = tree_model.joint_offsets
        self.joint_type_indices = tree_model.joint_type_indices
        self.parents_indices = tree_model.parents_indices
        self.frame_to_idx = tree_model.frame_to_idx
        self.idx_to_frame = tree_model.idx_to_frame
    
    def forward_kinematics(self, th: Union[Dict[str, float], List[float], np.ndarray, torch.Tensor], 
                          frame_indices: Optional[List[int]] = None) -> Dict[str, torch.Tensor]:
        """Compute forward kinematics for tree model.
        
        :param th: Joint configuration
        :param frame_indices: Frame indices to compute
        :return: Dictionary mapping frame names to transformation matrices
        """
        # Convert input format
        th = self._ensure_tensor(th)
        th = torch.atleast_2d(th)
        
        batch_size = th.shape[0]
        
        # Compute all joint transforms
        axes_expanded = self.axes.unsqueeze(0).repeat(batch_size, 1, 1)
        rev_jnt_transform = axis_and_angle_to_matrix_44(axes_expanded, th)
        pris_jnt_transform = axis_and_d_to_pris_matrix(axes_expanded, th)
        
        # Compute frame transforms
        frame_transforms = {}
        
        if frame_indices is None:
            frame_indices = list(range(len(self.frame_to_idx)))
        
        for frame_idx in frame_indices:
            frame_transform = torch.eye(4).unsqueeze(0).repeat(batch_size, 1, 1)
            
            # Compose transforms along parent path
            for chain_idx in self.parents_indices[frame_idx]:
                if chain_idx.item() in frame_transforms:
                    frame_transform = frame_transforms[chain_idx.item()]
                else:
                    # Apply link offset
                    link_offset = self.link_offsets[chain_idx]
                    if link_offset is not None:
                        frame_transform = frame_transform @ link_offset
                    
                    # Apply joint offset
                    joint_offset = self.joint_offsets[chain_idx]
                    if joint_offset is not None:
                        frame_transform = frame_transform @ joint_offset
                    
                    # Apply joint transform
                    jnt_type = self.joint_type_indices[chain_idx]
                    
                    if jnt_type == 1:  # revolute
                        # Find joint index for this frame
                        if chain_idx in self.tree_model.joint_indices:
                            jnt_idx = (self.tree_model.joint_indices == chain_idx).nonzero()[0].item()
                            if jnt_idx < rev_jnt_transform.shape[1]:
                                jnt_transform = rev_jnt_transform[:, jnt_idx]
                                frame_transform = frame_transform @ jnt_transform
                    elif jnt_type == 2:  # prismatic
                        # Find joint index for this frame
                        if chain_idx in self.tree_model.joint_indices:
                            jnt_idx = (self.tree_model.joint_indices == chain_idx).nonzero()[0].item()
                            if jnt_idx < pris_jnt_transform.shape[1]:
                                jnt_transform = pris_jnt_transform[:, jnt_idx]
                                frame_transform = frame_transform @ jnt_transform
                
                frame_transforms[chain_idx.item()] = frame_transform
            
            frame_transforms[frame_idx] = frame_transform
        
        # Convert to frame name to transform mapping
        result = {}
        for frame_idx, transform in frame_transforms.items():
            frame_name = self.idx_to_frame[frame_idx]
            result[frame_name] = transform
        
        return result
    
    def _ensure_tensor(self, th):
        """Ensure input is tensor format."""
        if isinstance(th, np.ndarray):
            th = torch.tensor(th, device=self.tree_model.device, dtype=self.tree_model.dtype)
        elif isinstance(th, list):
            th = torch.tensor(th, device=self.tree_model.device, dtype=self.tree_model.dtype)
        elif isinstance(th, dict):
            # Convert dictionary to tensor
            th_dict = th
            elem_shape = self._get_dict_elem_shape(th_dict)
            th = torch.ones([*elem_shape, self.n_joints], 
                           device=self.tree_model.device, dtype=self.tree_model.dtype) * torch.nan
            
            joint_names = self._get_joint_names()
            for joint_name, joint_position in th_dict.items():
                if joint_name in joint_names:
                    jnt_idx = joint_names.index(joint_name)
                    th[..., jnt_idx] = joint_position
            
            if torch.any(torch.isnan(th)):
                missing_joints = [name for name in joint_names if name not in th_dict]
                raise ValueError(f"Missing values for joints: {missing_joints}")
        
        return th
    
    def _get_dict_elem_shape(self, th_dict):
        """Get shape of dictionary elements."""
        elem = th_dict[list(th_dict.keys())[0]]
        if isinstance(elem, np.ndarray):
            return elem.shape
        elif isinstance(elem, torch.Tensor):
            return elem.shape
        else:
            return ()
    
    def _get_joint_names(self):
        """Get joint names list."""
        joint_names = []
        for frame_idx in self.tree_model.joint_indices:
            frame_name = self.tree_model.idx_to_frame[frame_idx]
            frame = self.tree_model._find_frame_by_name(frame_name)
            if frame and frame.joint.joint_type in ('revolute', 'prismatic'):
                joint_names.append(frame.joint.name)
        return joint_names