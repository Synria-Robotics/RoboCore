"""Chain View for Robot Model

ChainView provides a view of a specific kinematic chain from a RobotModel.
It references the RobotModel and stores joint index mappings for efficient computation.

Copyright (c) 2025 Synria Robotics Co., Ltd.

Licensed under the MIT License.

Author: Synria Robotics Team
Website: https://synriarobotics.ai
"""

from __future__ import annotations

from typing import List, Optional, Any
import numpy as np

from robocore.modeling.parser.utils import JointSpec


class ChainView:
    """Chain view - A subset view of RobotModel, automatically handles path.
    
    ChainView provides a lightweight view of a specific kinematic chain.
    It references the RobotModel and stores joint index mappings for efficient
    computation when the same chain is used multiple times.
    
    :param robot_model: RobotModel instance
    :param end_link: End link name
    :param base_link: Base link name
    :param path: List of JointSpec objects in the path (if None, will be computed)
    """
    
    def __init__(
        self,
        robot_model: Any,
        end_link: str,
        base_link: str,
        path: Optional[List[JointSpec]] = None
    ):
        """
        Initialize chain view.
        
        :param robot_model: RobotModel instance
        :param end_link: End link name
        :param base_link: Base link name
        :param path: List of JointSpec objects in the path (if None, will be computed)
        """
        self.robot_model = robot_model  # Reference to RobotModel
        
        # If path is not provided, find it
        if path is None:
            path = robot_model._find_path(base_link, end_link)
            if not path:
                raise ValueError(f"Cannot find path from {base_link} to {end_link}")
        
        self._path = path
        self.end_link = end_link
        self.base_link = base_link
        
        # Extract joint names from path (only actuated joints)
        self._chain_joint_names = [
            j.name for j in path 
            if j.joint_type in ('revolute', 'prismatic')
        ]
        
        # Map to indices in unified configuration space
        self._chain_indices = [
            robot_model._dof_name_to_index[name]
            for name in self._chain_joint_names
            if name in robot_model._dof_name_to_index
        ]
    
    def forward_kinematics(self, q_full: np.ndarray | List[float]) -> np.ndarray:
        """Compute forward kinematics using full configuration vector.
        
        :param q_full: Full configuration vector [nq_full]
        :return: 4x4 pose matrix
        """
        # Use robot model's FK with dynamic base_link/end_link
        # This automatically handles the path and extracts the correct joint values
        return self.robot_model.fk(
            q_full,
            base_link=self.base_link,
            end_link=self.end_link,
            return_end=True
        )
    
    def jacobian(self, q_full: np.ndarray | List[float]) -> np.ndarray:
        """Compute Jacobian matrix using full configuration vector.
        
        :param q_full: Full configuration vector [nq_full]
        :return: 6 x nq_full Jacobian matrix
        """
        # Use robot model's jacobian with dynamic base_link/end_link
        # This automatically returns the full config space Jacobian
        return self.robot_model.jacobian(
            q_full,
            base_link=self.base_link,
            end_link=self.end_link
        )
    
    def extract_chain_config(self, q_full: np.ndarray | List[float]) -> np.ndarray:
        """Extract chain joint values from full configuration.
        
        :param q_full: Full configuration vector [nq_full]
        :return: Chain joint values [n_chain]
        """
        return np.array(q_full)[self._chain_indices]
    
    @property
    def joint_names(self) -> List[str]:
        """Return chain joint names.
        
        :return: List of joint names in the chain
        """
        return self._chain_joint_names
    
    @property
    def joint_indices(self) -> List[int]:
        """Return chain joint indices in unified configuration space.
        
        :return: List of joint indices
        """
        return self._chain_indices
