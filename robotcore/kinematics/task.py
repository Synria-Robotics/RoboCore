"""Task abstraction for multi-link/multi-task control.

Provides unified interface for different types of kinematic tasks:
- absolute: end-effector pose tracking
- relative: relative pose constraints between groups
- centering: joint centering
- contact: contact/stationary constraints
"""
from __future__ import annotations
from typing import Dict, Any, Optional, Sequence, List, Union
from dataclasses import dataclass
import numpy as np


@dataclass
class Task:
    """Unified task representation for multi-link control.
    
    :param type: Task type ('absolute', 'relative', 'centering', 'contact')
    :param group: Primary group name (for absolute/centering/contact)
    :param group_a: First group name (for relative)
    :param group_b: Second group name (for relative)
    :param target: Target pose/position (4x4 matrix or 3D vector)
    :param weight: Task weight for weighted mode
    :param priority: Task priority for hierarchical mode (lower = higher priority)
    :param row_mask: Which dimensions to constrain (6D for pose, 3D for position)
    :param joint_indices: Which joints to control (None = all joints in group)
    """
    type: str
    group: Optional[str] = None
    group_a: Optional[str] = None
    group_b: Optional[str] = None
    target: Optional[Union[np.ndarray, Sequence[float]]] = None
    weight: float = 1.0
    priority: int = 0
    row_mask: Optional[Sequence[bool]] = None
    joint_indices: Optional[Sequence[int]] = None

    def __post_init__(self):
        """Validate task configuration."""
        if self.type == 'absolute':
            if self.group is None:
                raise ValueError("Absolute task requires 'group'")
            if self.target is None:
                raise ValueError("Absolute task requires 'target'")
        elif self.type == 'relative':
            if self.group_a is None or self.group_b is None:
                raise ValueError("Relative task requires 'group_a' and 'group_b'")
            if self.target is None:
                raise ValueError("Relative task requires 'target'")
        elif self.type == 'centering':
            if self.group is None:
                raise ValueError("Centering task requires 'group'")
        elif self.type == 'contact':
            if self.group is None:
                raise ValueError("Contact task requires 'group'")
        else:
            raise ValueError(f"Unknown task type: {self.type}")


def absolute_task(group: str, target: Union[np.ndarray, Sequence[float]], 
                 weight: float = 1.0, priority: int = 0,
                 row_mask: Optional[Sequence[bool]] = None) -> Task:
    """Create absolute pose tracking task.
    
    :param group: Group name
    :param target: Target pose (4x4) or position (3D)
    :param weight: Task weight
    :param priority: Task priority
    :param row_mask: Constraint dimensions
    :return: Absolute task
    """
    return Task(
        type='absolute',
        group=group,
        target=target,
        weight=weight,
        priority=priority,
        row_mask=row_mask
    )


def relative_task(group_a: str, group_b: str, target: Union[np.ndarray, Sequence[float]],
                 weight: float = 1.0, priority: int = 0,
                 row_mask: Optional[Sequence[bool]] = None) -> Task:
    """Create relative pose constraint task.
    
    :param group_a: First group name
    :param group_b: Second group name
    :param target: Desired relative transformation
    :param weight: Task weight
    :param priority: Task priority
    :param row_mask: Constraint dimensions
    :return: Relative task
    """
    return Task(
        type='relative',
        group_a=group_a,
        group_b=group_b,
        target=target,
        weight=weight,
        priority=priority,
        row_mask=row_mask
    )


def centering_task(group: str, weight: float = 1.0, priority: int = 0) -> Task:
    """Create joint centering task.
    
    :param group: Group name
    :param weight: Task weight
    :param priority: Task priority
    :return: Centering task
    """
    return Task(
        type='centering',
        group=group,
        weight=weight,
        priority=priority
    )


def contact_task(group: str, weight: float = 1.0, priority: int = 0) -> Task:
    """Create contact/stationary constraint task.
    
    :param group: Group name
    :param weight: Task weight
    :param priority: Task priority
    :return: Contact task
    """
    return Task(
        type='contact',
        group=group,
        weight=weight,
        priority=priority
    )


def organize_by_priority(tasks):
    """Organize tasks by priority level for hierarchical solving.
    
    :param tasks: List of Task objects
    :return: List of lists, each containing tasks at same priority (sorted high to low)
    """
    from collections import defaultdict
    priority_map = defaultdict(list)
    for task in tasks:
        priority_map[task.priority].append(task)
    
    # Sort by priority (0 is highest)
    sorted_priorities = sorted(priority_map.keys())
    return [[task for task in priority_map[p]] for p in sorted_priorities]

