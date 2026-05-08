"""Robot Model Abstraction

Copyright (c) 2025 Synria Robotics Co., Ltd.

Licensed under the MIT License.

Author: Synria Robotics Team
Website: https://synriarobotics.ai
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Any, ClassVar, Tuple, Union


@dataclass
class JointSpec:
    """Actuated joint specification.

    :param name: joint name.
    :param index: index in configuration.
    :param joint_type: revolute/prismatic.
    :param axis: axis vector (3,).
    :param parent: parent link.
    :param child: child link.
    :param origin_xyz: translation of joint frame.
    :param origin_rpy: rpy of joint frame.
    :param limit_lower: lower limit or None.
    :param limit_upper: upper limit or None.
    """

    name: str
    index: int
    joint_type: str
    axis: List[float]
    parent: str
    child: str
    origin_xyz: List[float]
    origin_rpy: List[float]
    limit_lower: Optional[float]
    limit_upper: Optional[float]
