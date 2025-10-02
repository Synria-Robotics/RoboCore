# Trajectory Planning Guide

## Overview

RoboCore provides comprehensive trajectory planning capabilities for robot motion generation:

- **Joint Space Trajectories**: Polynomial interpolation in joint coordinates
- **Cartesian Space Trajectories**: Task space path planning with automatic IK
- **Velocity Profiles**: Time-optimal motion with acceleration/jerk limits
- **Multi-Waypoint Planning**: Smooth trajectories through multiple points

## Quick Start

```python
from robocore.modeling.robot_model import RobotModel
from robocore.planning.trajectory import (
    quintic_polynomial_trajectory,
    linear_cartesian_trajectory,
    trapezoidal_velocity_profile
)

# Load robot
model = RobotModel('robot.urdf')

# Define start and end configurations
q_start = np.zeros(6)
q_end = np.array([1.0, 0.5, -0.3, 0.0, 0.5, 0.0])

# Generate quintic polynomial trajectory
t, q, qd, qdd = quintic_polynomial_trajectory(
    q_start, q_end, duration=2.0, num_points=100
)
```

---

## Joint Space Trajectories

### 1. Linear Interpolation

Simple linear interpolation with constant velocity:

```python
from robocore.planning.trajectory import linear_joint_trajectory

t, q, qd, qdd = linear_joint_trajectory(
    q_start, q_end, duration=2.0, num_points=100
)
```

**Characteristics**:
- ✓ Simple and fast
- ✗ Discontinuous velocity at endpoints (infinite acceleration)
- ✗ Not smooth

**Use when**: Simple point-to-point motion, speed is critical

---

### 2. Cubic Polynomial Trajectory

C¹ continuous trajectory (position + velocity continuous):

```python
from robocore.planning.trajectory import cubic_polynomial_trajectory

t, q, qd, qdd = cubic_polynomial_trajectory(
    q_start, q_end, duration=2.0, num_points=100,
    v_start=None,  # Optional: specify start velocity
    v_end=None     # Optional: specify end velocity
)
```

**Characteristics**:
- ✓ Smooth velocity transitions
- ✓ Zero velocity at endpoints (if not specified)
- ✗ Discontinuous acceleration at endpoints
- Formula: `q(t) = a₀ + a₁t + a₂t² + a₃t³`

**Use when**: Basic smooth motion, velocity control needed

---

### 3. Quintic Polynomial Trajectory (Recommended)

C² continuous trajectory (position + velocity + acceleration continuous):

```python
from robocore.planning.trajectory import quintic_polynomial_trajectory

t, q, qd, qdd = quintic_polynomial_trajectory(
    q_start, q_end, duration=2.0, num_points=100,
    v_start=None,  # Optional: start velocity
    v_end=None,    # Optional: end velocity
    a_start=None,  # Optional: start acceleration
    a_end=None     # Optional: end acceleration
)
```

**Characteristics**:
- ✓ Smoothest motion
- ✓ Zero velocity and acceleration at endpoints (if not specified)
- ✓ Minimizes jerk
- Formula: `q(t) = a₀ + a₁t + a₂t² + a₃t³ + a₄t⁴ + a₅t⁵`

**Use when**: High-quality smooth motion required (default choice)

---

### 4. Multi-Waypoint Trajectory

Connect multiple waypoints with smooth transitions:

```python
from robocore.planning.trajectory import multi_waypoint_trajectory

waypoints = np.array([
    q_start,
    q_intermediate_1,
    q_intermediate_2,
    q_end
])

t, q, qd, qdd = multi_waypoint_trajectory(
    waypoints,
    duration=0.5,        # Duration per segment
    num_points=50,       # Points per segment
    method='quintic'     # 'linear', 'cubic', or 'quintic'
)
```

**Characteristics**:
- ✓ Smooth velocity between segments
- ✓ Flexible interpolation method
- ✓ Efficient for complex paths

---

## Cartesian Space Trajectories

### 1. Linear Cartesian Trajectory

End-effector moves in straight line between poses:

```python
from robocore.planning.trajectory import linear_cartesian_trajectory
from robocore.kinematics.fk import forward_kinematics

# Get start and end poses
T_start = forward_kinematics(model, q_start, backend='numpy', return_end=True)
T_end = forward_kinematics(model, q_end, backend='numpy', return_end=True)

# Generate trajectory
t, poses, q = linear_cartesian_trajectory(
    model,
    T_start,           # 4x4 start pose
    T_end,             # 4x4 end pose
    duration=2.0,
    num_points=50,
    q_init=q_start,    # Initial guess for IK
    ik_backend='numpy',
    ik_method='dls'
)
```

**Returns**:
- `t`: Time array
- `poses`: Cartesian poses [N, 4, 4]
- `q`: Joint configurations [N, DOF]

**Features**:
- Linear position interpolation
- SLERP (Spherical Linear Interpolation) for orientation
- Automatic inverse kinematics solving

---

### 2. Circular Cartesian Trajectory

End-effector moves along circular arc:

```python
from robocore.planning.trajectory import circular_cartesian_trajectory

t, poses, q = circular_cartesian_trajectory(
    model,
    center=np.array([0.3, 0.0, 0.5]),      # Center of circle
    normal=np.array([0.0, 0.0, 1.0]),      # Normal to plane (Z-axis)
    radius=0.1,                             # Radius in meters
    start_angle=0,                          # Start angle (radians)
    end_angle=np.pi,                        # End angle (radians)
    duration=3.0,
    num_points=100,
    orientation='constant',                 # 'constant', 'tangent', or 3x3 matrix
    q_init=q_start
)
```

**Orientation Options**:
- `'constant'`: Keep initial orientation
- `'tangent'`: Orient along tangent to circle
- `3x3 matrix`: Use fixed orientation

**Use cases**: Arc welding, circular inspection paths, polishing

---

### 3. Cartesian Waypoint Trajectory

Linear interpolation through multiple Cartesian waypoints:

```python
from robocore.planning.trajectory import cartesian_waypoint_trajectory

waypoint_poses = np.array([
    T_start,
    T_intermediate_1,
    T_intermediate_2,
    T_end
])  # Shape: (num_waypoints, 4, 4)

durations = np.array([1.0, 0.5, 1.5])  # Or single float for all segments

t, poses, q = cartesian_waypoint_trajectory(
    model,
    waypoint_poses,
    durations,
    num_points_per_segment=50,
    q_init=q_start
)
```

---

## Velocity Profiles

### 1. Trapezoidal Velocity Profile

Standard industrial velocity profile with constant acceleration:

```python
from robocore.planning.trajectory import trapezoidal_velocity_profile

t, s, v, a = trapezoidal_velocity_profile(
    distance=1.0,      # Total distance
    v_max=0.5,         # Maximum velocity
    a_max=1.0,         # Maximum acceleration
    v_start=0.0,       # Start velocity (optional)
    v_end=0.0,         # End velocity (optional)
    num_points=100
)
```

**Phases**:
1. **Acceleration**: Linear ramp-up to `v_max`
2. **Cruise**: Constant velocity at `v_max`
3. **Deceleration**: Linear ramp-down to `v_end`

**Returns**:
- `t`: Time array
- `s`: Position (distance traveled)
- `v`: Velocity
- `a`: Acceleration

**Use when**: Standard point-to-point motion, known velocity/acceleration limits

---

### 2. S-Curve Velocity Profile

Jerk-limited profile for smoother motion:

```python
from robocore.planning.trajectory import s_curve_velocity_profile

t, s, v, a, j = s_curve_velocity_profile(
    distance=1.0,
    v_max=0.5,
    a_max=1.0,
    j_max=5.0,         # Maximum jerk
    v_start=0.0,
    v_end=0.0,
    num_points=100
)
```

**Phases** (7 phases):
1. Jerk-in (acceleration increases)
2. Constant acceleration
3. Jerk-out (acceleration decreases)
4. Constant velocity
5. Jerk-in (deceleration increases)
6. Constant deceleration
7. Jerk-out (deceleration decreases)

**Benefits**:
- Smoother motion (less vibration)
- Reduced mechanical stress
- Better for high-speed operations

**Use when**: High-precision applications, delicate payloads, high-speed motion

---

### 3. Apply Velocity Profile to Trajectory

Scale a geometric path to follow a velocity profile:

```python
from robocore.planning.trajectory import scale_trajectory_to_profile

# Generate geometric path (e.g., from waypoints)
q_path = np.linspace(q_start, q_end, 100)

# Generate velocity profile
_, _, v, _ = trapezoidal_velocity_profile(
    distance=1.0, v_max=0.5, a_max=1.0
)

# Combine
t, q, qd, qdd = scale_trajectory_to_profile(
    q_path, v, duration=2.0
)
```

**Use case**: Decouple geometric path from velocity planning

---

## Examples

### Example 1: Simple Pick-and-Place

```python
import numpy as np
from robocore.modeling.robot_model import RobotModel
from robocore.planning.trajectory import quintic_polynomial_trajectory

model = RobotModel('robot.urdf')

# Home position
q_home = np.zeros(6)

# Pick position
q_pick = np.array([1.0, 0.5, -0.3, 0.0, 0.5, 0.0])

# Place position
q_place = np.array([1.2, 0.3, -0.5, 0.0, 0.7, 0.0])

# 1. Home → Pick
t1, q1, qd1, qdd1 = quintic_polynomial_trajectory(
    q_home, q_pick, duration=2.0
)

# 2. Pick → Place
t2, q2, qd2, qdd2 = quintic_polynomial_trajectory(
    q_pick, q_place, duration=1.5
)

# 3. Place → Home
t3, q3, qd3, qdd3 = quintic_polynomial_trajectory(
    q_place, q_home, duration=2.0
)

# Concatenate trajectories
t_total = np.concatenate([t1, t2[1:] + t1[-1], t3[1:] + t1[-1] + t2[-1]])
q_total = np.vstack([q1, q2[1:], q3[1:]])
```

---

### Example 2: Circular Welding Path

```python
from robocore.planning.trajectory import circular_cartesian_trajectory

# Weld a circle in XY plane
center = np.array([0.5, 0.0, 0.3])  # Circle center
normal = np.array([0.0, 0.0, 1.0])  # Normal (Z-axis)

t, poses, q = circular_cartesian_trajectory(
    model,
    center=center,
    normal=normal,
    radius=0.05,              # 5cm radius
    start_angle=0,
    end_angle=2*np.pi,        # Full circle
    duration=10.0,            # 10 seconds
    orientation='tangent',    # Orient along weld direction
    q_init=q_start
)

# Execute trajectory
for qi in q:
    robot.move_to(qi)
```

---

### Example 3: Time-Optimal Velocity Profile

```python
from robocore.planning.trajectory import (
    multi_waypoint_trajectory,
    trapezoidal_velocity_profile,
    scale_trajectory_to_profile
)

# Generate geometric path through waypoints
waypoints = np.array([q1, q2, q3, q4, q5])
_, q_path, _, _ = multi_waypoint_trajectory(
    waypoints, duration=1.0, method='quintic'
)

# Compute path length
path_length = 0.0
for i in range(1, len(q_path)):
    path_length += np.linalg.norm(q_path[i] - q_path[i-1])

# Generate time-optimal velocity profile
_, _, v, _ = trapezoidal_velocity_profile(
    distance=path_length,
    v_max=2.0,      # Max joint velocity
    a_max=5.0,      # Max joint acceleration
)

# Apply profile to path
duration = len(v) * 0.01  # Assuming 100Hz
t, q, qd, qdd = scale_trajectory_to_profile(q_path, v, duration)
```

---

## Performance Tips

### 1. Use Batch Operations for GPU Acceleration

```python
# For large-scale trajectory validation
from robocore.kinematics.fk_utils.batch_fk_torch import batch_forward_kinematics_torch

# Validate entire trajectory on GPU
q_batch_torch = torch.tensor(q, dtype=torch.float32, device='cuda:0')
T_batch = batch_forward_kinematics_torch(model, q_batch_torch, end_link='tool0')

# Check collisions, workspace limits, etc.
```

### 2. Pre-compute IK Seeds

For Cartesian trajectories, use previous solution as seed:

```python
# Already implemented in linear_cartesian_trajectory
q_current = q_init.copy()
for T in poses:
    result = inverse_kinematics(model, T, q_current)  # Use q_current as seed
    q_current = result['q']  # Update for next iteration
```

### 3. Parallel Trajectory Generation

```python
import torch

# Generate multiple trajectories in parallel
q_starts = torch.tensor([q1, q2, q3, ...])  # [N, DOF]
q_ends = torch.tensor([q1_end, q2_end, q3_end, ...])  # [N, DOF]

# Vectorized cubic interpolation
t = torch.linspace(0, 1, 100)
q_batch = q_starts[:, None, :] + (q_ends - q_starts)[:, None, :] * t[None, :, None]
```

---

## Configuration

Use OmegaConf for trajectory settings:

```yaml
# config.yaml
trajectory:
  default_duration: 2.0
  default_num_points: 100
  method: 'quintic'  # 'linear', 'cubic', 'quintic'
  
  velocity_profile:
    type: 'trapezoidal'  # 'trapezoidal', 's_curve'
    v_max: 0.5
    a_max: 1.0
    j_max: 5.0
  
  ik:
    backend: 'numpy'
    method: 'dls'
    max_iterations: 100
    tolerance: 1e-4
```

```python
from robocore.configs.config_manager import ConfigManager

config = ConfigManager.load('config.yaml')

t, q, qd, qdd = quintic_polynomial_trajectory(
    q_start, q_end,
    duration=config.trajectory.default_duration,
    num_points=config.trajectory.default_num_points
)
```

---

## Demo Script

Run the comprehensive demo:

```bash
# Basic demo
python examples/demo_trajectory.py --robot bessica --arm left

# With matplotlib visualization
python examples/demo_trajectory.py --robot bessica --arm left --plot
```

**Demonstrates**:
1. Joint space polynomial trajectories (linear, cubic, quintic)
2. Multi-waypoint trajectories
3. Velocity profiles (trapezoidal, S-curve)
4. Cartesian space linear trajectory

---

## API Reference

### Joint Space

| Function | Description | Continuity |
|----------|-------------|------------|
| `linear_joint_trajectory` | Linear interpolation | C⁰ (position only) |
| `cubic_polynomial_trajectory` | Cubic polynomial | C¹ (pos + vel) |
| `quintic_polynomial_trajectory` | Quintic polynomial | C² (pos + vel + acc) |
| `multi_waypoint_trajectory` | Multiple waypoints | Depends on method |

### Cartesian Space

| Function | Description |
|----------|-------------|
| `linear_cartesian_trajectory` | Straight line in task space |
| `circular_cartesian_trajectory` | Circular arc in 3D |
| `cartesian_waypoint_trajectory` | Multiple Cartesian waypoints |

### Velocity Profiles

| Function | Description | Limits |
|----------|-------------|--------|
| `trapezoidal_velocity_profile` | Standard profile | Velocity, Acceleration |
| `s_curve_velocity_profile` | Jerk-limited | Velocity, Acceleration, Jerk |
| `constant_velocity_profile` | Constant speed | Velocity only |
| `scale_trajectory_to_profile` | Apply profile to path | - |

---

## References

1. **Trajectory Planning for Automatic Machines and Robots**
   - Biagiotti & Melchiorri (Springer, 2008)
   - Comprehensive theory on polynomial trajectories and velocity profiles

2. **Modern Robotics: Mechanics, Planning, and Control**
   - Lynch & Park (Cambridge, 2017)
   - Chapter 9: Trajectory Generation

3. **Robot Motion Planning**
   - Latombe (Springer, 1991)
   - Classic reference for motion planning algorithms

---

## Next Steps

After mastering trajectory planning, explore:

1. **Collision Detection**: Check trajectories for self-collision and environment collision
2. **Optimal Planning**: RRT, RRT*, PRM for complex environments
3. **Real-time Replanning**: Dynamic obstacle avoidance
4. **Force Control**: Combine trajectory with force/torque control
5. **Learning-based Planning**: Use RL/IL for complex tasks

See `robocore/planning/` for advanced planning algorithms (coming soon).
