# Trajectory Planning Examples

This directory contains examples demonstrating all trajectory planning methods in RoboCore.

## Examples

### 1. Joint Space Planning (`demo_joint_space_planning.py`)

Demonstrates all joint space trajectory planning methods:

- **Cubic Polynomial**: C1 continuous (position and velocity)
- **Quintic Polynomial**: C2 continuous (position, velocity, acceleration)
- **Septic Polynomial**: C3 continuous (position, velocity, acceleration, jerk)
- **B-Spline**: Smooth curves through multiple waypoints
- **Multi-Segment**: Piecewise polynomial trajectories

**Usage**:
```bash
python examples/planning/demo_joint_space_planning.py
python examples/planning/demo_joint_space_planning.py --plot
```

### 2. Cartesian Space Planning (`demo_cartesian_space_planning.py`)

Demonstrates all Cartesian space trajectory planning methods:

- **Linear Position**: Straight-line position interpolation
- **SLERP**: Spherical linear interpolation for orientation
- **Circular Arc**: Circular arc through three points
- **Spline Curve**: Smooth spline curves through multiple waypoints

**Usage**:
```bash
python examples/planning/demo_cartesian_space_planning.py
python examples/planning/demo_cartesian_space_planning.py --plot
```

### 3. Velocity Profiles (`demo_velocity_profiles.py`)

Demonstrates velocity profile generation:

- **Trapezoidal**: Three-phase profile (acceleration, constant, deceleration)
- **S-Curve**: Jerk-limited smooth profile

**Usage**:
```bash
python examples/planning/demo_velocity_profiles.py
python examples/planning/demo_velocity_profiles.py --plot
```

## Quick Start

Run all examples:
```bash
# Joint space planning
python examples/planning/demo_joint_space_planning.py --plot

# Cartesian space planning
python examples/planning/demo_cartesian_space_planning.py --plot

# Velocity profiles
python examples/planning/demo_velocity_profiles.py --plot
```

## Requirements

- numpy
- matplotlib (for visualization, optional)

## Notes

- All examples are standalone and don't require a robot model
- Use `--plot` flag to visualize trajectories
- Examples demonstrate basic usage; see documentation for advanced features

