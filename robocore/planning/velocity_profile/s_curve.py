"""S-Curve Velocity Profile

Copyright (c) 2025 Synria Robotics Co., Ltd.

Licensed under the MIT License.

Author: Synria Robotics Team
Website: https://synriarobotics.ai
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from robocore.planning.base import BaseTrajectoryPlanner


class SCurveVelocityProfile(BaseTrajectoryPlanner):
    """S-curve (jerk-limited) velocity profile generator.
    
    Generates smooth velocity profiles with limited jerk (rate of change of acceleration).
    """
    
    def plan(
        self,
        start: float,
        end: float,
        duration: Optional[float] = None,
        num_points: int = 100,
        v_max: Optional[float] = None,
        a_max: Optional[float] = None,
        j_max: Optional[float] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate S-curve velocity profile.
        
        :param start: Start position (scalar)
        :param end: End position (scalar)
        :param duration: Trajectory duration in seconds (optional)
        :param num_points: Number of points in trajectory
        :param v_max: Maximum velocity (optional)
        :param a_max: Maximum acceleration (optional)
        :param j_max: Maximum jerk (optional, default: 10 * a_max / duration if duration provided)
        :return: Dictionary with 't', 's', 'v', 'a', 'j'
        """
        start = float(start)
        end = float(end)
        distance = abs(end - start)
        direction = 1.0 if end >= start else -1.0
        
        # Default jerk
        if j_max is None:
            if duration is not None and a_max is not None:
                j_max = 10.0 * a_max / duration
            else:
                j_max = 10.0  # Default value
        
        # For simplicity, use a symmetric S-curve profile
        # Phase 1: Jerk up (0 to a_max)
        # Phase 2: Constant acceleration
        # Phase 3: Jerk down (a_max to 0)
        # Phase 4: Constant velocity
        # Phase 5: Jerk down (0 to -a_max)
        # Phase 6: Constant deceleration
        # Phase 7: Jerk up (-a_max to 0)
        
        # Compute time parameters
        t_jerk = a_max / j_max if a_max is not None else 0.1
        s_jerk = (j_max * t_jerk ** 3) / 6.0  # Distance during jerk phase
        
        if v_max is None:
            if duration is not None and a_max is not None:
                # Estimate v_max from duration
                # Simplified: assume symmetric profile
                v_max = distance / duration if duration > 0 else 1.0
            else:
                v_max = 1.0
        
        if a_max is None:
            if duration is not None:
                a_max = v_max / (duration / 3.0)  # Rough estimate
            else:
                a_max = 1.0
        
        # Compute total time (simplified symmetric profile)
        # Distance = 2 * (jerk phase distance) + constant accel distance + constant vel distance
        dist_jerk_total = 2 * s_jerk
        dist_accel = v_max * t_jerk  # Simplified
        dist_const_vel = distance - dist_jerk_total - 2 * dist_accel
        
        if dist_const_vel < 0:
            # No constant velocity phase
            dist_const_vel = 0.0
            t_const_vel = 0.0
            # Adjust v_max
            v_max = ((distance - dist_jerk_total) * j_max / a_max) ** 0.5
        else:
            t_const_vel = dist_const_vel / v_max if v_max > 0 else 0.0
        
        t_accel = (v_max - 0.5 * a_max * t_jerk) / a_max if a_max > 0 else 0.0
        t_total = 4 * t_jerk + 2 * t_accel + t_const_vel
        
        if duration is not None:
            # Scale time to match duration
            scale = duration / t_total if t_total > 0 else 1.0
            t_jerk *= scale
            t_accel *= scale
            t_const_vel *= scale
            t_total = duration
        
        # Time array
        t = self._linspace(0.0, t_total, num_points)
        
        # Generate profile
        s = self._zeros(num_points)
        v = self._zeros(num_points)
        a = self._zeros(num_points)
        j = self._zeros(num_points)
        
        for i in range(num_points):
            t_i = t[i] if self._backend_manager.is_numpy else float(t[i])
            
            if t_i <= t_jerk:
                # Phase 1: Jerk up
                j[i] = j_max
                a[i] = j_max * t_i
                v[i] = 0.5 * j_max * t_i ** 2
                s[i] = (j_max * t_i ** 3) / 6.0
            elif t_i <= t_jerk + t_accel:
                # Phase 2: Constant acceleration
                t_phase = t_i - t_jerk
                j[i] = 0.0
                a[i] = a_max
                v_jerk = 0.5 * j_max * t_jerk ** 2
                v[i] = v_jerk + a_max * t_phase
                s_jerk = (j_max * t_jerk ** 3) / 6.0
                s[i] = s_jerk + v_jerk * t_phase + 0.5 * a_max * t_phase ** 2
            elif t_i <= 2 * t_jerk + t_accel:
                # Phase 3: Jerk down (a_max to 0)
                t_phase = t_i - t_jerk - t_accel
                j[i] = -j_max
                a[i] = a_max - j_max * t_phase
                v_jerk = 0.5 * j_max * t_jerk ** 2
                v_accel = v_jerk + a_max * t_accel
                v[i] = v_accel + a_max * t_phase - 0.5 * j_max * t_phase ** 2
                s_jerk = (j_max * t_jerk ** 3) / 6.0
                s_accel = s_jerk + v_jerk * t_accel + 0.5 * a_max * t_accel ** 2
                s[i] = s_accel + v_accel * t_phase + 0.5 * a_max * t_phase ** 2 - (j_max * t_phase ** 3) / 6.0
            elif t_i <= 2 * t_jerk + t_accel + t_const_vel:
                # Phase 4: Constant velocity
                t_phase = t_i - 2 * t_jerk - t_accel
                j[i] = 0.0
                a[i] = 0.0
                v_jerk = 0.5 * j_max * t_jerk ** 2
                v_accel = v_jerk + a_max * t_accel
                v_const = v_accel + a_max * t_jerk - 0.5 * j_max * t_jerk ** 2
                v[i] = v_const
                # Compute s at end of phase 3
                s_jerk = (j_max * t_jerk ** 3) / 6.0
                s_accel = s_jerk + v_jerk * t_accel + 0.5 * a_max * t_accel ** 2
                s_phase3 = s_accel + v_accel * t_jerk + 0.5 * a_max * t_jerk ** 2 - (j_max * t_jerk ** 3) / 6.0
                s[i] = s_phase3 + v_const * t_phase
            elif t_i <= 3 * t_jerk + t_accel + t_const_vel:
                # Phase 5: Jerk down (0 to -a_max)
                t_phase = t_i - 2 * t_jerk - t_accel - t_const_vel
                j[i] = -j_max
                a[i] = -j_max * t_phase
                v_const = v_max  # Use v_max from phase 4
                v[i] = v_const - 0.5 * j_max * t_phase ** 2
                s_phase4 = s_jerk + v_jerk * t_accel + 0.5 * a_max * t_accel ** 2
                s_phase4 += v_accel * t_jerk + 0.5 * a_max * t_jerk ** 2 - (j_max * t_jerk ** 3) / 6.0
                s_phase4 += v_const * t_const_vel
                s[i] = s_phase4 + v_const * t_phase - (j_max * t_phase ** 3) / 6.0
            elif t_i <= 3 * t_jerk + 2 * t_accel + t_const_vel:
                # Phase 6: Constant deceleration
                t_phase = t_i - 3 * t_jerk - t_accel - t_const_vel
                j[i] = 0.0
                a[i] = -a_max
                v_const = v_max
                v_phase5 = v_const - 0.5 * j_max * t_jerk ** 2
                v[i] = v_phase5 - a_max * t_phase
                # Compute s at end of phase 5
                s_phase5 = s_phase4 + v_const * t_jerk - (j_max * t_jerk ** 3) / 6.0
                s[i] = s_phase5 + v_phase5 * t_phase - 0.5 * a_max * t_phase ** 2
            else:
                # Phase 7: Jerk up (-a_max to 0)
                t_phase = t_i - 3 * t_jerk - 2 * t_accel - t_const_vel
                j[i] = j_max
                a[i] = -a_max + j_max * t_phase
                v_const = v_max
                v_phase5 = v_const - 0.5 * j_max * t_jerk ** 2
                v_phase6 = v_phase5 - a_max * t_accel
                v[i] = v_phase6 - a_max * t_phase + 0.5 * j_max * t_phase ** 2
                # Compute s at end of phase 6
                s_phase6 = s_phase5 + v_phase5 * t_accel - 0.5 * a_max * t_accel ** 2
                s[i] = s_phase6 + v_phase6 * t_phase - 0.5 * a_max * t_phase ** 2 + (j_max * t_phase ** 3) / 6.0
        
        # Scale by direction and offset by start
        s = start + direction * s
        
        return {
            't': t,
            's': s,
            'v': v,
            'a': a,
            'j': j,
        }

