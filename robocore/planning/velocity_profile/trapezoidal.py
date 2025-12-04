"""Trapezoidal Velocity Profile

Copyright (c) 2025 Synria Robotics Co., Ltd.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see <https://www.gnu.org/licenses/>.

Author: Synria Robotics Team
Website: https://synriarobotics.ai
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from robocore.planning.base import BaseTrajectoryPlanner


class TrapezoidalVelocityProfile(BaseTrajectoryPlanner):
    """Trapezoidal velocity profile generator.
    
    Generates position, velocity, and acceleration profiles with
    constant acceleration phases (acceleration, constant velocity, deceleration).
    """
    
    def plan(
        self,
        start: float,
        end: float,
        duration: Optional[float] = None,
        num_points: int = 100,
        v_max: Optional[float] = None,
        a_max: Optional[float] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate trapezoidal velocity profile.
        
        :param start: Start position (scalar)
        :param end: End position (scalar)
        :param duration: Trajectory duration in seconds (optional if v_max and a_max provided)
        :param num_points: Number of points in trajectory
        :param v_max: Maximum velocity (optional)
        :param a_max: Maximum acceleration (optional)
        :return: Dictionary with 't', 's', 'v', 'a'
        """
        start = float(start)
        end = float(end)
        distance = abs(end - start)
        direction = 1.0 if end >= start else -1.0
        
        # Determine time parameters
        if duration is None:
            if v_max is None or a_max is None:
                raise ValueError("Either duration or (v_max and a_max) must be provided")
            
            # Compute minimum time
            # If distance is small, triangular profile (no constant velocity phase)
            t_accel = v_max / a_max
            dist_accel = 0.5 * a_max * t_accel ** 2
            
            if distance <= 2 * dist_accel:
                # Triangular profile
                t_accel = (distance / a_max) ** 0.5
                t_total = 2 * t_accel
                t_const = 0.0
                v_max = a_max * t_accel
            else:
                # Trapezoidal profile
                t_const = (distance - 2 * dist_accel) / v_max
                t_total = 2 * t_accel + t_const
        else:
            # Duration is given, compute v_max and a_max if needed
            if v_max is None and a_max is None:
                # Use default: equal acceleration and deceleration phases
                t_accel = duration / 3.0
                t_const = duration / 3.0
                a_max = distance / (t_accel ** 2)
                v_max = a_max * t_accel
                t_total = duration
            elif v_max is None:
                # Compute v_max from a_max
                # distance = v_max^2 / a_max + v_max * (T - 2*v_max/a_max)
                # Solve quadratic: a_max * T * v_max - v_max^2 = distance * a_max
                # v_max^2 - a_max * T * v_max + distance * a_max = 0
                discriminant = (a_max * duration) ** 2 - 4 * distance * a_max
                if discriminant < 0:
                    # Use triangular profile
                    t_accel = (distance / a_max) ** 0.5
                    t_total = 2 * t_accel
                    t_const = 0.0
                    v_max = a_max * t_accel
                else:
                    v_max = (a_max * duration - discriminant ** 0.5) / 2.0
                    t_accel = v_max / a_max
                    t_const = duration - 2 * t_accel
                    if t_const < 0:
                        # Triangular profile
                        t_accel = (distance / a_max) ** 0.5
                        t_total = 2 * t_accel
                        t_const = 0.0
                        v_max = a_max * t_accel
                    else:
                        t_total = duration
            elif a_max is None:
                # Compute a_max from v_max
                # Try trapezoidal first
                t_accel = v_max / 1.0  # Initial guess
                dist_accel = 0.5 * 1.0 * t_accel ** 2
                if distance <= 2 * dist_accel:
                    # Triangular profile
                    a_max = distance / ((duration / 2) ** 2)
                    t_accel = v_max / a_max
                    t_total = 2 * t_accel
                    t_const = 0.0
                else:
                    # Solve for a_max: distance = v_max^2/a_max + v_max*(T - 2*v_max/a_max)
                    # distance = v_max^2/a_max + v_max*T - 2*v_max^2/a_max
                    # distance = v_max*T - v_max^2/a_max
                    # a_max = v_max^2 / (v_max*T - distance)
                    if v_max * duration > distance:
                        a_max = (v_max ** 2) / (v_max * duration - distance)
                        t_accel = v_max / a_max
                        t_const = duration - 2 * t_accel
                        t_total = duration
                    else:
                        # Use triangular
                        a_max = distance / ((duration / 2) ** 2)
                        t_accel = v_max / a_max if a_max > 0 else duration / 2
                        t_total = 2 * t_accel
                        t_const = 0.0
            else:
                # Both v_max and a_max given, compute times
                t_accel = v_max / a_max
                dist_accel = 0.5 * a_max * t_accel ** 2
                if distance <= 2 * dist_accel:
                    # Triangular profile
                    t_accel = (distance / a_max) ** 0.5
                    t_total = 2 * t_accel
                    t_const = 0.0
                else:
                    t_const = (distance - 2 * dist_accel) / v_max
                    t_total = 2 * t_accel + t_const
                
                # Scale to match duration if provided
                if duration is not None and abs(t_total - duration) > 1e-6:
                    scale = duration / t_total
                    t_accel *= scale
                    t_const *= scale
                    t_total = duration
        
        # Time array
        t = self._linspace(0.0, t_total, num_points)
        
        # Generate profile
        s = self._zeros(num_points)
        v = self._zeros(num_points)
        a = self._zeros(num_points)
        
        for i in range(num_points):
            t_i = t[i] if self._backend_manager.is_numpy else float(t[i])
            
            if t_i <= t_accel:
                # Acceleration phase
                s[i] = 0.5 * a_max * t_i ** 2
                v[i] = a_max * t_i
                a[i] = a_max
            elif t_i <= t_accel + t_const:
                # Constant velocity phase
                s_accel = 0.5 * a_max * t_accel ** 2
                v_const = a_max * t_accel
                s[i] = s_accel + v_const * (t_i - t_accel)
                v[i] = v_const
                a[i] = 0.0
            else:
                # Deceleration phase
                s_accel = 0.5 * a_max * t_accel ** 2
                v_const = a_max * t_accel
                s_const = v_const * t_const if t_const > 0 else 0.0
                t_dec = t_i - t_accel - t_const
                s[i] = s_accel + s_const + v_const * t_dec - 0.5 * a_max * t_dec ** 2
                v[i] = v_const - a_max * t_dec
                a[i] = -a_max
        
        # Scale by direction and offset by start
        s = start + direction * s
        
        return {
            't': t,
            's': s,
            'v': v,
            'a': a,
        }

