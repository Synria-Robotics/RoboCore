"""Polynomial Interpolation for Joint Space Trajectory Planning

Copyright (c) 2025 Synria Robotics Co., Ltd.

Licensed under the MIT License.

Author: Synria Robotics Team
Website: https://synriarobotics.ai
"""

from __future__ import annotations

from typing import Any, Optional, Dict

import numpy as np

from robocore.planning.base import BaseTrajectoryPlanner


class CubicPolynomialPlanner(BaseTrajectoryPlanner):
    """Cubic polynomial trajectory planner for joint space.
    
    Generates C1 continuous trajectories (position and velocity continuous).
    Requires start and end positions and velocities.
    """
    
    def plan(
        self,
        start: Any,
        end: Any,
        duration: float,
        num_points: int,
        qd_start: Optional[Any] = None,
        qd_end: Optional[Any] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate cubic polynomial trajectory.
        
        :param start: Start joint positions [n]
        :param end: End joint positions [n]
        :param duration: Trajectory duration in seconds
        :param num_points: Number of points in trajectory
        :param qd_start: Start joint velocities [n], default zeros
        :param qd_end: End joint velocities [n], default zeros
        :return: Dictionary with 't', 'q', 'qd', 'qdd'
        """
        start = self._ensure_array(start)
        end = self._ensure_array(end)
        
        if start.ndim == 0:
            start = start.reshape(1)
        if end.ndim == 0:
            end = end.reshape(1)
        
        if len(start) != len(end):
            raise ValueError("Start and end must have same dimension")
        
        n_joints = len(start)
        
        # Default velocities
        if qd_start is None:
            qd_start = self._zeros(n_joints)
        else:
            qd_start = self._ensure_array(qd_start)
        
        if qd_end is None:
            qd_end = self._zeros(n_joints)
        else:
            qd_end = self._ensure_array(qd_end)
        
        # Time array
        t = self._linspace(0.0, duration, num_points)
        
        # Normalized time [0, 1]
        if self._backend_manager.is_numpy:
            s = t / duration
        else:
            s = t / duration
        
        # Compute polynomial coefficients for each joint
        # q(t) = a0 + a1*t + a2*t^2 + a3*t^3
        # qd(t) = a1 + 2*a2*t + 3*a3*t^2
        # qdd(t) = 2*a2 + 6*a3*t
        
        # Boundary conditions:
        # q(0) = start, q(T) = end
        # qd(0) = qd_start, qd(T) = qd_end
        
        # Solve for coefficients
        # a0 = start
        # a1 = qd_start
        # a2 = (3*(end - start) - (2*qd_start + qd_end)*T) / T^2
        # a3 = ((qd_start + qd_end)*T - 2*(end - start)) / T^3
        
        q = self._zeros((num_points, n_joints))
        qd = self._zeros((num_points, n_joints))
        qdd = self._zeros((num_points, n_joints))
        
        for j in range(n_joints):
            q0 = start[j]
            qf = end[j]
            qd0 = qd_start[j]
            qdf = qd_end[j]
            
            # Coefficients
            a0 = q0
            a1 = qd0
            a2 = (3.0 * (qf - q0) - (2.0 * qd0 + qdf) * duration) / (duration ** 2)
            a3 = ((qd0 + qdf) * duration - 2.0 * (qf - q0)) / (duration ** 3)
            
            # Evaluate polynomial
            if self._backend_manager.is_numpy:
                s_expanded = s.reshape(-1, 1) if s.ndim == 0 else s
            else:
                s_expanded = s.unsqueeze(-1) if s.ndim == 0 else s
            
            # Position: q(s*T) = a0 + a1*(s*T) + a2*(s*T)^2 + a3*(s*T)^3
            # Simplify: q(s) = a0 + a1*T*s + a2*T^2*s^2 + a3*T^3*s^3
            q[:, j] = a0 + a1 * duration * s + a2 * (duration ** 2) * (s ** 2) + a3 * (duration ** 3) * (s ** 3)
            
            # Velocity: qd(s*T) = a1*T + 2*a2*T^2*s + 3*a3*T^3*s^2
            qd[:, j] = a1 * duration + 2.0 * a2 * (duration ** 2) * s + 3.0 * a3 * (duration ** 3) * (s ** 2)
            
            # Acceleration: qdd(s*T) = 2*a2*T^2 + 6*a3*T^3*s
            qdd[:, j] = 2.0 * a2 * (duration ** 2) + 6.0 * a3 * (duration ** 3) * s
        
        return {
            't': t,
            'q': q,
            'qd': qd,
            'qdd': qdd,
        }


class QuinticPolynomialPlanner(BaseTrajectoryPlanner):
    """Quintic polynomial trajectory planner for joint space.
    
    Generates C2 continuous trajectories (position, velocity, and acceleration continuous).
    Requires start and end positions, velocities, and accelerations.
    """
    
    def plan(
        self,
        start: Any,
        end: Any,
        duration: float,
        num_points: int,
        qd_start: Optional[Any] = None,
        qd_end: Optional[Any] = None,
        qdd_start: Optional[Any] = None,
        qdd_end: Optional[Any] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate quintic polynomial trajectory.
        
        :param start: Start joint positions [n]
        :param end: End joint positions [n]
        :param duration: Trajectory duration in seconds
        :param num_points: Number of points in trajectory
        :param qd_start: Start joint velocities [n], default zeros
        :param qd_end: End joint velocities [n], default zeros
        :param qdd_start: Start joint accelerations [n], default zeros
        :param qdd_end: End joint accelerations [n], default zeros
        :return: Dictionary with 't', 'q', 'qd', 'qdd'
        """
        start = self._ensure_array(start)
        end = self._ensure_array(end)
        
        if start.ndim == 0:
            start = start.reshape(1)
        if end.ndim == 0:
            end = end.reshape(1)
        
        if len(start) != len(end):
            raise ValueError("Start and end must have same dimension")
        
        n_joints = len(start)
        
        # Default velocities and accelerations
        if qd_start is None:
            qd_start = self._zeros(n_joints)
        else:
            qd_start = self._ensure_array(qd_start)
        
        if qd_end is None:
            qd_end = self._zeros(n_joints)
        else:
            qd_end = self._ensure_array(qd_end)
        
        if qdd_start is None:
            qdd_start = self._zeros(n_joints)
        else:
            qdd_start = self._ensure_array(qdd_start)
        
        if qdd_end is None:
            qdd_end = self._zeros(n_joints)
        else:
            qdd_end = self._ensure_array(qdd_end)
        
        # Time array
        t = self._linspace(0.0, duration, num_points)
        
        # Normalized time [0, 1]
        s = t / duration
        
        # Compute polynomial coefficients for each joint
        # q(t) = a0 + a1*t + a2*t^2 + a3*t^3 + a4*t^4 + a5*t^5
        # qd(t) = a1 + 2*a2*t + 3*a3*t^2 + 4*a4*t^3 + 5*a5*t^4
        # qdd(t) = 2*a2 + 6*a3*t + 12*a4*t^2 + 20*a5*t^3
        
        # Boundary conditions:
        # q(0) = start, q(T) = end
        # qd(0) = qd_start, qd(T) = qd_end
        # qdd(0) = qdd_start, qdd(T) = qdd_end
        
        q = self._zeros((num_points, n_joints))
        qd = self._zeros((num_points, n_joints))
        qdd = self._zeros((num_points, n_joints))
        
        for j in range(n_joints):
            q0 = start[j]
            qf = end[j]
            qd0 = qd_start[j]
            qdf = qd_end[j]
            qdd0 = qdd_start[j]
            qddf = qdd_end[j]
            
            # Solve for coefficients using boundary conditions
            # Using matrix form: A * [a0, a1, a2, a3, a4, a5]^T = b
            T = duration
            T2 = T * T
            T3 = T2 * T
            T4 = T3 * T
            T5 = T4 * T
            
            if self._backend_manager.is_numpy:
                # Build coefficient matrix
                A = np.array([
                    [1, 0, 0, 0, 0, 0],  # q(0) = q0
                    [0, 1, 0, 0, 0, 0],  # qd(0) = qd0
                    [0, 0, 2, 0, 0, 0],  # qdd(0) = qdd0
                    [1, T, T2, T3, T4, T5],  # q(T) = qf
                    [0, 1, 2*T, 3*T2, 4*T3, 5*T4],  # qd(T) = qdf
                    [0, 0, 2, 6*T, 12*T2, 20*T3],  # qdd(T) = qddf
                ])
                
                b = np.array([q0, qd0, qdd0, qf, qdf, qddf])
                
                # Solve linear system
                coeffs = np.linalg.solve(A, b)
            else:
                import torch
                A = torch.tensor([
                    [1, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0],
                    [0, 0, 2, 0, 0, 0],
                    [1, T, T2, T3, T4, T5],
                    [0, 1, 2*T, 3*T2, 4*T3, 5*T4],
                    [0, 0, 2, 6*T, 12*T2, 20*T3],
                ], device=self._backend_manager.get_device(), dtype=self._backend_manager.get_dtype())
                
                b = torch.tensor([q0, qd0, qdd0, qf, qdf, qddf], 
                                device=self._backend_manager.get_device(), 
                                dtype=self._backend_manager.get_dtype())
                
                coeffs = torch.linalg.solve(A, b)
            
            a0, a1, a2, a3, a4, a5 = coeffs
            
            # Evaluate polynomial
            # Position: q(s*T) = sum(ai * (s*T)^i)
            # Simplify: q(s) = sum(ai * T^i * s^i)
            s2 = s * s
            s3 = s2 * s
            s4 = s3 * s
            s5 = s4 * s
            
            q[:, j] = (a0 + a1 * T * s + a2 * T2 * s2 + 
                      a3 * T3 * s3 + a4 * T4 * s4 + a5 * T5 * s5)
            
            # Velocity: qd(s*T) = sum(i * ai * T^i * s^(i-1))
            qd[:, j] = (a1 * T + 2 * a2 * T2 * s + 3 * a3 * T3 * s2 + 
                       4 * a4 * T4 * s3 + 5 * a5 * T5 * s4)
            
            # Acceleration: qdd(s*T) = sum(i*(i-1) * ai * T^i * s^(i-2))
            qdd[:, j] = (2 * a2 * T2 + 6 * a3 * T3 * s + 
                        12 * a4 * T4 * s2 + 20 * a5 * T5 * s3)
        
        return {
            't': t,
            'q': q,
            'qd': qd,
            'qdd': qdd,
        }


class SepticPolynomialPlanner(BaseTrajectoryPlanner):
    """Septic (7th order) polynomial trajectory planner for joint space.
    
    Generates C3 continuous trajectories (position, velocity, acceleration, and jerk continuous).
    Requires start and end positions, velocities, accelerations, and jerks.
    """
    
    def plan(
        self,
        start: Any,
        end: Any,
        duration: float,
        num_points: int,
        qd_start: Optional[Any] = None,
        qd_end: Optional[Any] = None,
        qdd_start: Optional[Any] = None,
        qdd_end: Optional[Any] = None,
        qddd_start: Optional[Any] = None,
        qddd_end: Optional[Any] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate septic polynomial trajectory.
        
        :param start: Start joint positions [n]
        :param end: End joint positions [n]
        :param duration: Trajectory duration in seconds
        :param num_points: Number of points in trajectory
        :param qd_start: Start joint velocities [n], default zeros
        :param qd_end: End joint velocities [n], default zeros
        :param qdd_start: Start joint accelerations [n], default zeros
        :param qdd_end: End joint accelerations [n], default zeros
        :param qddd_start: Start joint jerks [n], default zeros
        :param qddd_end: End joint jerks [n], default zeros
        :return: Dictionary with 't', 'q', 'qd', 'qdd', 'qddd'
        """
        start = self._ensure_array(start)
        end = self._ensure_array(end)
        
        if start.ndim == 0:
            start = start.reshape(1)
        if end.ndim == 0:
            end = end.reshape(1)
        
        if len(start) != len(end):
            raise ValueError("Start and end must have same dimension")
        
        n_joints = len(start)
        
        # Default values
        if qd_start is None:
            qd_start = self._zeros(n_joints)
        else:
            qd_start = self._ensure_array(qd_start)
        
        if qd_end is None:
            qd_end = self._zeros(n_joints)
        else:
            qd_end = self._ensure_array(qd_end)
        
        if qdd_start is None:
            qdd_start = self._zeros(n_joints)
        else:
            qdd_start = self._ensure_array(qdd_start)
        
        if qdd_end is None:
            qdd_end = self._zeros(n_joints)
        else:
            qdd_end = self._ensure_array(qdd_end)
        
        if qddd_start is None:
            qddd_start = self._zeros(n_joints)
        else:
            qddd_start = self._ensure_array(qddd_start)
        
        if qddd_end is None:
            qddd_end = self._zeros(n_joints)
        else:
            qddd_end = self._ensure_array(qddd_end)
        
        # Time array
        t = self._linspace(0.0, duration, num_points)
        s = t / duration
        
        # Compute polynomial coefficients
        # q(t) = a0 + a1*t + a2*t^2 + a3*t^3 + a4*t^4 + a5*t^5 + a6*t^6 + a7*t^7
        # qd(t) = a1 + 2*a2*t + 3*a3*t^2 + 4*a4*t^3 + 5*a5*t^4 + 6*a6*t^5 + 7*a7*t^6
        # qdd(t) = 2*a2 + 6*a3*t + 12*a4*t^2 + 20*a5*t^3 + 30*a6*t^4 + 42*a7*t^5
        # qddd(t) = 6*a3 + 24*a4*t + 60*a5*t^2 + 120*a6*t^3 + 210*a7*t^4
        
        q = self._zeros((num_points, n_joints))
        qd = self._zeros((num_points, n_joints))
        qdd = self._zeros((num_points, n_joints))
        qddd = self._zeros((num_points, n_joints))
        
        for j in range(n_joints):
            q0 = start[j]
            qf = end[j]
            qd0 = qd_start[j]
            qdf = qd_end[j]
            qdd0 = qdd_start[j]
            qddf = qdd_end[j]
            qddd0 = qddd_start[j]
            qdddf = qddd_end[j]
            
            T = duration
            T2 = T * T
            T3 = T2 * T
            T4 = T3 * T
            T5 = T4 * T
            T6 = T5 * T
            T7 = T6 * T
            
            # Build coefficient matrix for 8 boundary conditions
            if self._backend_manager.is_numpy:
                A = np.array([
                    [1, 0, 0, 0, 0, 0, 0, 0],  # q(0) = q0
                    [0, 1, 0, 0, 0, 0, 0, 0],  # qd(0) = qd0
                    [0, 0, 2, 0, 0, 0, 0, 0],  # qdd(0) = qdd0
                    [0, 0, 0, 6, 0, 0, 0, 0],  # qddd(0) = qddd0
                    [1, T, T2, T3, T4, T5, T6, T7],  # q(T) = qf
                    [0, 1, 2*T, 3*T2, 4*T3, 5*T4, 6*T5, 7*T6],  # qd(T) = qdf
                    [0, 0, 2, 6*T, 12*T2, 20*T3, 30*T4, 42*T5],  # qdd(T) = qddf
                    [0, 0, 0, 6, 24*T, 60*T2, 120*T3, 210*T4],  # qddd(T) = qdddf
                ])
                
                b = np.array([q0, qd0, qdd0, qddd0, qf, qdf, qddf, qdddf])
                coeffs = np.linalg.solve(A, b)
            else:
                import torch
                A = torch.tensor([
                    [1, 0, 0, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0, 0, 0],
                    [0, 0, 2, 0, 0, 0, 0, 0],
                    [0, 0, 0, 6, 0, 0, 0, 0],
                    [1, T, T2, T3, T4, T5, T6, T7],
                    [0, 1, 2*T, 3*T2, 4*T3, 5*T4, 6*T5, 7*T6],
                    [0, 0, 2, 6*T, 12*T2, 20*T3, 30*T4, 42*T5],
                    [0, 0, 0, 6, 24*T, 60*T2, 120*T3, 210*T4],
                ], device=self._backend_manager.get_device(), dtype=self._backend_manager.get_dtype())
                
                b = torch.tensor([q0, qd0, qdd0, qddd0, qf, qdf, qddf, qdddf],
                                device=self._backend_manager.get_device(),
                                dtype=self._backend_manager.get_dtype())
                coeffs = torch.linalg.solve(A, b)
            
            a0, a1, a2, a3, a4, a5, a6, a7 = coeffs
            
            # Evaluate polynomial
            s2 = s * s
            s3 = s2 * s
            s4 = s3 * s
            s5 = s4 * s
            s6 = s5 * s
            s7 = s6 * s
            
            q[:, j] = (a0 + a1 * T * s + a2 * T2 * s2 + a3 * T3 * s3 +
                      a4 * T4 * s4 + a5 * T5 * s5 + a6 * T6 * s6 + a7 * T7 * s7)
            
            qd[:, j] = (a1 * T + 2 * a2 * T2 * s + 3 * a3 * T3 * s2 +
                       4 * a4 * T4 * s3 + 5 * a5 * T5 * s4 + 6 * a6 * T6 * s5 + 7 * a7 * T7 * s6)
            
            qdd[:, j] = (2 * a2 * T2 + 6 * a3 * T3 * s + 12 * a4 * T4 * s2 +
                         20 * a5 * T5 * s3 + 30 * a6 * T6 * s4 + 42 * a7 * T7 * s5)
            
            qddd[:, j] = (6 * a3 * T3 + 24 * a4 * T4 * s + 60 * a5 * T5 * s2 +
                         120 * a6 * T6 * s3 + 210 * a7 * T7 * s4)
        
        return {
            't': t,
            'q': q,
            'qd': qd,
            'qdd': qdd,
            'qddd': qddd,
        }

