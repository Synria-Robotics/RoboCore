"""Multi-chain inverse kinematics solver.

Unified solver supporting weighted least squares and hierarchical nullspace projection
for arbitrary number of kinematic chains.
"""
from __future__ import annotations
from typing import Dict, List, Any, Sequence, Optional
import numpy as np

from robocore.modeling.robot_model import RobotModel
from robocore.kinematics.task import Task
from robocore.kinematics.jacobian import jacobian as single_jacobian
from robocore.utils.backend import get_backend


class MultiChainIKSolver:
    """Inverse kinematics solver for multi-chain systems.
    
    :param groups: Dictionary mapping group names to RobotModel instances
    """
    
    def __init__(self, groups: Dict[str, RobotModel]):
        self.groups = groups
        self.group_names = list(groups.keys())
    
    def solve_weighted(self,
                      tasks: List[Task],
                      q0_by_group: Dict[str, Sequence[float]],
                      *,
                      max_iters: int = 100,
                      tol: float = 1e-3,
                      damping: float = 1e-3,
                      step_limit: float = 0.2,
                      backend: str = 'numpy',
                      verbose: bool = False) -> Dict[str, Any]:
        """Solve multi-chain IK using weighted least squares.
        
        Minimizes weighted sum: Σ w_i ||e_i||^2 using damped least squares (DLS).
        
        :param tasks: List of Task objects
        :param q0_by_group: Initial joint configuration per group
        :param max_iters: Maximum iterations
        :param tol: Convergence tolerance
        :param damping: DLS damping factor
        :param step_limit: Maximum joint step per iteration (rad)
        :param backend: Computation backend ('numpy' or 'torch')
        :param verbose: Print iteration info
        :return: Solution dictionary with q_by_group, success, iters, residual
        """
        # Build concatenated joint vector
        q_by = {k: np.array(q0_by_group[k], dtype=float) for k in self.group_names}
        n_by = {k: len(q_by[k]) for k in self.group_names}
        
        # Index mapping
        idx_by = {}
        offset = 0
        for k in self.group_names:
            idx_by[k] = (offset, offset + n_by[k])
            offset += n_by[k]
        n_total = offset
        
        q = np.concatenate([q_by[k] for k in self.group_names])
        
        def slice_group(vec, name):
            i0, i1 = idx_by[name]
            return vec[i0:i1]
        
        def assign_group(vec, name, part):
            i0, i1 = idx_by[name]
            vec[i0:i1] = part
        
        for it in range(max_iters):
            errs = []
            Jrows = []
            
            for task in tasks:
                if task.type == 'absolute':
                    g = task.group
                    model = self.groups[g]
                    qg = slice_group(q, g)
                    T_cur = model.fk(qg, backend=backend, return_end=True)
                    T_cur = T_cur.detach().cpu().numpy() if hasattr(T_cur, 'detach') else np.array(T_cur)
                    
                    e = self._pose_error_np(T_cur, np.array(task.target))
                    Jg = single_jacobian(model, qg, backend=backend)
                    Jg = Jg.detach().cpu().numpy() if hasattr(Jg, 'detach') else np.array(Jg)
                    
                    # Pad into full vector
                    Jpad = np.zeros((6, n_total))
                    i0, i1 = idx_by[g]
                    Jpad[:, i0:i1] = Jg
                    
                    # Row mask
                    if task.row_mask is not None:
                        m = np.array([bool(x) for x in task.row_mask])
                        e = e[m]
                        Jpad = Jpad[m, :]
                    
                    w = float(task.weight) ** 0.5
                    errs.append(w * e)
                    Jrows.append(w * Jpad)
                
                elif task.type == 'relative':
                    ga = task.group_a
                    gb = task.group_b
                    ma = self.groups[ga]
                    mb = self.groups[gb]
                    qa = slice_group(q, ga)
                    qb = slice_group(q, gb)
                    
                    Ta = ma.fk(qa, backend=backend, return_end=True)
                    Tb = mb.fk(qb, backend=backend, return_end=True)
                    
                    if hasattr(Ta, 'detach'):
                        Ta = Ta.detach().cpu().numpy()
                        Tb = Tb.detach().cpu().numpy()
                    Ta = np.array(Ta)
                    Tb = np.array(Tb)
                    
                    e = self._relative_pose_error_np(Ta, Tb, np.array(task.target))
                    # Use analytic relative Jacobian for speed
                    Jr = self._relative_jacobian_analytic(ma, mb, qa, qb, backend=backend)
                    Jr = Jr.detach().cpu().numpy() if hasattr(Jr, 'detach') else np.array(Jr)
                    
                    # Place Jr into columns of a+b
                    Jpad = np.zeros((Jr.shape[0], n_total))
                    i0a, i1a = idx_by[ga]
                    i0b, i1b = idx_by[gb]
                    Jpad[:, i0a:i1a] = Jr[:, :n_by[ga]]
                    Jpad[:, i0b:i1b] = Jr[:, n_by[ga]:]
                    
                    if task.row_mask is not None:
                        m = np.array([bool(x) for x in task.row_mask])
                        e = e[m]
                        Jpad = Jpad[m, :]
                    
                    w = float(task.weight) ** 0.5
                    errs.append(w * e)
                    Jrows.append(w * Jpad)
                
                else:
                    raise ValueError(f"Unsupported task type: {task.type}")
            
            e_total = np.concatenate(errs) if errs else np.zeros(0)
            if e_total.size == 0:
                break
            
            J_total = np.vstack(Jrows)
            JT = J_total.T
            
            # DLS solve
            A = J_total @ JT + (damping ** 2) * np.eye(J_total.shape[0])
            dq = JT @ np.linalg.solve(A, e_total)
            
            # Step limit
            dq_norm = np.linalg.norm(dq)
            if dq_norm > step_limit:
                dq = dq * (step_limit / dq_norm)
            
            q += dq
            
            if verbose:
                print(f"[Iter {it+1}] residual={np.linalg.norm(e_total):.6f}, step={dq_norm:.6f}")
            
            if np.linalg.norm(e_total) < tol:
                break
        
        # Extract results
        q_out = {}
        for name in self.group_names:
            q_out[name] = slice_group(q, name).tolist()
        
        return {
            'q_by_group': q_out,
            'success': True,
            'iters': it + 1,
            'residual': float(np.linalg.norm(e_total)) if e_total.size else 0.0,
        }
    
    def solve_hierarchical(self,
                          task_groups: List[List[Task]],
                          q0_by_group: Dict[str, Sequence[float]],
                          *,
                          max_iters: int = 100,
                          tol: float = 1e-3,
                          damping: float = 1e-4,
                          step_limit: float = 0.15,
                          backend: str = 'numpy',
                          verbose: bool = False) -> Dict[str, Any]:
        """Solve multi-chain IK using hierarchical nullspace projection.
        
        Higher priority tasks are solved exactly; lower priority tasks project into nullspace.
        
        :param task_groups: List of task lists, ordered by priority (highest first)
        :param q0_by_group: Initial joint configuration per group
        :param max_iters: Maximum iterations
        :param tol: Convergence tolerance
        :param damping: Pseudoinverse damping
        :param step_limit: Maximum joint step per iteration
        :param backend: Computation backend
        :param verbose: Print iteration info
        :return: Solution dictionary
        """
        # Build concatenated joint vector
        q_by = {k: np.array(q0_by_group[k], dtype=float) for k in self.group_names}
        n_by = {k: len(q_by[k]) for k in self.group_names}
        
        idx_by = {}
        offset = 0
        for k in self.group_names:
            idx_by[k] = (offset, offset + n_by[k])
            offset += n_by[k]
        n_total = offset
        
        q = np.concatenate([q_by[k] for k in self.group_names])
        
        def slice_group(vec, name):
            i0, i1 = idx_by[name]
            return vec[i0:i1]
        
        for it in range(max_iters):
            N_compound = np.eye(n_total)  # Compound nullspace projector
            total_step = np.zeros(n_total)
            max_error = 0.0
            
            for priority, tasks in enumerate(task_groups):
                errs = []
                Jrows = []
                
                for task in tasks:
                    if task.type == 'absolute':
                        g = task.group
                        model = self.groups[g]
                        qg = slice_group(q, g)
                        T_cur = model.fk(qg, backend=backend, return_end=True)
                        T_cur = T_cur.detach().cpu().numpy() if hasattr(T_cur, 'detach') else np.array(T_cur)
                        
                        e = self._pose_error_np(T_cur, np.array(task.target))
                        Jg = single_jacobian(model, qg, backend=backend)
                        Jg = Jg.detach().cpu().numpy() if hasattr(Jg, 'detach') else np.array(Jg)
                        
                        Jpad = np.zeros((6, n_total))
                        i0, i1 = idx_by[g]
                        Jpad[:, i0:i1] = Jg
                        
                        if task.row_mask is not None:
                            m = np.array([bool(x) for x in task.row_mask])
                            e = e[m]
                            Jpad = Jpad[m, :]
                        
                        errs.append(e)
                        Jrows.append(Jpad)
                    
                    elif task.type == 'relative':
                        ga = task.group_a
                        gb = task.group_b
                        ma = self.groups[ga]
                        mb = self.groups[gb]
                        qa = slice_group(q, ga)
                        qb = slice_group(q, gb)
                        
                        Ta = ma.fk(qa, backend=backend, return_end=True)
                        Tb = mb.fk(qb, backend=backend, return_end=True)
                        
                        if hasattr(Ta, 'detach'):
                            Ta = Ta.detach().cpu().numpy()
                            Tb = Tb.detach().cpu().numpy()
                        Ta = np.array(Ta)
                        Tb = np.array(Tb)
                        
                        e = self._relative_pose_error_np(Ta, Tb, np.array(task.target))
                        # Use analytic relative Jacobian for speed
                        Jr = self._relative_jacobian_analytic(ma, mb, qa, qb, backend=backend)
                        Jr = Jr.detach().cpu().numpy() if hasattr(Jr, 'detach') else np.array(Jr)
                        
                        Jpad = np.zeros((Jr.shape[0], n_total))
                        i0a, i1a = idx_by[ga]
                        i0b, i1b = idx_by[gb]
                        Jpad[:, i0a:i1a] = Jr[:, :n_by[ga]]
                        Jpad[:, i0b:i1b] = Jr[:, n_by[ga]:]
                        
                        if task.row_mask is not None:
                            m = np.array([bool(x) for x in task.row_mask])
                            e = e[m]
                            Jpad = Jpad[m, :]
                        
                        errs.append(e)
                        Jrows.append(Jpad)
                
                if not errs:
                    continue
                
                e_level = np.concatenate(errs)
                J_level = np.vstack(Jrows)
                
                # Project into compound nullspace
                J_proj = J_level @ N_compound
                
                # Damped pseudoinverse
                JJT = J_proj @ J_proj.T + (damping ** 2) * np.eye(J_proj.shape[0])
                dq_level = N_compound @ J_proj.T @ np.linalg.solve(JJT, e_level)
                
                total_step += dq_level
                
                # Update nullspace projector
                J_pinv = J_proj.T @ np.linalg.solve(JJT, np.eye(J_proj.shape[0]))
                N_level = np.eye(n_total) - J_pinv @ J_proj
                N_compound = N_compound @ N_level
                
                max_error = max(max_error, np.linalg.norm(e_level))
            
            # Step limit
            step_norm = np.linalg.norm(total_step)
            if step_norm > step_limit:
                total_step = total_step * (step_limit / step_norm)
            
            q += total_step
            
            if verbose:
                print(f"[Hier Iter {it+1}] max_error={max_error:.6f}, step={step_norm:.6f}")
            
            if max_error < tol:
                break
        
        q_out = {}
        for name in self.group_names:
            q_out[name] = slice_group(q, name).tolist()
        
        return {
            'q_by_group': q_out,
            'success': True,
            'iters': it + 1,
            'residual': max_error,
        }
    
    @staticmethod
    def _pose_error_np(T_current: np.ndarray, T_target: np.ndarray) -> np.ndarray:
        """Compute 6D pose error (position + axis-angle orientation).
        
        :param T_current: Current 4x4 transformation
        :param T_target: Target 4x4 transformation
        :return: 6D error vector [dx, dy, dz, wx, wy, wz]
        """
        e_pos = T_target[0:3, 3] - T_current[0:3, 3]
        
        R_cur = T_current[0:3, 0:3]
        R_tgt = T_target[0:3, 0:3]
        R_err = R_tgt @ R_cur.T
        
        # Axis-angle from rotation matrix
        trace = np.trace(R_err)
        theta = np.arccos(np.clip((trace - 1) / 2, -1, 1))
        
        if theta < 1e-6:
            e_ori = np.zeros(3)
        else:
            axis = np.array([
                R_err[2, 1] - R_err[1, 2],
                R_err[0, 2] - R_err[2, 0],
                R_err[1, 0] - R_err[0, 1]
            ]) / (2 * np.sin(theta))
            e_ori = theta * axis
        
        return np.concatenate([e_pos, e_ori])
    
    @staticmethod
    def _relative_pose_error_np(T_a: np.ndarray, T_b: np.ndarray, 
                                T_rel_desired: np.ndarray) -> np.ndarray:
        """Compute relative pose error in frame A.
        
        :param T_a: First chain end-effector pose
        :param T_b: Second chain end-effector pose
        :param T_rel_desired: Desired relative transform T_a^{-1} @ T_b
        :return: 6D error vector in frame A
        """
        T_rel_current = np.linalg.inv(T_a) @ T_b
        
        p_rel_cur = T_rel_current[0:3, 3]
        p_rel_des = T_rel_desired[0:3, 3]
        e_pos = p_rel_des - p_rel_cur
        
        R_rel_cur = T_rel_current[0:3, 0:3]
        R_rel_des = T_rel_desired[0:3, 0:3]
        R_err = R_rel_des @ R_rel_cur.T
        
        trace = np.trace(R_err)
        theta = np.arccos(np.clip((trace - 1) / 2, -1, 1))
        
        if theta < 1e-6:
            e_ori = np.zeros(3)
        else:
            axis = np.array([
                R_err[2, 1] - R_err[1, 2],
                R_err[0, 2] - R_err[2, 0],
                R_err[1, 0] - R_err[0, 1]
            ]) / (2 * np.sin(theta))
            e_ori = theta * axis
        
        return np.concatenate([e_pos, e_ori])
    
    @staticmethod
    @staticmethod
    def _relative_jacobian_analytic(model_a: RobotModel, model_b: RobotModel,
                                    q_a: np.ndarray, q_b: np.ndarray,
                                    backend: str = 'numpy') -> np.ndarray:
        """Compute relative Jacobian using analytical method.
        
        For relative pose T_rel = inv(T_a) @ T_b, the Jacobian is:
        J_rel = [-Ad(T_rel) @ J_a, J_b]
        
        where Ad is the adjoint transformation.
        
        :param model_a: First chain model
        :param model_b: Second chain model
        :param q_a: First chain joint configuration
        :param q_b: Second chain joint configuration
        :param backend: Computation backend
        :return: 6 × (n_a + n_b) Jacobian matrix
        """
        # Compute individual Jacobians using analytic method
        J_a = single_jacobian(model_a, q_a, backend=backend, method='analytic')
        J_b = single_jacobian(model_b, q_b, backend=backend, method='analytic')
        
        # Convert to numpy if needed
        if hasattr(J_a, 'detach'):
            J_a = J_a.detach().cpu().numpy()
            J_b = J_b.detach().cpu().numpy()
        J_a = np.array(J_a)
        J_b = np.array(J_b)
        
        # Compute forward kinematics
        T_a = model_a.fk(q_a, backend=backend, return_end=True)
        T_b = model_b.fk(q_b, backend=backend, return_end=True)
        
        if hasattr(T_a, 'detach'):
            T_a = T_a.detach().cpu().numpy()
            T_b = T_b.detach().cpu().numpy()
        T_a = np.array(T_a)
        T_b = np.array(T_b)
        
        # Relative transform: T_rel = inv(T_a) @ T_b
        T_rel = np.linalg.inv(T_a) @ T_b
        
        # Compute adjoint: Ad(T_rel) transforms velocities from frame a to frame b
        R_rel = T_rel[0:3, 0:3]
        p_rel = T_rel[0:3, 3]
        
        # Skew-symmetric matrix
        p_skew = np.array([
            [0, -p_rel[2], p_rel[1]],
            [p_rel[2], 0, -p_rel[0]],
            [-p_rel[1], p_rel[0], 0]
        ])
        
        # Adjoint matrix: [R, p_skew @ R; 0, R]
        Ad = np.zeros((6, 6))
        Ad[0:3, 0:3] = R_rel
        Ad[0:3, 3:6] = p_skew @ R_rel
        Ad[3:6, 3:6] = R_rel
        
        # Relative Jacobian: J_rel = [-Ad(T_rel) @ J_a, J_b]
        J_rel = np.hstack([-Ad @ J_a, J_b])
        
        return J_rel
    
    @staticmethod
    def _relative_jacobian_numeric(model_a: RobotModel, model_b: RobotModel,
                                   q_a: np.ndarray, q_b: np.ndarray,
                                   backend: str = 'numpy',
                                   eps: float = 1e-7) -> np.ndarray:
        """Compute relative Jacobian using numerical differentiation.
        
        :param model_a: First chain model
        :param model_b: Second chain model
        :param q_a: First chain joint configuration
        :param q_b: Second chain joint configuration
        :param backend: Computation backend
        :param eps: Finite difference epsilon
        :return: 6 × (n_a + n_b) Jacobian matrix
        """
        def rel_pose_vec(qa, qb):
            Ta = model_a.fk(qa, backend=backend, return_end=True)
            Tb = model_b.fk(qb, backend=backend, return_end=True)
            if hasattr(Ta, 'detach'):
                Ta = Ta.detach().cpu().numpy()
                Tb = Tb.detach().cpu().numpy()
            Ta = np.array(Ta)
            Tb = np.array(Tb)
            
            T_rel = np.linalg.inv(Ta) @ Tb
            
            p = T_rel[0:3, 3]
            R = T_rel[0:3, 0:3]
            trace = np.trace(R)
            theta = np.arccos(np.clip((trace - 1) / 2, -1, 1))
            
            if theta < 1e-6:
                ori = np.zeros(3)
            else:
                axis = np.array([
                    R[2, 1] - R[1, 2],
                    R[0, 2] - R[2, 0],
                    R[1, 0] - R[0, 1]
                ]) / (2 * np.sin(theta))
                ori = theta * axis
            
            return np.concatenate([p, ori])
        
        q_combined = np.concatenate([q_a, q_b])
        n_total = len(q_combined)
        J = np.zeros((6, n_total))
        
        p0 = rel_pose_vec(q_a, q_b)
        
        for i in range(n_total):
            q_plus = q_combined.copy()
            q_plus[i] += eps
            
            if i < len(q_a):
                qa_p = q_plus[:len(q_a)]
                qb_p = q_b
            else:
                qa_p = q_a
                qb_p = q_plus[len(q_a):]
            
            p_plus = rel_pose_vec(qa_p, qb_p)
            J[:, i] = (p_plus - p0) / eps
        
        return J
