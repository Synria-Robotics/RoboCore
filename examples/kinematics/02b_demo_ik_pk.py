"""
Inverse Kinematics Demo with Pytorch Kinematics

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

import numpy as np
import torch
import pytorch_kinematics as pk
import argparse
import time
from robocore.utils.beauty_logger import beauty_print_array, beauty_print
from robocore.transform.conversions import quaternion_to_matrix, matrix_to_quaternion


def compute_jacobian(chain, q, end_link, epsilon=1e-5):
    """Compute Jacobian using finite differences (more reliable than built-in for IK)."""
    n = len(q)
    J = torch.zeros((6, n), dtype=q.dtype, device=q.device)
    
    # Current pose
    q_tensor = q.unsqueeze(0)
    ret = chain.forward_kinematics(q_tensor, end_only=False)
    tg = ret[end_link]
    m = tg.get_matrix()[0]  # (4, 4)
    p_ref = m[:3, 3]
    R_ref = m[:3, :3]
    
    for i in range(n):
        # Forward difference
        qp = q.clone()
        qp[i] += epsilon
        qp_tensor = qp.unsqueeze(0)
        
        ret_pos = chain.forward_kinematics(qp_tensor, end_only=False)
        tg_pos = ret_pos[end_link]
        m_pos = tg_pos.get_matrix()[0]
        p_pos = m_pos[:3, 3]
        R_pos = m_pos[:3, :3]
        
        # Linear velocity
        J[:3, i] = (p_pos - p_ref) / epsilon
        
        # Angular velocity (rotation error)
        # Compute relative rotation: R_error = R_pos @ R_ref^T
        R_error = R_pos @ R_ref.T
        # Convert to axis-angle using pytorch_kinematics
        # matrix_to_axis_angle returns compact representation (axis * angle)
        axis_angle_vec = pk.matrix_to_axis_angle(R_error.unsqueeze(0))[0]  # (3,)
        J[3:, i] = axis_angle_vec / epsilon
    
    return J


def compute_pose_error(T_current, T_target):
    """Compute 6D pose error (position + orientation)."""
    # Position error
    e_pos = T_target[:3, 3] - T_current[:3, 3]
    
    # Orientation error (axis-angle representation)
    R_current = T_current[:3, :3]
    R_target = T_target[:3, :3]
    R_error = R_target @ R_current.T
    
    # Convert rotation error to axis-angle using pytorch_kinematics
    axis_angle_result = pk.matrix_to_axis_angle(R_error.unsqueeze(0))
    if isinstance(axis_angle_result, tuple):
        axis, angle = axis_angle_result
        axis = axis[0]  # Remove batch dimension
        angle = angle[0] if angle.dim() > 0 else angle
    else:
        # If it returns a single tensor, extract axis and angle
        axis_angle_vec = axis_angle_result[0]  # (3,) compact representation
        angle = torch.norm(axis_angle_vec)
        axis = axis_angle_vec / (angle + 1e-10)
    e_ori = axis * angle
    
    return torch.cat([e_pos, e_ori])


def solve_ik_dls(chain, T_target, q0, end_link, max_iters=100, pos_tol=1e-4, ori_tol=1e-3, 
                  damping=1e-3, step_size=1.0):
    """Solve IK using Damped Least Squares (DLS) method."""
    device = q0.device
    dtype = q0.dtype
    n = len(q0)
    
    q = q0.clone()
    best_q = q.clone()
    best_err = float('inf')
    best_pos_err = float('inf')
    best_ori_err = float('inf')
    
    for iter in range(max_iters):
        # Forward kinematics
        q_tensor = q.unsqueeze(0)
        ret = chain.forward_kinematics(q_tensor, end_only=False)
        tg = ret[end_link]
        T_current = tg.get_matrix()[0]  # (4, 4)
        
        # Compute error
        e = compute_pose_error(T_current, T_target)
        pos_err = torch.norm(e[:3])
        ori_err = torch.norm(e[3:])
        total_err = torch.norm(e)
        
        # Update best solution
        if total_err < best_err:
            best_err = total_err.item()
            best_pos_err = pos_err.item()
            best_ori_err = ori_err.item()
            best_q = q.clone()
        
        # Check convergence
        if pos_err < pos_tol and ori_err < ori_tol:
            return {
                'success': True,
                'q': q.detach().cpu().numpy(),
                'iters': iter + 1,
                'pos_err': pos_err.item(),
                'ori_err': ori_err.item(),
                'err_norm': total_err.item()
            }
        
        # Compute Jacobian using pytorch_kinematics
        J = compute_jacobian(chain, q, end_link)
        
        # Damped Least Squares: dq = J^T (J J^T + lambda^2 I)^(-1) e
        J_JT = J @ J.T
        damping_matrix = damping * torch.eye(6, dtype=dtype, device=device)
        dq = J.T @ torch.linalg.solve(J_JT + damping_matrix, e)
        
        # Apply step with step size
        q = q + step_size * dq
        
        # Clamp to joint limits if available
        # (pytorch_kinematics doesn't provide joint limits directly, so we skip this)
    
    return {
        'success': False,
        'q': best_q.detach().cpu().numpy(),
        'iters': max_iters,
        'pos_err': best_pos_err,
        'ori_err': best_ori_err,
        'err_norm': best_err
    }


def main(args):
    start_time = time.time()
    import os
    model_path = str(args.model_path)
    end_link = args.end_link
    
    # Verify file path exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"URDF file not found: {model_path}")
    
    # Load robot description from URDF
    with open(model_path, 'rb') as f:
        urdf_bytes = f.read()
    # Use base_link as root to match robocore's coordinate system
    chain = pk.build_serial_chain_from_urdf(urdf_bytes, end_link, root_link_name='base_link')
    
    # Print chain info
    print(chain)
    print(chain.get_joint_parameter_names())
    
    # Build target pose from input
    T_target = torch.eye(4, dtype=torch.float32)
    T_target[:3, 3] = torch.tensor(args.end_pose[:3], dtype=torch.float32)
    quat_target_xyzw = np.array(args.end_pose[3:])  # [x, y, z, w]
    # Convert quaternion [x, y, z, w] to rotation matrix using robocore's function
    from robocore.transform.conversions import quaternion_to_matrix
    R_target = torch.tensor(quaternion_to_matrix(quat_target_xyzw), dtype=torch.float32)
    T_target[:3, :3] = R_target
    
    # Generate random initial guess
    n_dof = len(chain.get_joint_parameter_names())
    rng = np.random.default_rng(42)
    q_init = torch.tensor(rng.uniform(-np.pi/2, np.pi/2, n_dof), dtype=torch.float32)
    
    beauty_print(f"Initial Guess (radians):")
    print(f"  q_init = {beauty_print_array(q_init.numpy())}")
    
    beauty_print(f"Target End-Effector Pose:")
    print(f"  Position: {beauty_print_array(T_target[:3, 3].numpy())}")
    quat_display = matrix_to_quaternion(T_target[:3, :3].numpy())
    print(f"  Quaternion (xyzw): {beauty_print_array(quat_display, precision=6)}")
    
    # Solve IK
    ik_result = solve_ik_dls(
        chain,
        T_target,
        q_init,
        end_link,
        max_iters=args.max_iters,
        pos_tol=args.pos_tol,
        ori_tol=args.ori_tol,
        damping=args.damping,
        step_size=args.step_size
    )
    
    beauty_print(f"IK Solution:")
    print(f"  Success: {ik_result['success']}")
    print(f"  Iterations: {ik_result['iters']}")
    print(f"  Position Error: {ik_result['pos_err']:.6e} m")
    print(f"  Orientation Error: {ik_result['ori_err']:.6e} rad")
    print(f"  Total Error: {ik_result['err_norm']:.6e}")
    
    beauty_print(f"Solved Joint Angles (radians):")
    print(f"  q_ik = {beauty_print_array(ik_result['q'])}")
    beauty_print(f"Solved Joint Angles (degrees):")
    print(f"  q_ik = {beauty_print_array(np.rad2deg(ik_result['q']))}")
    
    # Verify solution with FK
    q_ik_tensor = torch.tensor(ik_result['q'], dtype=torch.float32).unsqueeze(0)
    ret_verify = chain.forward_kinematics(q_ik_tensor, end_only=False)
    tg_verify = ret_verify[end_link]
    T_verify = tg_verify.get_matrix()[0]
    
    beauty_print(f"Verification (FK of IK solution):")
    print(f"  Position: {beauty_print_array(T_verify[:3, 3].numpy())}")
    print(f"  Position Error: {np.linalg.norm(T_verify[:3, 3].numpy() - T_target[:3, 3].numpy()):.6e} m")
    
    end_time = time.time()
    beauty_print(f"Pytorch Kinematics IKComputation Time: {end_time - start_time: .6f} seconds")


if __name__ == "__main__":
    from synriard import get_model_path
    
    model_path = get_model_path("Alicia_D", version="v5_6", variant="gripper_100mm", model_format="urdf")
    
    parser = argparse.ArgumentParser(description="Inverse Kinematics Demo with Pytorch Kinematics")
    parser.add_argument('--model-path', type=str,
                        default=model_path,
                        help='Path to URDF file (default: Alicia-D)')
    parser.add_argument('--end-link', type=str, default='Link6', help='End-effector link name')
    parser.add_argument('--end-pose', type=float, nargs='+', 
                        default=[0.17006, 0.01704, 0.20533, 0.042114, 0.828366, 0.083037, 0.552396],
                        help='Target end-effector pose as 7 floats (px, py, pz, qx, qy, qz, qw)')
    parser.add_argument('--max-iters', type=int, default=100, help='Maximum iterations')
    parser.add_argument('--pos-tol', type=float, default=1e-4, help='Position tolerance (m)')
    parser.add_argument('--ori-tol', type=float, default=1e-3, help='Orientation tolerance (rad)')
    parser.add_argument('--damping', type=float, default=1e-3, help='Damping factor for DLS')
    parser.add_argument('--step-size', type=float, default=1.0, help='Step size multiplier')
    args = parser.parse_args()
    main(args)

