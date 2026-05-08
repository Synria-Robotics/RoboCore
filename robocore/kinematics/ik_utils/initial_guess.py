"""Initial Guess Generation for Inverse Kinematics

This module provides utilities for generating initial guesses for IK solvers.

Copyright (c) 2025 Synria Robotics Co., Ltd.

Licensed under the MIT License.

Author: Synria Robotics Team
Website: https://synriarobotics.ai
"""

from typing import Optional
import numpy as np

# Try to import scipy for advanced sampling methods
_HAS_SCIPY = False
try:
    from scipy.stats import qmc
    _HAS_SCIPY = True
except ImportError:
    pass


def generate_initial_guesses(
    model,
    num_guesses: int,
    strategy: str = 'random',
    seed: Optional[int] = None,
    scale: float = 1.0,
    base_q0: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Generate initial guesses for IK.
    
    :param model: RobotModel instance
    :param num_guesses: Number of initial guesses to generate
    :param strategy: 'zero', 'random', 'sobol', 'latin', 'center', 'uniform'
    :param seed: Random seed for reproducibility
    :param scale: Scale factor for joint limits (0.0 to 1.0)
    :param base_q0: Base configuration for noise-based strategies [n_dof]
    :return: Initial guesses array [num_guesses, n_dof]
    """
    if num_guesses <= 0:
        raise ValueError(f"num_guesses must be positive, got {num_guesses}")
    
    n_dof = model.num_chain_dof
    rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()
    
    # Get joint limits
    joint_lower = []
    joint_upper = []
    for js in model._chain_dof_list:
        lo = js.limit_lower if js.limit_lower is not None else -1.0
        hi = js.limit_upper if js.limit_upper is not None else 1.0
        joint_lower.append(lo)
        joint_upper.append(hi)
    joint_lower = np.array(joint_lower)
    joint_upper = np.array(joint_upper)
    
    # Apply scale to joint limits
    if scale < 1.0:
        mid = 0.5 * (joint_lower + joint_upper)
        span = 0.5 * (joint_upper - joint_lower) * scale
        joint_lower = mid - span
        joint_upper = mid + span
    
    strategy = strategy.lower()
    
    if strategy == 'zero':
        # All zeros
        guesses = np.zeros((num_guesses, n_dof))
    
    elif strategy == 'center':
        # Joint space center
        center = 0.5 * (joint_lower + joint_upper)
        guesses = np.tile(center, (num_guesses, 1))
    
    elif strategy == 'random':
        # Uniform random sampling
        guesses = np.zeros((num_guesses, n_dof))
        for i in range(num_guesses):
            alpha = rng.random(n_dof)
            guesses[i] = joint_lower + alpha * (joint_upper - joint_lower)
    
    elif strategy == 'uniform':
        # Simple uniform sampling (same as random, kept for compatibility)
        guesses = np.zeros((num_guesses, n_dof))
        for i in range(num_guesses):
            alpha = rng.random(n_dof)
            guesses[i] = joint_lower + alpha * (joint_upper - joint_lower)
    
    elif strategy == 'sobol':
        # Sobol sequence (quasi-random, better space-filling)
        if not _HAS_SCIPY:
            # Fallback to random if scipy not available
            guesses = np.zeros((num_guesses, n_dof))
            for i in range(num_guesses):
                alpha = rng.random(n_dof)
                guesses[i] = joint_lower + alpha * (joint_upper - joint_lower)
        else:
            try:
                sampler = qmc.Sobol(d=n_dof, seed=rng.integers(0, 2**31) if seed is not None else None)
                # Generate more samples than needed for better distribution
                n_samples = max(num_guesses, 2**n_dof) if n_dof <= 6 else num_guesses * 2
                samples_all = sampler.random(n=n_samples)
                # Select evenly spaced samples
                if len(samples_all) > num_guesses:
                    indices = np.linspace(0, len(samples_all) - 1, num_guesses, dtype=int)
                    samples = samples_all[indices]
                else:
                    samples = samples_all[:num_guesses]
                
                # Scale to joint limits
                guesses = joint_lower + samples * (joint_upper - joint_lower)
            except Exception:
                # Fallback to Latin Hypercube if Sobol fails
                sampler = qmc.LatinHypercube(d=n_dof, seed=rng.integers(0, 2**31) if seed is not None else None)
                samples = sampler.random(n=num_guesses)
                guesses = joint_lower + samples * (joint_upper - joint_lower)
    
    elif strategy == 'latin':
        # Latin Hypercube sampling
        if not _HAS_SCIPY:
            # Fallback to random if scipy not available
            guesses = np.zeros((num_guesses, n_dof))
            for i in range(num_guesses):
                alpha = rng.random(n_dof)
                guesses[i] = joint_lower + alpha * (joint_upper - joint_lower)
        else:
            try:
                sampler = qmc.LatinHypercube(d=n_dof, seed=rng.integers(0, 2**31) if seed is not None else None)
                samples = sampler.random(n=num_guesses)
                guesses = joint_lower + samples * (joint_upper - joint_lower)
            except Exception:
                # Fallback to random if Latin Hypercube fails
                guesses = np.zeros((num_guesses, n_dof))
                for i in range(num_guesses):
                    alpha = rng.random(n_dof)
                    guesses[i] = joint_lower + alpha * (joint_upper - joint_lower)
    
    else:
        raise ValueError(f"Unknown strategy '{strategy}'. Supported: 'zero', 'random', 'sobol', 'latin', 'center', 'uniform'")
    
    # If base_q0 is provided and strategy supports it, add noise around base
    if base_q0 is not None and strategy in ('random', 'uniform'):
        # Use base_q0 as first guess, add noise for others
        base_q0 = np.asarray(base_q0)
        if base_q0.shape != (n_dof,):
            raise ValueError(f"base_q0 must have shape ({n_dof},), got {base_q0.shape}")
        
        guesses[0] = base_q0
        # Add small noise to remaining guesses
        noise_scale = 0.1  # radians
        for i in range(1, num_guesses):
            noise = rng.normal(0, noise_scale, n_dof)
            guesses[i] = np.clip(base_q0 + noise, joint_lower, joint_upper)
    
    return guesses
