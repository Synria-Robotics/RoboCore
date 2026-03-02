"""Shared helpers for Pinocchio comparison: same (base_link, end_link) chain as RoboCore.

RC uses (base_link, end_link) to define the chain (e.g. 6 DOF arm only).
Pinocchio loads the full URDF (e.g. 8 DOF with gripper). We build a reduced Pinocchio
model that locks joints outside the RC chain so both use the same segment and DOF.
"""

import numpy as np

try:
    import pinocchio
    _HAS_PIN = True
except ImportError:
    _HAS_PIN = False


def get_chain_joint_names(rc_model):
    """Return actuated joint names in RC chain order (base_link -> end_link)."""
    chain_indices = rc_model._get_joint_indices(rc_model.base_link, rc_model.end_link)
    return [rc_model.joint_list[idx].name for idx in chain_indices]


def build_pinocchio_reduced_to_chain(model_path, rc_model):
    """Build Pinocchio model reduced to the same chain as RC (base_link -> end_link).

    Locks all joints not in the RC chain so pin_model.nq == pin_model.nv == rc_model.num_chain_dof.
    Returns (pin_model, pin_data) or (None, None) if Pinocchio not available / reduction fails.
    """
    if not _HAS_PIN:
        return None, None
    full_model = pinocchio.buildModelFromUrdf(model_path)
    chain_names = set(get_chain_joint_names(rc_model))
    nq_chain = rc_model.num_chain_dof
    if full_model.nv == nq_chain:
        # Full model already matches (e.g. URDF has only arm)
        return full_model, full_model.createData()
    to_lock = []
    for i in range(1, full_model.njoints):
        name = full_model.names[i]
        if name not in chain_names:
            j = full_model.joints[i]
            if j.nq > 0:
                to_lock.append(i)
    if not to_lock:
        return full_model, full_model.createData()
    try:
        q_ref = pinocchio.neutral(full_model)
        red_model = pinocchio.buildReducedModel(full_model, to_lock, q_ref)
        red_data = red_model.createData()
        if red_model.nq == nq_chain and red_model.nv == nq_chain:
            return red_model, red_data
    except Exception:
        pass
    return full_model, full_model.createData()
