"""Optional PyTorch Kinematics + Pinocchio helpers for Alicia-D backend benchmarks.

Install separately (not RoboCore core deps)::

    pip install pytorch_kinematics pin

Copyright (c) 2025 Synria Robotics Co., Ltd.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np
from numpy.linalg import norm, solve

_TORCH = None
_PK = None
_PIN = None
_IMPORT_ERRORS: List[str] = []

try:
    import torch as _torch

    _TORCH = _torch
except ImportError as e:
    _IMPORT_ERRORS.append(f"torch: {e}")

try:
    import pytorch_kinematics as _pk

    _PK = _pk
except ImportError as e:
    _IMPORT_ERRORS.append(f"pytorch_kinematics: {e}")

try:
    import pinocchio as _pin

    _PIN = _pin
except ImportError as e:
    _IMPORT_ERRORS.append(f"pinocchio: {e}")


def pk_pin_status() -> Tuple[bool, str]:
    """Return (ok, message) for optional benchmark backends."""
    if _TORCH is None or _PK is None or _PIN is None:
        return False, "; ".join(_IMPORT_ERRORS) if _IMPORT_ERRORS else "missing optional deps"
    return True, "ok"


def pin_q_full_from_chain(pin_model, q_chain: np.ndarray, pin_q_indices: Sequence[int]) -> np.ndarray:
    """Build Pinocchio ``q_full`` from chain coordinates.

    :param pin_model: Pinocchio model
    :param q_chain: values in chain order matching ``pin_q_indices``
    :return: full configuration vector
    """
    q_full = _PIN.neutral(pin_model).copy()
    q_chain = np.asarray(q_chain, dtype=np.float64).ravel()
    for j, pin_idx in enumerate(pin_q_indices):
        if pin_idx < len(q_full):
            q_full[pin_idx] = float(q_chain[j])
    return q_full


def pin_ee_homogeneous(pin_model, pin_data, q_full, end_frame_id, end_joint_id) -> np.ndarray:
    """End-effector 4x4 after FK at ``q_full``.

    :return: (4, 4) array
    """
    _PIN.forwardKinematics(pin_model, pin_data, q_full)
    if end_frame_id is not None:
        _PIN.updateFramePlacements(pin_model, pin_data)
        return np.asarray(pin_data.oMf[end_frame_id].homogeneous, dtype=np.float64)
    return np.asarray(pin_data.oMi[end_joint_id].homogeneous, dtype=np.float64)


def pin_target_pose_from_chain_q(
    pin_model, pin_data, q_chain, pin_q_indices, end_frame_id, end_joint_id
) -> np.ndarray:
    """Pin FK target for IK (same convention as ``solve_ik_pinocchio``).

    :param q_chain: chain joint vector aligned with ``pin_q_indices``
    :return: (4, 4) homogeneous matrix
    """
    q_full = pin_q_full_from_chain(pin_model, q_chain, pin_q_indices)
    return pin_ee_homogeneous(pin_model, pin_data, q_full, end_frame_id, end_joint_id)


def solve_ik_pinocchio(
    pin_model,
    pin_data,
    target_pose: np.ndarray,
    q_init: np.ndarray,
    pin_q_indices: Sequence[int],
    pin_v_indices: Sequence[int],
    end_joint_id: Optional[int],
    end_frame_id: Optional[int],
    pos_tol: float,
    ori_tol: float,
    max_iters: int,
    damping: float,
    step_size: float,
    robot_model=None,
) -> dict:
    """Pinocchio CLIK (same logic as ``03c_demo_ik_pk.py``).

    :param target_pose: (4, 4) desired ``oMf`` / ``oMi`` in Pinocchio convention
    :param q_init: actuated joints only, chain order
    :return: result dict with success, q, iters, pos_err, ori_err
    """
    q_full = pin_q_full_from_chain(pin_model, q_init, pin_q_indices)

    joint_limits = None
    if robot_model is not None:
        chain_indices = robot_model._get_joint_indices(robot_model.base_link, robot_model.end_link)
        joint_limits = [
            (robot_model.joint_list[idx].limit_lower, robot_model.joint_list[idx].limit_upper)
            for idx in chain_indices
        ]

    oMdes = _PIN.SE3(target_pose[:3, :3], target_pose[:3, 3])
    min_damping, max_damping = 1e-4, 5e-2
    base_damping = max(damping, min_damping)

    def fk_err(qf):
        _PIN.forwardKinematics(pin_model, pin_data, qf)
        if end_frame_id is not None:
            _PIN.updateFramePlacements(pin_model, pin_data)
            iMd = pin_data.oMf[end_frame_id].actInv(oMdes)
        else:
            iMd = pin_data.oMi[end_joint_id].actInv(oMdes)
        e = _PIN.log6(iMd).vector
        return e, iMd

    for it in range(max_iters):
        err, iMd = fk_err(q_full)
        pos_err, ori_err = norm(err[:3]), norm(err[3:])
        if pos_err < pos_tol and ori_err < ori_tol:
            return {
                "success": True,
                "q": np.array([q_full[idx] for idx in pin_q_indices]),
                "iters": it + 1,
                "pos_err": pos_err,
                "ori_err": ori_err,
            }

        if end_frame_id is not None:
            _PIN.computeJointJacobians(pin_model, pin_data, q_full)
            J_full = _PIN.getFrameJacobian(
                pin_model, pin_data, end_frame_id, _PIN.ReferenceFrame.LOCAL_WORLD_ALIGNED
            )
        else:
            _PIN.computeJointJacobians(pin_model, pin_data, q_full)
            J_full = _PIN.getJointJacobian(
                pin_model, pin_data, end_joint_id, _PIN.ReferenceFrame.LOCAL_WORLD_ALIGNED
            )

        J = J_full[:, pin_v_indices] if len(pin_v_indices) > 0 else J_full
        Jlog = _PIN.Jlog6(iMd.inverse())
        J_se3 = -Jlog @ J

        try:
            _u, s, _vt = np.linalg.svd(J_se3, full_matrices=False)
            cond_num = s[0] / (s[-1] + 1e-10)
            if cond_num > 1e6:
                adaptive_damping = max_damping
            elif cond_num > 1e4:
                adaptive_damping = min_damping + (max_damping - min_damping) * (cond_num - 1e4) / (1e6 - 1e4)
            else:
                adaptive_damping = min_damping
        except Exception:
            adaptive_damping = base_damping

        JJt = J_se3 @ J_se3.T + adaptive_damping * np.eye(6)
        v = -J_se3.T @ solve(JJt, err)
        v_norm = norm(v)
        if v_norm > 0.5:
            v *= 0.5 / v_norm

        v_full = np.zeros(pin_model.nv)
        for i_v, pin_v_idx in enumerate(pin_v_indices):
            if pin_v_idx < len(v_full):
                v_full[pin_v_idx] = v[i_v]

        q_full_new = _PIN.integrate(pin_model, q_full, v_full * step_size)
        if joint_limits is not None:
            for j, (lo, hi) in enumerate(joint_limits):
                pin_idx = pin_q_indices[j]
                if pin_idx >= len(q_full_new):
                    continue
                if lo is not None:
                    q_full_new[pin_idx] = max(q_full_new[pin_idx], lo)
                if hi is not None:
                    q_full_new[pin_idx] = min(q_full_new[pin_idx], hi)
        q_full = q_full_new

    err, _ = fk_err(q_full)
    pos_err, ori_err = norm(err[:3]), norm(err[3:])
    return {
        "success": False,
        "q": np.array([q_full[idx] for idx in pin_q_indices]),
        "iters": max_iters,
        "pos_err": pos_err,
        "ori_err": ori_err,
    }


@dataclass
class AliciaPkPinFKJac:
    """PyTorch Kinematics chain + Pinocchio model for FK/Jacobian timing."""

    rc_model: Any
    urdf_path: Path
    base_link: str
    end_link: str
    device: Any
    dtype: Any
    chain: Any
    pin_model: Any
    pin_data: Any
    pin_q_indices: List[int]
    pin_v_indices: List[int]
    end_frame_id: Optional[int]
    end_joint_id: Optional[int]

    @classmethod
    def build(cls, rc_model: Any, urdf_path: Path, base_link: str, end_link: str, device_str: str) -> "AliciaPkPinFKJac":
        """Load PK chain and Pinocchio model; map joints like ``01c`` / ``02c`` demos."""
        ok, _ = pk_pin_status()
        if not ok:
            raise RuntimeError("pk_pin_status() failed")

        dtype = _TORCH.float64
        device = _TORCH.device(device_str)
        urdf_bytes = Path(urdf_path).read_bytes()
        chain = _PK.build_serial_chain_from_urdf(urdf_bytes, end_link, root_link_name=base_link)
        chain = chain.to(dtype=dtype, device=device)

        pin_model = _PIN.buildModelFromUrdf(str(urdf_path))
        pin_data = pin_model.createData()

        chain_indices = rc_model._get_joint_indices(rc_model.base_link, rc_model.end_link)
        actuated_joint_names = [rc_model.joint_list[idx].name for idx in chain_indices]
        pin_q_indices: List[int] = []
        pin_v_indices: List[int] = []
        for joint_name in actuated_joint_names:
            if pin_model.existJointName(joint_name):
                jid = pin_model.getJointId(joint_name)
                pin_q_indices.append(pin_model.idx_qs[jid] if jid < len(pin_model.idx_qs) else jid)
                pin_v_indices.append(pin_model.idx_vs[jid] if jid < len(pin_model.idx_vs) else jid)

        end_frame_id = None
        end_joint_id = None
        if pin_model.existFrame(end_link):
            end_frame_id = pin_model.getFrameId(end_link)
        elif pin_model.existJointName(end_link):
            end_joint_id = pin_model.getJointId(end_link)
        else:
            end_joint_id = len(pin_model.joints) - 1

        return cls(
            rc_model=rc_model,
            urdf_path=urdf_path,
            base_link=base_link,
            end_link=end_link,
            device=device,
            dtype=dtype,
            chain=chain,
            pin_model=pin_model,
            pin_data=pin_data,
            pin_q_indices=pin_q_indices,
            pin_v_indices=pin_v_indices,
            end_frame_id=end_frame_id,
            end_joint_id=end_joint_id,
        )

    def pk_fk1(self, q_np: np.ndarray) -> None:
        """One FK call (batch=1); side effect only for timing."""
        q = _TORCH.as_tensor(q_np, dtype=self.dtype, device=self.device).unsqueeze(0)
        ret = self.chain.forward_kinematics(q, end_only=False)
        _ = ret[self.end_link].get_matrix()

    def pk_fk_batch(self, q_bn: np.ndarray) -> None:
        """Batched FK; ``q_bn`` shape (B, n)."""
        q = _TORCH.as_tensor(q_bn, dtype=self.dtype, device=self.device)
        ret = self.chain.forward_kinematics(q, end_only=False)
        _ = ret[self.end_link].get_matrix()

    def pin_fk1(self, q_np: np.ndarray) -> None:
        _ = self.pin_fk_matrix(q_np)

    def pin_fk_matrix(self, q_np: np.ndarray) -> np.ndarray:
        q_full = pin_q_full_from_chain(self.pin_model, q_np, self.pin_q_indices)
        return pin_ee_homogeneous(self.pin_model, self.pin_data, q_full, self.end_frame_id, self.end_joint_id)

    def pin_fk_batch(self, q_bn: np.ndarray) -> None:
        for i in range(q_bn.shape[0]):
            self.pin_fk1(q_bn[i])

    def pin_fk_batch_matrix(self, q_bn: np.ndarray) -> np.ndarray:
        return np.stack([self.pin_fk_matrix(q_bn[i]) for i in range(q_bn.shape[0])], axis=0)

    def pk_jac1(self, q_np: np.ndarray) -> None:
        q = _TORCH.as_tensor(q_np, dtype=self.dtype, device=self.device)
        _ = self.chain.jacobian(q)

    def pk_jac_batch(self, q_bn: np.ndarray) -> None:
        q = _TORCH.as_tensor(q_bn, dtype=self.dtype, device=self.device)
        _ = self.chain.jacobian(q)

    def pin_jac1(self, q_np: np.ndarray) -> None:
        _ = self.pin_jacobian_matrix(q_np)

    def pin_jacobian_matrix(self, q_np: np.ndarray) -> np.ndarray:
        q_full = pin_q_full_from_chain(self.pin_model, q_np, self.pin_q_indices)
        _PIN.forwardKinematics(self.pin_model, self.pin_data, q_full)
        if self.end_frame_id is not None:
            _PIN.updateFramePlacements(self.pin_model, self.pin_data)
            _PIN.computeJointJacobians(self.pin_model, self.pin_data, q_full)
            J_full = _PIN.getFrameJacobian(
                self.pin_model, self.pin_data, self.end_frame_id, _PIN.ReferenceFrame.LOCAL_WORLD_ALIGNED
            )
        else:
            _PIN.computeJointJacobians(self.pin_model, self.pin_data, q_full)
            J_full = _PIN.getJointJacobian(
                self.pin_model, self.pin_data, self.end_joint_id, _PIN.ReferenceFrame.LOCAL_WORLD_ALIGNED
            )
        J = J_full[:, self.pin_v_indices] if len(self.pin_v_indices) > 0 else J_full
        return np.asarray(J, dtype=np.float64)

    def pin_jac_batch(self, q_bn: np.ndarray) -> None:
        for i in range(q_bn.shape[0]):
            self.pin_jac1(q_bn[i])

    def pin_jac_batch_matrix(self, q_bn: np.ndarray) -> np.ndarray:
        return np.stack([self.pin_jacobian_matrix(q_bn[i]) for i in range(q_bn.shape[0])], axis=0)

    def sync_torch(self) -> None:
        if self.device.type == "cuda":
            _TORCH.cuda.synchronize()


@dataclass
class AliciaPkPinIK:
    """PseudoInverseIK + Pinocchio CLIK for IK benchmark timing."""

    fkctx: AliciaPkPinFKJac
    ik_pk: Any
    T_pin_batch: np.ndarray
    pin_ik_end_joint_id: Optional[int]
    pos_tol: float
    ori_tol: float
    max_iters: int
    damping: float
    step_size: float
    last_T_pk1: Any = None
    last_T_pkb: Any = None

    @classmethod
    def build(
        cls,
        fkctx: AliciaPkPinFKJac,
        *,
        pos_tol: float,
        ori_tol: float,
        max_iters: int,
        damping: float,
        step_size: float,
        num_retries: int,
    ) -> "AliciaPkPinIK":
        from pytorch_kinematics.ik import PseudoInverseIK

        jl = _TORCH.tensor(fkctx.rc_model.chain_joint_limit, dtype=fkctx.dtype, device=fkctx.device)
        ik_pk = PseudoInverseIK(
            fkctx.chain,
            pos_tolerance=pos_tol,
            rot_tolerance=ori_tol,
            max_iterations=max_iters,
            lr=step_size,
            regularlization=damping,
            num_retries=num_retries,
            joint_limits=jl,
        )
        pin_ik_end_joint_id = fkctx.end_joint_id if fkctx.end_frame_id is None else None
        return cls(
            fkctx=fkctx,
            ik_pk=ik_pk,
            T_pin_batch=np.zeros((0, 4, 4)),
            pin_ik_end_joint_id=pin_ik_end_joint_id,
            pos_tol=pos_tol,
            ori_tol=ori_tol,
            max_iters=max_iters,
            damping=damping,
            step_size=step_size,
        )

    def set_T_pin_batch(self, q_star_b: np.ndarray) -> None:
        """Precompute Pinocchio targets from chain-space reference poses (rows)."""
        B = q_star_b.shape[0]
        out = np.zeros((B, 4, 4), dtype=np.float64)
        for i in range(B):
            out[i] = pin_target_pose_from_chain_q(
                self.fkctx.pin_model,
                self.fkctx.pin_data,
                q_star_b[i],
                self.fkctx.pin_q_indices,
                self.fkctx.end_frame_id,
                self.fkctx.end_joint_id,
            )
        self.T_pin_batch = out

    def pk_targets_from_q_star(self, q_star: np.ndarray, q_star_b: np.ndarray) -> Tuple[Any, Any]:
        """PK-frame targets: FK(q) as torch (1,4,4) and (B,4,4)."""
        q1 = _TORCH.as_tensor(q_star, dtype=self.fkctx.dtype, device=self.fkctx.device).unsqueeze(0)
        r1 = self.fkctx.chain.forward_kinematics(q1, end_only=False)
        T1 = r1[self.fkctx.end_link].get_matrix()

        qb = _TORCH.as_tensor(q_star_b, dtype=self.fkctx.dtype, device=self.fkctx.device)
        rb = self.fkctx.chain.forward_kinematics(qb, end_only=False)
        Tb = rb[self.fkctx.end_link].get_matrix()
        self.last_T_pk1 = T1
        self.last_T_pkb = Tb
        return T1, Tb

    def pk_solve1(self, T_4x4: Any) -> None:
        from pytorch_kinematics.transforms import Transform3d

        tt = Transform3d(matrix=T_4x4)
        _ = self.ik_pk.solve(tt)

    def pk_solve_batch(self, T_b444: Any) -> None:
        from pytorch_kinematics.transforms import Transform3d

        tt = Transform3d(matrix=T_b444)
        _ = self.ik_pk.solve(tt)

    def pin_solve1(self, T_pin: np.ndarray, q0: np.ndarray) -> None:
        _ = solve_ik_pinocchio(
            self.fkctx.pin_model,
            self.fkctx.pin_data,
            T_pin,
            q0,
            self.fkctx.pin_q_indices,
            self.fkctx.pin_v_indices,
            self.pin_ik_end_joint_id,
            self.fkctx.end_frame_id,
            self.pos_tol,
            self.ori_tol,
            self.max_iters,
            self.damping,
            self.step_size,
            robot_model=self.fkctx.rc_model,
        )

    def pin_solve_batch(self, q0_b: np.ndarray) -> None:
        for i in range(self.T_pin_batch.shape[0]):
            self.pin_solve1(self.T_pin_batch[i], q0_b[i])

    def sync_torch(self) -> None:
        self.fkctx.sync_torch()
