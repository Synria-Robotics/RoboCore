"""Robot Model Abstraction

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

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Any, ClassVar, Tuple, Union

from .parser.urdf_parser import load_urdf, URDFJoint
from .parser.mjcf_parser import load_mjcf
from robocore.kinematics.fk import forward_kinematics
from robocore.kinematics.ik import inverse_kinematics
from robocore.kinematics.jacobian import jacobian
from robocore.kinematics.utils import relative_pose_error, relative_jacobian
from robocore.utils.backend import get_backend
import numpy as np

from robocore.utils.beauty_logger import beauty_print, beauty_print_array


# Lazy import to avoid circular dependency
def _get_workspace_analyzer():
    from robocore.analysis.workspace_analyzer import WorkspaceAnalyzer
    return WorkspaceAnalyzer


@dataclass
class JointSpec:
    """Actuated joint specification.

    :param name: joint name.
    :param index: index in configuration.
    :param joint_type: revolute/prismatic.
    :param axis: axis vector (3,).
    :param parent: parent link.
    :param child: child link.
    :param origin_xyz: translation of joint frame.
    :param origin_rpy: rpy of joint frame.
    :param limit: (lower, upper) or None.
    """

    name: str
    index: int
    joint_type: str
    axis: List[float]
    parent: str
    child: str
    origin_xyz: List[float]
    origin_rpy: List[float]
    limit: Optional[Sequence[Optional[float]]]


_PARSED_ROBOT_CACHE: Dict[str, Dict[str, Any]] = {}


class RobotModel:
    """Generic serial chain robot.

    :param file_path: URDF (or MJCF in future) path.
    :param end_link: override end-effector link name.
    """

    def __init__(self, file_path: str | Path, end_link: Optional[str] = None, _parsed: Optional[Dict[str, Any]] = None):
        """Initialize robot model.

        :param file_path: path to URDF or MJCF file.
        :param end_link: end-effector link name (auto-detect if None).
        """
        self.file_path = str(file_path)
        # Auto-detect format by file extension (simple heuristic). If '.xml' we try MJCF first.
        path_lower = str(self.file_path).lower()
        parsed: Dict[str, Any] | None = _parsed
        # Cache key (normalize path)
        cache_key = str(Path(self.file_path).resolve())
        if parsed is None:
            # Try cache first
            cached = _PARSED_ROBOT_CACHE.get(cache_key)
            if cached is not None:
                parsed = cached
            else:
                # Parse new
                parsed = None
                if path_lower.endswith('.xml'):
                    try:
                        parsed = load_mjcf(self.file_path)
                    except Exception as e:
                        beauty_print(f"⚠️ MJCF parse failed ({e}); falling back to URDF parser", type="warning")
                if parsed is None:
                    parsed = load_urdf(self.file_path)
                # Store in cache (shallow dict copy to avoid accidental mutation)
                _PARSED_ROBOT_CACHE[cache_key] = parsed
        # Keep reference for spawn_chain
        self._parsed_source = parsed
        self.name = parsed.get("name", "")
        self._raw_joints = parsed["joints"]
        self.base_link = parsed["base_links"][0] if parsed["base_links"] else self._raw_joints[0].parent
        self._graph = self._build_graph(self._raw_joints)
        self._chain_joints = self._linearize_chain(self.base_link, end_link)
        self._actuated = []
        idx = 0
        for j in self._chain_joints:
            if j.joint_type in ("revolute", "prismatic"):
                self._actuated.append(
                    JointSpec(
                        name=j.name,
                        index=idx,
                        joint_type=j.joint_type,
                        axis=j.axis,
                        parent=j.parent,
                        child=j.child,
                        origin_xyz=j.origin_xyz,
                        origin_rpy=j.origin_rpy,
                        limit=(j.limit_lower, j.limit_upper),
                    )
                )
                idx += 1
        self.end_link = end_link or (self._chain_joints[-1].child if self._chain_joints else self.base_link)

        # Workspace cache (lazy-loaded on first access)
        self._workspace_analyzer = None
        self._workspace_points = None  # Cached reachable workspace points
        self._workspace_kdtree = None  # KDTree for fast reachability checks
        self._workspace_bounds = None  # Bounding box

        # Multi-link groups (name -> RobotModel via spawn_chain)
        self._groups: Dict[str, "RobotModel"] = {}

        # Only print load message if this is the first parse (heuristic: not from cache or explicit _parsed)
        if cache_key not in _PARSED_ROBOT_CACHE or _parsed is not None:
            # already handled above; but avoid duplicate logs for spawn_chain
            pass
        else:
            beauty_print(f"📦 Loading robot model from: {self.file_path}")
            beauty_print(f"✓ Robot loaded: {self.num_dof()} DOF, end_link={self.end_link}", type="success")
            

    @staticmethod
    def _build_graph(joints: List[URDFJoint]):
        g: Dict[str, List[URDFJoint]] = {}
        for j in joints:
            g.setdefault(j.parent, []).append(j)
        return g

    def _linearize_chain(self, base: str, end_link: Optional[str]) -> List[URDFJoint]:
        if end_link is None:
            # choose longest path by simple DFS
            best: List[URDFJoint] = []

            def dfs(link: str, path: List[URDFJoint]):
                nonlocal best
                if len(path) > len(best):
                    best = path.copy()
                for j in self._graph.get(link, []):
                    path.append(j)
                    dfs(j.child, path)
                    path.pop()

            dfs(base, [])
            return best
        # else find path to end_link
        res: List[URDFJoint] = []
        found = False

        def dfs2(link: str, path: List[URDFJoint]):
            nonlocal found, res
            if found:
                return
            if link == end_link:
                res = path.copy()
                found = True
                return
            for j in self._graph.get(link, []):
                path.append(j)
                dfs2(j.child, path)
                path.pop()

        dfs2(base, [])
        return res

    # ---------------- Public API -----------------
    def num_dof(self) -> int:
        """Return number of actuated joints.

        :return: dof.
        """
        return len(self._actuated)

    def joint_names(self) -> List[str]:
        """Actuated joint names.

        :return: names.
        """
        return [j.name for j in self._actuated]

    @property
    def joint_limits(self) -> List[Optional[Sequence[Optional[float]]]]:
        """Return joint limits for all actuated joints.

        :return: list of (lower, upper) limits.
        """
        return [j.limit for j in self._actuated]

    def name_to_index(self) -> Dict[str, int]:
        """Map joint name to index.

        :return: mapping.
        """
        return {j.name: j.index for j in self._actuated}

    # ------------- Kinematics ---------------------
    def fk(self, q: Sequence[float] | Any, *, backend: str = 'auto', return_end: bool = False,
           device: Any | None = None, dtype: Any | None = None) -> Dict[str, Any] | Any:
        """Compute forward kinematics.

        :param q: joint configuration length = dof.
        :param backend: 'auto'|'numpy'|'torch'
        :param return_end: if True, return only end-effector pose.
        :param device: torch device (if backend='torch')
        :param dtype: torch dtype (if backend='torch')
        :return: dict link_name -> 4x4 pose matrix or single 4x4 pose if return_end=True
        """
        if len(q) != self.num_dof():
            raise ValueError("Expected q of length %d" % self.num_dof())
        return forward_kinematics(
            self,
            q,
            backend=backend,
            return_end=return_end,
            device=device,
            dtype=dtype
        )

    def ik(self, target_pose: List[List[float]], q_initial: Optional[Sequence[float]] = None,
           backend: str = 'auto', method: str = 'pinv', max_iters: int = 120,
           pos_tol: float = 1e-4, ori_tol: float = 1e-4, multi_start: int = 0,
           multi_noise: float = 0.3, random_seed: Optional[int] = None,
           torch_device: Optional[str] = None, torch_dtype: Optional[Any] = None,
           **solver_kwargs) -> Dict[str, Any]:
        """Compute IK for the robot model.
        :param target_pose: 4x4 target pose as nested list.
        :param q_initial: initial guess (if None, uses zero vector).
        :param backend: 'auto'|'numpy'|'torch'
        :param method: 'pinv'|'dls'|'transpose'
        :param max_iters: maximum iterations.
        :param pos_tol: position tolerance (meters).
        :param ori_tol: orientation tolerance (radians).
        :param multi_start: extra random restarts count (0 disable)
        :param multi_noise: gaussian noise scale (radians) for restarts
        :param random_seed: seed for reproducibility
        :param torch_device: specify torch device when backend='torch' (e.g. 'cpu' or 'cuda')
        :param torch_dtype: specify torch dtype (e.g. torch.float32) when backend='torch'
        :param solver_kwargs: additional solver parameters.
        :return: dict with keys 'q', 'success', 'pos_err', 'ori_err', 'iters'
        """
        if q_initial is None:
            q_initial = [0.0] * self.num_dof()
        if len(q_initial) != self.num_dof():
            raise ValueError("Expected initial q of length %d" % self.num_dof())
        return inverse_kinematics(
            self,
            target_pose,
            q_initial,
            backend=backend,
            method=method,
            max_iters=max_iters,
            pos_tol=pos_tol,
            ori_tol=ori_tol,
            multi_start=multi_start,
            multi_noise=multi_noise,
            random_seed=random_seed,
            torch_device=torch_device,
            torch_dtype=torch_dtype,
            **solver_kwargs
        )

    def jacobian(self, q: Sequence[float] | Any, *, backend: str = 'auto', method: str = 'analytic',
                 epsilon: float = 5e-5, use_central_diff: bool = True,
                 device: Any | None = None, dtype: Any | None = None,
                 target_link: str | None = None,
                 joint_indices: Sequence[int] | None = None,
                 row_mask: Sequence[int | bool] | None = None) -> Any:
        """Compute 6×n geometric Jacobian matrix.
        The Jacobian relates joint velocities to end-effector spatial velocity
        (linear + angular). Uses axis-angle representation for orientation.
        :param q: joint configuration of length = dof.
        :param backend: 'auto'|'numpy'|'torch'
        :param method: 'analytic'|'numeric'|'autograd'
        :param epsilon: finite-difference step size (numeric method only).
        :param use_central_diff: use central differences for numeric method (more accurate than forward).
        :param device: torch device for torch backend (e.g., 'cpu', 'cuda').
        :param dtype: torch dtype for torch backend. Defaults to float64 if omitted.
        :return: 6×n Jacobian matrix (numpy.ndarray or torch.Tensor).
        """
        if len(q) != self.num_dof():
            raise ValueError("Expected q of length %d" % self.num_dof())
        return jacobian(
            self,
            q,
            backend=backend,
            method=method,
            epsilon=epsilon,
            use_central_diff=use_central_diff,
            device=device,
            dtype=dtype,
            target_link=target_link,
            joint_indices=joint_indices,
            row_mask=row_mask,
        )

    def random_q(self, rng=None, scale: float = 0.5):
        """
        Generate a random joint configuration within joint limits.
        
        :param rng: NumPy random generator (if None, creates a new one with random seed)
        :param scale: scaling factor for the joint range (0.0 to 1.0, default: 0.5)
                      0.5 means sample from middle 50% of each joint's range
        :return: list of random joint values (length = dof())
        
        Example::
        
            >>> model = RobotModel("robot.urdf")
            >>> q = model.random_q()  # Random configuration
            >>> q = model.random_q(scale=0.8)  # Use 80% of joint range
            >>> rng = np.random.default_rng(42)
            >>> q = model.random_q(rng=rng)  # Reproducible random
        """
        if rng is None:
            rng = np.random.default_rng()
        q = [0.0] * self.num_dof()
        for js in self._actuated:
            lo, hi = -1.0, 1.0
            if js.limit:
                if js.limit[0] is not None:
                    lo = js.limit[0]
                if js.limit[1] is not None:
                    hi = js.limit[1]
            mid = 0.5 * (lo + hi)
            span = 0.5 * (hi - lo) * scale
            q[js.index] = float(rng.uniform(mid - span, mid + span))
        return q

    # -------- Multi-chain support / spawn ---------
    def spawn_chain(self, end_link: str) -> "RobotModel":
        """Create a lightweight chain-specific view sharing the same parsed URDF.

        :param end_link: target end link in the original kinematic tree.
        :return: new RobotModel instance whose DOF/order corresponds to the chain from base_link to end_link.
        """
        return RobotModel(self.file_path, end_link=end_link, _parsed=self._parsed_source)

    def available_leaf_links(self) -> List[str]:
        """Return leaf links (no outgoing joints) in the full parsed tree.

        Useful to discover multiple end-effectors (e.g., left/right grippers) for spawn_chain.
        """
        parents = set(j.parent for j in self._raw_joints)
        children = set(j.child for j in self._raw_joints)
        # Leaf = appears as child but never as parent
        return sorted(list(children - parents))

    # -------- Multi-link groups and whole-body kinematics ---------
    def add_groups(self, groups: Dict[str, str]) -> Dict[str, "RobotModel"]:
        """
        :param groups: Mapping group_name -> end_link
        :return: Mapping group_name -> spawned RobotModel
        """
        for name, end in groups.items():
            self._groups[name] = self.spawn_chain(end)
        return dict(self._groups)

    def group(self, name: str) -> Dict[str, Any]:
        """
        :param name: Group name
        :return: Group info {joint_indices, end_link, model}
        """
        if not self._groups or name not in self._groups:
            raise ValueError(f"Group '{name}' not found. Available: {list(self._groups.keys())}")

        group_model = self._groups[name]
        return {
            'joint_indices': list(range(len(group_model.joint_names()))),
            'end_link': group_model.end_link,
            'model': group_model
        }

    def groups(self) -> Dict[str, "RobotModel"]:
        """
        :return: Current group mapping name -> RobotModel
        """
        return dict(self._groups)

    def ik_tasks(self, tasks: List[Union[Dict[str, Any], Any]], q0_by_group: Dict[str, Sequence[float]], *,
                 mode: str = 'weighted', max_iters: int = 100, tol: float = 1e-3,
                 damping: float = 1e-3, step_limit: float = 0.2, backend: str = 'auto',
                 verbose: bool = False) -> Dict[str, Any]:
        """
        :param tasks: List of task dictionaries or Task objects
        :param q0_by_group: Initial joint configurations by group
        :param mode: 'weighted'|'hierarchical'
        :param max_iters: Maximum iterations
        :param tol: Convergence tolerance
        :param damping: DLS damping factor
        :param step_limit: Maximum step size
        :param backend: Backend for computation
        :param verbose: Print progress
        :return: Solution results
        """
        if not self._groups:
            raise ValueError("No groups defined. Call add_groups first.")

        # Convert Task objects to dictionaries
        task_dicts = []
        for task in tasks:
            if hasattr(task, 'type'):  # Task object
                task_dict = {
                    'type': task.type,
                    'group': task.group,
                    'group_a': task.group_a,
                    'group_b': task.group_b,
                    'target': task.target,
                    'weight': task.weight,
                    'priority': task.priority,
                    'row_mask': task.row_mask,
                    'joint_indices': task.joint_indices
                }
                task_dicts.append(task_dict)
            else:  # Dictionary
                task_dicts.append(task)

        b = get_backend() if backend == 'auto' else backend
        
        # Use new multi-chain solver
        from robocore.kinematics.solvers.multi_chain_solver import MultiChainIKSolver
        from robocore.kinematics.task import Task
        
        # Convert dicts to Task objects
        task_objs = []
        for td in task_dicts:
            task_objs.append(Task(
                type=td.get('type'),
                group=td.get('group'),
                group_a=td.get('group_a'),
                group_b=td.get('group_b'),
                target=td.get('target'),
                weight=td.get('weight', 1.0),
                priority=td.get('priority', 0),
                row_mask=td.get('row_mask'),
                joint_indices=td.get('joint_indices')
            ))
        
        solver = MultiChainIKSolver(self._groups)
        
        if mode == 'weighted':
            return solver.solve_weighted(
                task_objs, q0_by_group, max_iters=max_iters, tol=tol,
                damping=damping, step_limit=step_limit, backend=b, verbose=verbose
            )
        elif mode == 'hierarchical':
            # Organize by priority
            from robocore.kinematics.task import organize_by_priority
            task_groups = organize_by_priority(task_objs)
            
            return solver.solve_hierarchical(
                task_groups, q0_by_group, max_iters=max_iters, tol=tol,
                damping=damping, step_limit=step_limit, backend=b, verbose=verbose
            )
        else:
            raise ValueError("Unknown mode, expected 'weighted'|'hierarchical'")

    def multi_task_ik_weighted(self,
                               tasks: List[Dict[str, Any]],
                               q0_by_group: Dict[str, Sequence[float]],
                               *,
                               max_iters: int = 100,
                               tol: float = 1e-3,
                               damping: float = 1e-3,
                               step_limit: float = 0.2,
                               backend: str = 'numpy',
                               verbose: bool = False) -> Dict[str, Any]:
        """
        :param tasks: List of task dicts. Absolute: {'type':'absolute','group':name,'target':T,'weight':w,'row_mask':[...]} Relative: {'type':'relative','group_a':A,'group_b':B,'target':T_rel,'weight':w,'row_mask':[...]}
        :param q0_by_group: Mapping name -> initial q
        :param max_iters: Max iterations
        :param tol: Convergence tolerance
        :param damping: DLS damping
        :param step_limit: Joint step limit
        :param backend: Backend for per-chain ops
        :param verbose: Print iteration logs
        :return: {'q_by_group', 'success', 'iters', 'residual'}
        """
        if not self._groups:
            raise ValueError("No groups defined. Call add_groups first.")
        names = list(self._groups.keys())
        # Build concatenated q vector
        q_by = {k: np.array(q0_by_group[k], dtype=float) for k in names}
        n_by = {k: len(q_by[k]) for k in names}
        idx_by = {}
        offset = 0
        for k in names:
            idx_by[k] = (offset, offset + n_by[k])
            offset += n_by[k]
        n_total = offset
        q = np.concatenate([q_by[k] for k in names])

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
                if task.get('type') == 'absolute':
                    g = task['group']
                    model = self._groups[g]
                    qg = slice_group(q, g)
                    T_cur = model.fk(qg, backend=backend, return_end=True)
                    T_cur = T_cur.detach().cpu().numpy() if hasattr(T_cur, 'detach') else np.array(T_cur)
                    e = self._pose_error_np(T_cur, np.array(task['target']))
                    Jg = jacobian(model, qg, backend=backend)
                    Jg = Jg.detach().cpu().numpy() if hasattr(Jg, 'detach') else np.array(Jg)
                    # pad into whole vector
                    Jpad = np.zeros((6, n_total))
                    i0, i1 = idx_by[g]
                    Jpad[:, i0:i1] = Jg
                    # row mask
                    mask = task.get('row_mask')
                    if mask is not None:
                        m = np.array([bool(x) for x in mask])
                        e = e[m]
                        Jpad = Jpad[m, :]
                    w = float(task.get('weight', 1.0)) ** 0.5
                    errs.append(w * e)
                    Jrows.append(w * Jpad)
                elif task.get('type') == 'relative':
                    ga = task['group_a']
                    gb = task['group_b']
                    ma = self._groups[ga]
                    mb = self._groups[gb]
                    qa = slice_group(q, ga)
                    qb = slice_group(q, gb)
                    Ta = ma.fk(qa, backend=backend, return_end=True)
                    Tb = mb.fk(qb, backend=backend, return_end=True)
                    if hasattr(Ta, 'detach'):
                        Ta = Ta.detach().cpu().numpy()
                        Tb = Tb.detach().cpu().numpy()
                    Ta = np.array(Ta)
                    Tb = np.array(Tb)
                    e = relative_pose_error(Ta, Tb, np.array(task['target']))
                    Jr = relative_jacobian(ma, mb, qa, qb, backend=backend)
                    Jr = Jr.detach().cpu().numpy() if hasattr(Jr, 'detach') else np.array(Jr)
                    # place Jr into columns of a+b
                    Jpad = np.zeros((Jr.shape[0], n_total))
                    i0a, i1a = idx_by[ga]
                    i0b, i1b = idx_by[gb]
                    Jpad[:, i0a:i1a] = Jr[:, :n_by[ga]]
                    Jpad[:, i0b:i1b] = Jr[:, n_by[ga]:]
                    mask = task.get('row_mask')
                    if mask is not None:
                        m = np.array([bool(x) for x in mask])
                        e = e[m]
                        Jpad = Jpad[m, :]
                    w = float(task.get('weight', 1.0)) ** 0.5
                    errs.append(w * e)
                    Jrows.append(w * Jpad)
                else:
                    raise ValueError("Unknown task type")

            e_total = np.concatenate(errs) if errs else np.zeros(0)
            if e_total.size == 0:
                break
            J_total = np.vstack(Jrows)
            JT = J_total.T
            A = J_total @ JT + (damping ** 2) * np.eye(J_total.shape[0])
            dq = JT @ np.linalg.solve(A, e_total)
            dq = np.clip(dq, -step_limit, step_limit)
            q += dq
            if verbose:
                print(f"Iter {it+1}: residual={np.linalg.norm(e_total):.6f}, |dq|={np.linalg.norm(dq):.6f}")
            if np.linalg.norm(e_total) < tol:
                break

        # Split back
        q_out = {}
        for name in names:
            q_out[name] = slice_group(q, name).tolist()
        return {
            'q_by_group': q_out,
            'success': True,
            'iters': it + 1,
            'residual': float(np.linalg.norm(e_total)) if e_total.size else 0.0,
        }

    @staticmethod
    def _pose_error_np(T_current: np.ndarray, T_target: np.ndarray) -> np.ndarray:
        """
        :param T_current: Current pose 4x4
        :param T_target: Target pose 4x4
        :return: 6D pose error
        """
        p_c = T_current[0:3, 3]
        p_t = T_target[0:3, 3]
        e_pos = p_t - p_c
        R_c = T_current[0:3, 0:3]
        R_t = T_target[0:3, 0:3]
        R_err = R_t @ R_c.T
        angle = np.arccos(max(-1.0, min(1.0, (np.trace(R_err) - 1.0) * 0.5)))
        if angle < 1e-12:
            e_ori = np.zeros(3)
        else:
            wx = R_err[2, 1] - R_err[1, 2]
            wy = R_err[0, 2] - R_err[2, 0]
            wz = R_err[1, 0] - R_err[0, 1]
            axis = np.array([wx, wy, wz]) / (2.0 * np.sin(angle) + 1e-12)
            e_ori = axis * angle
        return np.concatenate([e_pos, e_ori])

    def random_q_batch(self, batch_size: int, seed: int = None, scale: float = 0.5):
        """
        Generate a batch of random joint configurations within joint limits.
        
        :param batch_size: number of configurations to generate
        :param seed: random seed for reproducibility (if None, uses random seed)
        :param scale: scaling factor for the joint range (0.0 to 1.0, default: 0.5)
        :return: NumPy array of shape (batch_size, dof())
        
        Example::
        
            >>> model = RobotModel("robot.urdf")
            >>> q_batch = model.random_q_batch(100)  # 100 random configs
            >>> q_batch = model.random_q_batch(100, seed=42)  # Reproducible
            >>> q_batch = model.random_q_batch(100, scale=0.8)  # Use 80% of range
        """
        rng = np.random.default_rng(seed)
        n_joints = self.num_dof()
        q_batch = np.zeros((batch_size, n_joints))

        for i in range(batch_size):
            for js in self._actuated:
                lo, hi = -1.0, 1.0
                if js.limit:
                    if js.limit[0] is not None:
                        lo = js.limit[0]
                    if js.limit[1] is not None:
                        hi = js.limit[1]
                mid = 0.5 * (lo + hi)
                span = 0.5 * (hi - lo) * scale
                q_batch[i, js.index] = rng.uniform(mid - span, mid + span)

        return q_batch

    # -------- Workspace Analysis (lazy-loaded) ---------
    def compute_workspace(self, num_samples: int = 5000, method: str = 'monte_carlo',
                          force_recompute: bool = False, verbose: bool = True) -> np.ndarray:
        """Compute and cache reachable workspace for this chain.

        This method samples the joint space and computes FK to build a point cloud
        representing the end-effector's reachable positions. The result is cached
        for fast subsequent reachability checks.

        :param num_samples: number of random joint configurations to sample
        :param method: sampling method ('monte_carlo', 'grid', 'sobol')
        :param force_recompute: if True, ignore cached workspace and recompute
        :param verbose: print progress messages
        :return: numpy array of shape (num_samples, 3) containing reachable points
        """
        if not force_recompute and self._workspace_points is not None:
            if verbose:
                beauty_print(f"Using cached workspace ({len(self._workspace_points)} points)", type="info")
            return self._workspace_points

        if verbose:
            beauty_print(f"Computing workspace for {self.end_link} ({num_samples} samples)...", type="info")

        # Lazy-load analyzer
        if self._workspace_analyzer is None:
            WorkspaceAnalyzer = _get_workspace_analyzer()
            self._workspace_analyzer = WorkspaceAnalyzer(self, backend='numpy')

        # Compute workspace
        points = self._workspace_analyzer.compute_reachable_workspace(
            num_samples=num_samples,
            method=method,
            use_parallel=False,  # Keep simple for MVP
            seed=None
        )

        # Cache results
        self._workspace_points = points
        self._workspace_bounds = self._workspace_analyzer.get_workspace_bounds(points)

        # Build KDTree for fast lookups
        try:
            from scipy.spatial import cKDTree
            self._workspace_kdtree = cKDTree(points)
            if verbose:
                beauty_print(
                    f"✓ Workspace cached: {len(points)} points, bounds={self._workspace_bounds}", type="success")
        except ImportError:
            if verbose:
                beauty_print("⚠️ scipy not available, reachability checks will be slower", type="warning")

        return points

    def is_point_reachable(self, point: np.ndarray | list, tolerance: float = 0.05,
                           auto_compute: bool = True, num_samples: int = 5000) -> bool:
        """Check if a 3D point is within the reachable workspace.

        :param point: 3D position [x, y, z]
        :param tolerance: distance threshold in meters (default 5cm)
        :param auto_compute: if True and workspace not cached, compute it automatically
        :param num_samples: samples to use if auto-computing workspace
        :return: True if point is reachable (within tolerance of cached workspace)
        """
        point = np.asarray(point).flatten()[:3]

        # Auto-compute workspace if needed
        if self._workspace_points is None:
            if auto_compute:
                self.compute_workspace(num_samples=num_samples, verbose=False)
            else:
                raise ValueError("Workspace not computed. Call compute_workspace() first or set auto_compute=True.")

        # Fast bounding box check first
        if self._workspace_bounds is not None:
            margin = tolerance
            if not (self._workspace_bounds['x'][0] - margin <= point[0] <= self._workspace_bounds['x'][1] + margin and
                    self._workspace_bounds['y'][0] - margin <= point[1] <= self._workspace_bounds['y'][1] + margin and
                    self._workspace_bounds['z'][0] - margin <= point[2] <= self._workspace_bounds['z'][1] + margin):
                return False

        # KDTree query for precise check
        if self._workspace_kdtree is not None:
            dist, _ = self._workspace_kdtree.query(point)
            return dist <= tolerance
        else:
            # Fallback: brute-force distance check
            dists = np.linalg.norm(self._workspace_points - point, axis=1)
            return np.min(dists) <= tolerance

    def get_workspace_bounds(self) -> Dict[str, Tuple[float, float]] | None:
        """Return cached workspace bounding box or None if not computed."""
        return self._workspace_bounds

    def summary(self, show_chain: bool = False, title: str = "Robot Model Summary"):
        """Print a concise summary of the robot model.

        :param show_chain: Whether to print internal actuated chain details
        :param title: Custom title for the summary
        """
        beauty_print(title, type="module", centered=True)
        beauty_print(f"Name: {self.name}")
        beauty_print(f"File: {self.file_path}")
        beauty_print(f"DOF: {self.num_dof()}  |  End Link: {self.end_link}")
        beauty_print(f"Base Link: {self.base_link}")
        beauty_print(f"Actuated Joints: {self.joint_names()}")

        if show_chain:
            beauty_print("Actuated Chain Details:")
            for j in self._actuated:
                limit_str = f"[{j.limit[0]:.3f}, {j.limit[1]:.3f}]" if j.limit and j.limit[0] is not None else "unlimited"
                beauty_print(
                    f"  [{j.index}] {j.name} ({j.joint_type})\n"
                    f"      parent: {j.parent} -> child: {j.child}\n"
                    f"      axis: {beauty_print_array(j.axis)}  limits: {limit_str}"
                )

    def print_tree(self, show_joints: bool = True, show_fixed: bool = False):
        """Print kinematic tree structure showing body/link connections.
        
        :param show_joints: Whether to include joint names/types in the tree
        :param show_fixed: Whether to include fixed joints in the tree
        """
        beauty_print("Kinematic Tree Structure", type="module", centered=True)
        # Build a complete parent-child graph for tree visualization
        visited = set()
        chain_child_links = {j.child for j in self._chain_joints}
        # Choose an even lighter pastel green; fallback to basic bright green if 256-color unsupported.
        import os
        term = os.environ.get("TERM", "")
        if "256" in term or "xterm" in term or "screen" in term:
            GREEN = "\033[38;5;157m"  # very light mint green
        else:
            GREEN = "\033[92m"  # basic bright green fallback
        RESET = "\033[0m"

        def print_subtree(link: str, prefix: str = "", is_last: bool = True):
            """Recursively print tree structure.

            When show_joints is False only link hierarchy is shown (no joint rows).
            """
            if link in visited:
                return
            visited.add(link)

            # Determine connector symbols (for this link relative to its parent)
            connector = "└── " if is_last else "├── "
            extension = "    " if is_last else "│   "

            # Whether this link is on active chain
            link_on_chain = (link == self.base_link) or (link in chain_child_links)

            # Print current link
            if link == self.base_link:
                base_label = f"{link} (base)"
                if link_on_chain:
                    base_label = f"{GREEN}{base_label}{RESET}"
                beauty_print(base_label)
            else:
                line = f"{prefix}{connector}{link}"
                if link_on_chain:
                    line = f"{GREEN}{line}{RESET}"
                print(line)

            # Children (full list for traversal)
            children_all = self._graph.get(link, [])

            if show_joints:
                # Visible joints for printing
                if show_fixed:
                    children_print = children_all
                else:
                    children_print = [j for j in children_all if j.joint_type in ("revolute", "prismatic")]

                # Print visible joints
                for i, joint in enumerate(children_print):
                    is_last_child = (i == len(children_print) - 1)
                    joint_symbol = "⚙" if joint.joint_type in ("revolute", "prismatic") else "⊗"
                    joint_prefix = prefix + extension
                    joint_connector = "└── " if is_last_child else "├── "
                    in_chain = any(j.name == joint.name for j in self._chain_joints)
                    joint_line = f"{joint_prefix}{joint_connector}{joint_symbol} {joint.name} ({joint.joint_type})"
                    if in_chain:
                        joint_line = f"{GREEN}{joint_line}{RESET}"
                    print(joint_line)
                    child_prefix = prefix + extension + ("    " if is_last_child else "│   ")
                    print_subtree(joint.child, child_prefix, True)

                # Recurse through hidden (filtered-out) joints so deeper actuated joints are not lost
                hidden = [j for j in children_all if j not in children_print]
                for hidden_joint in hidden:
                    # Do not alter prefix depth since no joint line printed
                    print_subtree(hidden_joint.child, prefix + extension, False)
            else:
                # Link-only view: consider all joints to derive child links (including fixed)
                child_links = [j.child for j in children_all]
                for i, child_link in enumerate(child_links):
                    is_last_child = (i == len(child_links) - 1)
                    next_prefix = prefix + extension
                    print_subtree(child_link, next_prefix, is_last_child)

        print_subtree(self.base_link)

        # Legend
        beauty_print("\nLegend:")
        if show_joints:
            print("  ⚙  = Actuated joint (revolute/prismatic)")
            print("  ⊗  = Fixed joint")
            print("  Green = Active chain (links & joints) to selected end-effector")
        else:
            print("  Green = Active chain (links) to selected end-effector")


class BimanualRobotModel(RobotModel):
    """Bimanual robot model with automatic left/right arm groups.
    
    Extends RobotModel to provide dual-arm kinematics with automatic group management.
    The fk/ik/jacobian methods operate on both arms simultaneously.
    
    :param file_path: URDF or MJCF file path
    :param left_end_link: Left arm end-effector link name
    :param right_end_link: Right arm end-effector link name
    """

    def __init__(self, file_path: str | Path, left_end_link: str, right_end_link: str):
        """Initialize bimanual robot model.
        
        :param file_path: Path to URDF or MJCF file
        :param left_end_link: Left arm end-effector link name
        :param right_end_link: Right arm end-effector link name
        """
        # Initialize base RobotModel (full kinematic tree)
        super().__init__(file_path, end_link=None)

        # Create left and right arm groups
        self.add_groups({
            'left_arm': left_end_link,
            'right_arm': right_end_link
        })

        self.left_model = self._groups['left_arm']
        self.right_model = self._groups['right_arm']
        self.left_end_link = left_end_link
        self.right_end_link = right_end_link

        beauty_print(f"✓ Bimanual robot initialized:", type="success")
        beauty_print(f"  Left arm: {self.left_model.num_dof()} DOF, end: {left_end_link}")
        beauty_print(f"  Right arm: {self.right_model.num_dof()} DOF, end: {right_end_link}")

    def fk(self, q_left: Sequence[float], q_right: Sequence[float],
           *, backend: str = 'auto', mode: str = 'indep', **kwargs) -> Dict[str, Any]:
        """Compute forward kinematics for both arms.
        
        :param q_left: Left arm joint configuration
        :param q_right: Right arm joint configuration
        :param backend: 'auto'|'numpy'|'torch'
        :param mode: 'indep'|'relative'|'mirror'
        :return: Dict with 'left' and 'right' end-effector poses
        """
        from robocore.kinematics.bimanual import bimanual_forward_kinematics
        return bimanual_forward_kinematics(
            self.left_model, self.right_model,
            q_left, q_right,
            backend=backend, mode=mode, **kwargs
        )

    def ik(self, target_left=None, target_right=None,
           q0_left: Optional[Sequence[float]] = None,
           q0_right: Optional[Sequence[float]] = None,
           *, backend: str = 'auto', method: str = 'dls',
           coordination: str = 'indep',
           T_rel_grasp=None, T_left_initial=None, T_right_initial=None,
           **kwargs) -> Dict[str, Any]:
        """Compute inverse kinematics for both arms.
        
        :param target_left: Left arm target pose (4x4)
        :param target_right: Right arm target pose (4x4)
        :param q0_left: Initial left configuration
        :param q0_right: Initial right configuration
        :param backend: 'auto'|'numpy'|'torch'
        :param method: 'dls'|'pinv'|'transpose'
        :param coordination: 'indep'|'relative_pose'|'mirror'
        :param T_rel_grasp: Relative grasp transform (for relative_pose mode)
        :param T_left_initial: Left initial reference (for mirror mode)
        :param T_right_initial: Right initial reference (for mirror mode)
        :return: Dict with 'q_left', 'q_right', 'success_left', 'success_right'
        """
        from robocore.kinematics.bimanual import bimanual_inverse_kinematics

        if q0_left is None:
            q0_left = [0.0] * self.left_model.num_dof()
        if q0_right is None:
            q0_right = [0.0] * self.right_model.num_dof()

        return bimanual_inverse_kinematics(
            self.left_model, self.right_model,
            target_left=target_left, target_right=target_right,
            q0_left=q0_left, q0_right=q0_right,
            backend=backend, method=method, coordination=coordination,
            T_rel_grasp=T_rel_grasp,
            T_left_initial=T_left_initial,
            T_right_initial=T_right_initial,
            **kwargs
        )

    def jacobian(self, q_left: Sequence[float], q_right: Sequence[float],
                 *, backend: str = 'auto', mode: str = 'indep', **kwargs) -> Any:
        """Compute Jacobian for both arms.
        
        :param q_left: Left arm joint configuration
        :param q_right: Right arm joint configuration
        :param backend: 'auto'|'numpy'|'torch'
        :param mode: 'indep'|'relative'
        :return: Block-diagonal or relative Jacobian matrix
        """
        from robocore.kinematics.bimanual import bimanual_jacobian
        return bimanual_jacobian(
            self.left_model, self.right_model,
            q_left, q_right,
            backend=backend, mode=mode, **kwargs
        )

    def block_jacobian(self, q_by_group: Dict[str, Sequence[float]], *, backend: str = 'auto') -> np.ndarray:
        """
        :param q_by_group: Mapping name -> joint vector
        :param backend: 'auto'|'numpy'|'torch'
        :return: Block-diagonal Jacobian for all groups stacked as 6*k rows
        """
        if not self._groups:
            raise ValueError("No groups defined. Call add_groups first.")

        b = get_backend() if backend == 'auto' else backend
        if b == 'numpy':
            from robocore.kinematics.jacobian_utils.bimanual_jacobian_solver_numpy import BiMultiLinkJacobianSolverNumpy
            solver = BiMultiLinkJacobianSolverNumpy(self._groups)
            return solver.block_jacobian(q_by_group, backend='numpy')
        elif b == 'torch':
            from robocore.kinematics.jacobian_utils.bimanual_jacobian_solver_torch import BiMultiLinkJacobianSolverTorch
            solver = BiMultiLinkJacobianSolverTorch(self._groups)
            return solver.block_jacobian(q_by_group, backend='torch')
        else:
            raise ValueError("Unsupported backend, expected 'auto'|'numpy'|'torch'")
    
    def relative_jacobian_between(self, group_a: str, group_b: str,
                                      q_a: Sequence[float], q_b: Sequence[float], *,
                                      backend: str = 'auto') -> np.ndarray:
        """
        :param group_a: First group name
        :param group_b: Second group name
        :param q_a: Joint vector of group_a
        :param q_b: Joint vector of group_b
        :param backend: 'auto'|'numpy'|'torch'
        :return: 6 x (n_a + n_b) relative Jacobian (pose)
        """
        if not self._groups:
            raise ValueError("No groups defined. Call add_groups first.")

        b = get_backend() if backend == 'auto' else backend
        if b == 'numpy':
            from robocore.kinematics.jacobian_utils.bimanual_jacobian_solver_numpy import BiMultiLinkJacobianSolverNumpy
            solver = BiMultiLinkJacobianSolverNumpy(self._groups)
            return solver.relative_jacobian_between(group_a, group_b, q_a, q_b, backend='numpy')
        elif b == 'torch':
            from robocore.kinematics.jacobian_utils.bimanual_jacobian_solver_torch import BiMultiLinkJacobianSolverTorch
            solver = BiMultiLinkJacobianSolverTorch(self._groups)
            return solver.relative_jacobian_between(group_a, group_b, q_a, q_b, backend='torch')
        else:
            raise ValueError("Unsupported backend, expected 'auto'|'numpy'|'torch'")


__all__ = ["RobotModel", "BimanualRobotModel", "JointSpec"]