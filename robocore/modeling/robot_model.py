"""Robot model abstraction.

Loads from URDF (and later MJCF) and provides forward kinematics.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Any

from .parser.urdf_parser import load_urdf, URDFJoint
from .parser.mjcf_parser import load_mjcf
from robocore.kinematics.fk import forward_kinematics
from robocore.kinematics.ik import inverse_kinematics
from robocore.kinematics.jacobian import jacobian
import numpy as np

from robocore.utils.beauty_logger import beauty_print, beauty_print_array


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


class RobotModel:
    """Generic serial chain robot.

    :param file_path: URDF (or MJCF in future) path.
    :param end_link: override end-effector link name.
    """

    def __init__(self, file_path: str | Path, end_link: Optional[str] = None):
        """Initialize robot model.

        :param file_path: path to URDF or MJCF file.
        :param end_link: end-effector link name (auto-detect if None).
        """
        self.file_path = str(file_path)
        # Auto-detect format by file extension (simple heuristic). If '.xml' we try MJCF first.
        path_lower = str(self.file_path).lower()
        parsed = None
        if path_lower.endswith('.xml'):
            try:
                parsed = load_mjcf(self.file_path)
            except Exception as e:
                beauty_print(f"⚠️ MJCF parse failed ({e}); falling back to URDF parser", type="warning")
        if parsed is None:
            parsed = load_urdf(self.file_path)
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

    def forward_kinematics(self, q: Sequence[float], return_numpy: bool = True):
        """Legacy FK interface for backward compatibility with IK/Jacobian solvers.

        :param q: joint values with length dof().
        :param return_numpy: if True, return NumPy arrays; else convert to lists.
        :return: dict link->(4x4 pose matrix), end-effector pose under key 'end'.
        """
        return self.fk(q, backend='numpy', return_end=False)

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
                 device: Any | None = None, dtype: Any | None = None) -> Any:
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
            dtype=dtype
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

    def print_tree(self, show_fixed: bool = False):
        """Print kinematic tree structure showing body/link connections.

        :param show_fixed: Whether to include fixed joints in the tree
        """
        beauty_print("Kinematic Tree Structure", type="module", centered=True)

        # Build a complete parent-child graph for tree visualization
        visited = set()

        def print_subtree(link: str, prefix: str = "", is_last: bool = True):
            """Recursively print tree structure."""
            if link in visited:
                return
            visited.add(link)

            # Determine connector symbols
            connector = "└── " if is_last else "├── "
            extension = "    " if is_last else "│   "

            # Print current link
            if link == self.base_link:
                beauty_print(f"{link} (base)")
            else:
                print(f"{prefix}{connector}{link}")

            # Get children from graph
            children_joints = self._graph.get(link, [])

            # Filter based on show_fixed flag
            if not show_fixed:
                children_joints = [j for j in children_joints if j.joint_type in ("revolute", "prismatic")]

            # Print children
            for i, joint in enumerate(children_joints):
                is_last_child = (i == len(children_joints) - 1)

                # Print joint info
                joint_symbol = "⚙" if joint.joint_type in ("revolute", "prismatic") else "⊗"
                joint_prefix = prefix + extension
                joint_connector = "└── " if is_last_child else "├── "

                # Check if this joint is in the active chain
                in_chain = any(j.name == joint.name for j in self._chain_joints)
                chain_marker = " ★" if in_chain else ""

                print(f"{joint_prefix}{joint_connector}{joint_symbol} {joint.name} ({joint.joint_type}){chain_marker}")

                # Recursively print child link
                child_prefix = prefix + extension + ("    " if is_last_child else "│   ")
                print_subtree(joint.child, child_prefix, True)

        print_subtree(self.base_link)

        # Legend
        beauty_print("\nLegend:")
        print("  ⚙  = Actuated joint (revolute/prismatic)")
        print("  ⊗  = Fixed joint")
        print("  ★  = Part of active chain to end-effector")


__all__ = ["RobotModel", "JointSpec"]
