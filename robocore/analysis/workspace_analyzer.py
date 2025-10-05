"""Workspace Analysis Module

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
from typing import TYPE_CHECKING, Tuple, Optional, Dict, Any, List
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import warnings

if TYPE_CHECKING:
    from robocore.modeling.robot_model import RobotModel


class WorkspaceAnalyzer:
    """
    Analyze robot workspace characteristics.
    
    Provides methods to compute:
    - Reachable workspace: all points the end-effector can reach
    - Dexterous workspace: points reachable with arbitrary orientation
    - Workspace volume and boundaries
    - Singularity-free regions
    
    Examples
    --------
    >>> from robocore.modeling.robot_model import RobotModel
    >>> from robocore.analysis.workspace_analyzer import WorkspaceAnalyzer
    >>> 
    >>> model = RobotModel('robot.urdf')
    >>> analyzer = WorkspaceAnalyzer(model)
    >>> 
    >>> # Compute reachable workspace
    >>> points = analyzer.compute_reachable_workspace(
    ...     num_samples=10000,
    ...     method='monte_carlo'
    ... )
    >>> 
    >>> # Get workspace bounds
    >>> bounds = analyzer.get_workspace_bounds(points)
    >>> print(f"Workspace range: X={bounds['x']}, Y={bounds['y']}, Z={bounds['z']}")
    >>> 
    >>> # Estimate workspace volume
    >>> volume = analyzer.estimate_workspace_volume(points)
    >>> print(f"Workspace volume: {volume:.4f} m³")
    """
    
    def __init__(self, model: "RobotModel", backend: str = 'numpy'):
        """
        Initialize workspace analyzer.
        
        Parameters
        ----------
        model : RobotModel
            Robot model
        backend : str
            Computation backend ('numpy' or 'torch')
        """
        self.model = model
        self.backend = backend
        self.num_dof = model.num_dof()
        
        # Cache for workspace data
        self._workspace_cache: Dict[str, Any] = {}
    
    def compute_reachable_workspace(
        self,
        num_samples: int = 10000,
        method: str = 'monte_carlo',
        q_limits: Optional[np.ndarray] = None,
        use_parallel: bool = False,
        num_workers: Optional[int] = None,
        seed: Optional[int] = None
    ) -> np.ndarray:
        """
        Compute reachable workspace points.
        
        The reachable workspace is the set of all points in 3D space that
        the end-effector can reach with at least one orientation.
        
        Parameters
        ----------
        num_samples : int
            Number of random samples
        method : str
            Sampling method:
            - 'monte_carlo': Random uniform sampling
            - 'grid': Grid-based sampling
            - 'sobol': Sobol sequence (quasi-random)
        q_limits : np.ndarray, optional
            Joint limits, shape (dof, 2). Default: [-π, π] for each joint
        use_parallel : bool
            Use parallel processing (for large num_samples)
        num_workers : int, optional
            Number of parallel workers
        seed : int, optional
            Random seed for reproducibility
        
        Returns
        -------
        points : np.ndarray
            Reachable workspace points, shape (num_valid, 3)
        
        Examples
        --------
        >>> points = analyzer.compute_reachable_workspace(
        ...     num_samples=50000,
        ...     method='monte_carlo',
        ...     use_parallel=True
        ... )
        >>> print(f"Reachable points: {len(points)}")
        """
        from robocore.kinematics.fk import forward_kinematics
        
        if seed is not None:
            np.random.seed(seed)
        
        # Default joint limits
        if q_limits is None:
            q_limits = np.tile([-np.pi, np.pi], (self.num_dof, 1))
        
        # Generate samples
        if method == 'monte_carlo':
            q_samples = self._sample_monte_carlo(num_samples, q_limits)
        elif method == 'grid':
            q_samples = self._sample_grid(num_samples, q_limits)
        elif method == 'sobol':
            q_samples = self._sample_sobol(num_samples, q_limits)
        else:
            raise ValueError(f"Unknown sampling method: {method}")
        
        # Compute FK for all samples
        if use_parallel and num_samples > 1000:
            points = self._compute_fk_parallel(q_samples, num_workers)
        else:
            points = self._compute_fk_sequential(q_samples)
        
        # Cache results
        self._workspace_cache['reachable_points'] = points
        self._workspace_cache['method'] = method
        self._workspace_cache['num_samples'] = num_samples
        
        return points
    
    def compute_dexterous_workspace(
        self,
        num_samples: int = 10000,
        num_orientations: int = 8,
        q_limits: Optional[np.ndarray] = None,
        tolerance: float = 0.01,
        use_parallel: bool = False
    ) -> np.ndarray:
        """
        Compute dexterous workspace.
        
        The dexterous workspace contains points that can be reached with
        multiple (at least num_orientations) different orientations.
        
        Parameters
        ----------
        num_samples : int
            Number of position samples
        num_orientations : int
            Minimum number of orientations required
        q_limits : np.ndarray, optional
            Joint limits
        tolerance : float
            Position tolerance for grouping (meters)
        use_parallel : bool
            Use parallel processing
        
        Returns
        -------
        points : np.ndarray
            Dexterous workspace points
        
        Examples
        --------
        >>> # Points reachable with at least 8 different orientations
        >>> dex_points = analyzer.compute_dexterous_workspace(
        ...     num_samples=20000,
        ...     num_orientations=8
        ... )
        """
        from robocore.kinematics.fk import forward_kinematics
        
        # Generate diverse samples
        if q_limits is None:
            q_limits = np.tile([-np.pi, np.pi], (self.num_dof, 1))
        
        q_samples = self._sample_monte_carlo(num_samples, q_limits)
        
        # Compute FK
        points = self._compute_fk_sequential(q_samples)
        
        # Group points by position (cluster nearby points)
        from scipy.spatial import cKDTree
        tree = cKDTree(points)
        
        # Count orientations per position
        dexterous_mask = np.zeros(len(points), dtype=bool)
        
        for i, point in enumerate(points):
            # Find nearby points (same position, different orientations)
            nearby_indices = tree.query_ball_point(point, tolerance)
            if len(nearby_indices) >= num_orientations:
                dexterous_mask[i] = True
        
        dexterous_points = points[dexterous_mask]
        
        # Remove duplicates (keep unique positions)
        if len(dexterous_points) > 0:
            tree_dex = cKDTree(dexterous_points)
            unique_mask = np.ones(len(dexterous_points), dtype=bool)
            
            for i in range(len(dexterous_points)):
                if not unique_mask[i]:
                    continue
                nearby = tree_dex.query_ball_point(dexterous_points[i], tolerance)
                unique_mask[nearby[1:]] = False  # Keep first, remove rest
            
            dexterous_points = dexterous_points[unique_mask]
        
        self._workspace_cache['dexterous_points'] = dexterous_points
        
        return dexterous_points
    
    def get_workspace_bounds(
        self,
        points: Optional[np.ndarray] = None
    ) -> Dict[str, Tuple[float, float]]:
        """
        Get workspace bounding box.
        
        Parameters
        ----------
        points : np.ndarray, optional
            Workspace points. If None, use cached reachable workspace
        
        Returns
        -------
        bounds : dict
            Dictionary with keys 'x', 'y', 'z' containing (min, max) tuples
        
        Examples
        --------
        >>> bounds = analyzer.get_workspace_bounds()
        >>> print(f"X range: [{bounds['x'][0]:.3f}, {bounds['x'][1]:.3f}]")
        """
        if points is None:
            if 'reachable_points' not in self._workspace_cache:
                raise ValueError("No workspace points available. Run compute_reachable_workspace first.")
            points = self._workspace_cache['reachable_points']
        
        bounds = {
            'x': (np.min(points[:, 0]), np.max(points[:, 0])),
            'y': (np.min(points[:, 1]), np.max(points[:, 1])),
            'z': (np.min(points[:, 2]), np.max(points[:, 2]))
        }
        
        return bounds
    
    def estimate_workspace_volume(
        self,
        points: Optional[np.ndarray] = None,
        method: str = 'convex_hull'
    ) -> float:
        """
        Estimate workspace volume.
        
        Parameters
        ----------
        points : np.ndarray, optional
            Workspace points
        method : str
            Volume estimation method:
            - 'convex_hull': Convex hull volume (upper bound)
            - 'alpha_shape': Alpha shape volume (more accurate)
            - 'voxel': Voxel-based estimation
        
        Returns
        -------
        volume : float
            Estimated volume in m³
        
        Examples
        --------
        >>> volume = analyzer.estimate_workspace_volume(method='convex_hull')
        >>> print(f"Workspace volume: {volume:.4f} m³")
        """
        if points is None:
            if 'reachable_points' not in self._workspace_cache:
                raise ValueError("No workspace points available.")
            points = self._workspace_cache['reachable_points']
        
        if method == 'convex_hull':
            from scipy.spatial import ConvexHull
            try:
                hull = ConvexHull(points)
                return hull.volume
            except Exception as e:
                warnings.warn(f"ConvexHull failed: {e}. Returning bounding box volume.")
                return self._estimate_bounding_box_volume(points)
        
        elif method == 'voxel':
            return self._estimate_voxel_volume(points)
        
        elif method == 'alpha_shape':
            # Alpha shape requires additional dependencies
            warnings.warn("Alpha shape not implemented. Using convex hull.")
            return self.estimate_workspace_volume(points, method='convex_hull')
        
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def check_point_in_workspace(
        self,
        point: np.ndarray,
        points: Optional[np.ndarray] = None,
        tolerance: float = 0.05
    ) -> bool:
        """
        Check if a point is in the reachable workspace.
        
        Parameters
        ----------
        point : np.ndarray
            3D point to check
        points : np.ndarray, optional
            Workspace points
        tolerance : float
            Distance threshold (meters)
        
        Returns
        -------
        in_workspace : bool
            True if point is reachable
        
        Examples
        --------
        >>> target = np.array([0.5, 0.2, 0.3])
        >>> reachable = analyzer.check_point_in_workspace(target)
        """
        if points is None:
            if 'reachable_points' not in self._workspace_cache:
                raise ValueError("No workspace points available.")
            points = self._workspace_cache['reachable_points']
        
        from scipy.spatial import cKDTree
        tree = cKDTree(points)
        dist, _ = tree.query(point)
        
        return dist <= tolerance
    
    def compute_workspace_density(
        self,
        points: Optional[np.ndarray] = None,
        grid_resolution: int = 20
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute workspace density distribution.
        
        Returns a 3D grid showing point density in different regions.
        
        Parameters
        ----------
        points : np.ndarray, optional
            Workspace points
        grid_resolution : int
            Grid resolution per axis
        
        Returns
        -------
        density : np.ndarray
            3D density array, shape (grid_resolution, grid_resolution, grid_resolution)
        grid_coords : np.ndarray
            Grid coordinate arrays (x, y, z)
        
        Examples
        --------
        >>> density, (x, y, z) = analyzer.compute_workspace_density()
        >>> # Find highest density region
        >>> max_idx = np.unravel_index(np.argmax(density), density.shape)
        >>> print(f"Densest region: ({x[max_idx[0]]}, {y[max_idx[1]]}, {z[max_idx[2]]})")
        """
        if points is None:
            if 'reachable_points' not in self._workspace_cache:
                raise ValueError("No workspace points available.")
            points = self._workspace_cache['reachable_points']
        
        bounds = self.get_workspace_bounds(points)
        
        # Create 3D grid
        x_edges = np.linspace(bounds['x'][0], bounds['x'][1], grid_resolution + 1)
        y_edges = np.linspace(bounds['y'][0], bounds['y'][1], grid_resolution + 1)
        z_edges = np.linspace(bounds['z'][0], bounds['z'][1], grid_resolution + 1)
        
        # Compute histogram
        density, edges = np.histogramdd(
            points,
            bins=[x_edges, y_edges, z_edges]
        )
        
        # Grid centers
        x_centers = (x_edges[:-1] + x_edges[1:]) / 2
        y_centers = (y_edges[:-1] + y_edges[1:]) / 2
        z_centers = (z_edges[:-1] + z_edges[1:]) / 2
        
        return density, (x_centers, y_centers, z_centers)
    
    def find_singularity_free_regions(
        self,
        num_samples: int = 5000,
        manipulability_threshold: float = 0.01,
        q_limits: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Find workspace regions without singularities.
        
        Parameters
        ----------
        num_samples : int
            Number of samples
        manipulability_threshold : float
            Minimum manipulability measure
        q_limits : np.ndarray, optional
            Joint limits
        
        Returns
        -------
        points : np.ndarray
            Singularity-free workspace points
        
        Examples
        --------
        >>> safe_points = analyzer.find_singularity_free_regions(
        ...     manipulability_threshold=0.05
        ... )
        """
        from robocore.kinematics.fk import forward_kinematics
        from robocore.kinematics.jacobian import jacobian
        
        if q_limits is None:
            q_limits = np.tile([-np.pi, np.pi], (self.num_dof, 1))
        
        q_samples = self._sample_monte_carlo(num_samples, q_limits)
        
        safe_points = []
        
        for q in q_samples:
            # Compute manipulability
            J = jacobian(self.model, q, backend=self.backend)
            
            # Manipulability measure: sqrt(det(J @ J.T))
            if isinstance(J, np.ndarray):
                manipulability = np.sqrt(np.linalg.det(J @ J.T))
            else:
                import torch
                manipulability = torch.sqrt(torch.det(J @ J.T)).item()
            
            if manipulability >= manipulability_threshold:
                T = forward_kinematics(self.model, q, backend=self.backend, return_end=True)
                if isinstance(T, np.ndarray):
                    safe_points.append(T[:3, 3])
                else:
                    safe_points.append(T[:3, 3].cpu().numpy())
        
        if len(safe_points) == 0:
            warnings.warn("No singularity-free points found. Try lowering threshold.")
            return np.array([])
        
        return np.array(safe_points)
    
    def visualize_workspace(
        self,
        points: Optional[np.ndarray] = None,
        show_bounds: bool = True,
        show_density: bool = False,
        alpha: float = 0.3,
        figsize: Tuple[int, int] = (10, 8)
    ):
        """
        Visualize workspace in 3D.
        
        Parameters
        ----------
        points : np.ndarray, optional
            Workspace points to visualize
        show_bounds : bool
            Show bounding box
        show_density : bool
            Show density heatmap
        alpha : float
            Point transparency
        figsize : tuple
            Figure size
        
        Examples
        --------
        >>> analyzer.visualize_workspace(show_density=True)
        """
        try:
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D
        except ImportError:
            raise ImportError("matplotlib required for visualization")
        
        if points is None:
            if 'reachable_points' not in self._workspace_cache:
                raise ValueError("No workspace points available.")
            points = self._workspace_cache['reachable_points']
        
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot points
        ax.scatter(points[:, 0], points[:, 1], points[:, 2],
                  c='b', marker='.', alpha=alpha, s=1)
        
        # Bounding box
        if show_bounds:
            bounds = self.get_workspace_bounds(points)
            self._plot_bounding_box(ax, bounds)
        
        # Density
        if show_density:
            density, (x, y, z) = self.compute_workspace_density(points)
            # Plot highest density voxels
            threshold = np.percentile(density.flatten(), 90)
            high_density = density > threshold
            
            for i in range(len(x)):
                for j in range(len(y)):
                    for k in range(len(z)):
                        if high_density[i, j, k]:
                            ax.scatter(x[i], y[j], z[k], 
                                     c='r', marker='s', s=20, alpha=0.5)
        
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title('Robot Workspace')
        
        # Equal aspect ratio
        bounds = self.get_workspace_bounds(points)
        max_range = max(
            bounds['x'][1] - bounds['x'][0],
            bounds['y'][1] - bounds['y'][0],
            bounds['z'][1] - bounds['z'][0]
        )
        
        mid_x = (bounds['x'][1] + bounds['x'][0]) / 2
        mid_y = (bounds['y'][1] + bounds['y'][0]) / 2
        mid_z = (bounds['z'][1] + bounds['z'][0]) / 2
        
        ax.set_xlim(mid_x - max_range/2, mid_x + max_range/2)
        ax.set_ylim(mid_y - max_range/2, mid_y + max_range/2)
        ax.set_zlim(mid_z - max_range/2, mid_z + max_range/2)
        
        plt.tight_layout()
        plt.show()
    
    # ========== Private Methods ==========
    
    def _sample_monte_carlo(
        self,
        num_samples: int,
        q_limits: np.ndarray
    ) -> np.ndarray:
        """Random uniform sampling in joint space."""
        q_samples = np.random.uniform(
            q_limits[:, 0],
            q_limits[:, 1],
            size=(num_samples, self.num_dof)
        )
        return q_samples
    
    def _sample_grid(
        self,
        num_samples: int,
        q_limits: np.ndarray
    ) -> np.ndarray:
        """Grid-based sampling in joint space."""
        points_per_dim = int(np.ceil(num_samples ** (1.0 / self.num_dof)))
        
        grids = []
        for i in range(self.num_dof):
            grids.append(np.linspace(q_limits[i, 0], q_limits[i, 1], points_per_dim))
        
        mesh = np.meshgrid(*grids, indexing='ij')
        q_samples = np.column_stack([m.ravel() for m in mesh])
        
        # Subsample if too many
        if len(q_samples) > num_samples:
            indices = np.random.choice(len(q_samples), num_samples, replace=False)
            q_samples = q_samples[indices]
        
        return q_samples
    
    def _sample_sobol(
        self,
        num_samples: int,
        q_limits: np.ndarray
    ) -> np.ndarray:
        """Sobol quasi-random sampling."""
        try:
            from scipy.stats import qmc
            sampler = qmc.Sobol(d=self.num_dof, scramble=True)
            q_unit = sampler.random(num_samples)
            
            # Scale to joint limits
            q_samples = q_limits[:, 0] + q_unit * (q_limits[:, 1] - q_limits[:, 0])
            return q_samples
        except ImportError:
            warnings.warn("scipy.stats.qmc not available. Using monte_carlo.")
            return self._sample_monte_carlo(num_samples, q_limits)
    
    def _compute_fk_sequential(self, q_samples: np.ndarray) -> np.ndarray:
        """Compute FK sequentially."""
        from robocore.kinematics.fk import forward_kinematics
        
        points = []
        for q in q_samples:
            T = forward_kinematics(self.model, q, backend=self.backend, return_end=True)
            if isinstance(T, np.ndarray):
                points.append(T[:3, 3])
            else:
                import torch
                points.append(T[:3, 3].cpu().numpy())
        
        return np.array(points)
    
    def _compute_fk_parallel(
        self,
        q_samples: np.ndarray,
        num_workers: Optional[int] = None
    ) -> np.ndarray:
        """Compute FK in parallel (for large datasets)."""
        from robocore.kinematics.fk import forward_kinematics
        
        def compute_point(q):
            T = forward_kinematics(self.model, q, backend='numpy', return_end=True)
            return T[:3, 3]
        
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            points = list(executor.map(compute_point, q_samples))
        
        return np.array(points)
    
    def _estimate_bounding_box_volume(self, points: np.ndarray) -> float:
        """Estimate volume using bounding box."""
        bounds = self.get_workspace_bounds(points)
        volume = (
            (bounds['x'][1] - bounds['x'][0]) *
            (bounds['y'][1] - bounds['y'][0]) *
            (bounds['z'][1] - bounds['z'][0])
        )
        return volume
    
    def _estimate_voxel_volume(
        self,
        points: np.ndarray,
        resolution: int = 50
    ) -> float:
        """Estimate volume using voxel grid."""
        bounds = self.get_workspace_bounds(points)
        
        # Create voxel grid
        x_edges = np.linspace(bounds['x'][0], bounds['x'][1], resolution + 1)
        y_edges = np.linspace(bounds['y'][0], bounds['y'][1], resolution + 1)
        z_edges = np.linspace(bounds['z'][0], bounds['z'][1], resolution + 1)
        
        # Count occupied voxels
        hist, _ = np.histogramdd(points, bins=[x_edges, y_edges, z_edges])
        occupied_voxels = np.sum(hist > 0)
        
        # Voxel volume
        voxel_volume = (
            (bounds['x'][1] - bounds['x'][0]) / resolution *
            (bounds['y'][1] - bounds['y'][0]) / resolution *
            (bounds['z'][1] - bounds['z'][0]) / resolution
        )
        
        return occupied_voxels * voxel_volume
    
    def _plot_bounding_box(self, ax, bounds: Dict):
        """Plot bounding box."""
        x_min, x_max = bounds['x']
        y_min, y_max = bounds['y']
        z_min, z_max = bounds['z']
        
        # Draw edges
        edges = [
            # Bottom face
            ([x_min, x_max], [y_min, y_min], [z_min, z_min]),
            ([x_min, x_max], [y_max, y_max], [z_min, z_min]),
            ([x_min, x_min], [y_min, y_max], [z_min, z_min]),
            ([x_max, x_max], [y_min, y_max], [z_min, z_min]),
            # Top face
            ([x_min, x_max], [y_min, y_min], [z_max, z_max]),
            ([x_min, x_max], [y_max, y_max], [z_max, z_max]),
            ([x_min, x_min], [y_min, y_max], [z_max, z_max]),
            ([x_max, x_max], [y_min, y_max], [z_max, z_max]),
            # Vertical edges
            ([x_min, x_min], [y_min, y_min], [z_min, z_max]),
            ([x_max, x_max], [y_min, y_min], [z_min, z_max]),
            ([x_min, x_min], [y_max, y_max], [z_min, z_max]),
            ([x_max, x_max], [y_max, y_max], [z_min, z_max]),
        ]
        
        for edge in edges:
            ax.plot(edge[0], edge[1], edge[2], 'k--', alpha=0.5, linewidth=1)


def analyze_workspace_comparison(
    model1: "RobotModel",
    model2: "RobotModel",
    num_samples: int = 10000,
    names: Optional[Tuple[str, str]] = None
) -> Dict[str, Any]:
    """
    Compare workspaces of two robots.
    
    Parameters
    ----------
    model1, model2 : RobotModel
        Robot models to compare
    num_samples : int
        Number of samples per robot
    names : tuple of str, optional
        Names for the robots
    
    Returns
    -------
    comparison : dict
        Comparison metrics
    
    Examples
    --------
    >>> model_6dof = RobotModel('robot_6dof.urdf')
    >>> model_7dof = RobotModel('robot_7dof.urdf')
    >>> comparison = analyze_workspace_comparison(model_6dof, model_7dof)
    >>> print(f"Volume ratio: {comparison['volume_ratio']:.2f}")
    """
    if names is None:
        names = ("Robot 1", "Robot 2")
    
    analyzer1 = WorkspaceAnalyzer(model1)
    analyzer2 = WorkspaceAnalyzer(model2)
    
    # Compute workspaces
    points1 = analyzer1.compute_reachable_workspace(num_samples)
    points2 = analyzer2.compute_reachable_workspace(num_samples)
    
    # Get metrics
    bounds1 = analyzer1.get_workspace_bounds(points1)
    bounds2 = analyzer2.get_workspace_bounds(points2)
    
    volume1 = analyzer1.estimate_workspace_volume(points1)
    volume2 = analyzer2.estimate_workspace_volume(points2)
    
    comparison = {
        'names': names,
        'num_points': (len(points1), len(points2)),
        'bounds': (bounds1, bounds2),
        'volumes': (volume1, volume2),
        'volume_ratio': volume1 / volume2 if volume2 > 0 else np.inf,
        'points': (points1, points2)
    }
    
    return comparison
