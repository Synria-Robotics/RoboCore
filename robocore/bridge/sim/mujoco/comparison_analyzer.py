"""Comparison analyzer for different control modes and trajectories.

Copyright (c) 2025 Synria Robotics Co., Ltd.

Licensed under the MIT License.

Author: Synria Robotics Team
Website: https://synriarobotics.ai
"""

from typing import Dict, Any, List, Optional
import numpy as np

from .trajectory_evaluator import TrajectoryEvaluator


class ComparisonAnalyzer:
    """Compare trajectory execution across different conditions.
    
    Supports comparison of:
    - Different control modes (position, velocity, torque)
    - Different trajectories (smooth vs non-smooth)
    - Different servo limits
    """
    
    def __init__(self, dt: float = 0.002):
        """Initialize comparison analyzer.
        
        Parameters
        ----------
        dt : float
            Time step for evaluation
        """
        self.evaluator = TrajectoryEvaluator(dt)
        self.comparisons = []
    
    def add_comparison(
        self,
        name: str,
        execution_data: Dict[str, np.ndarray],
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Add execution data for comparison.
        
        Parameters
        ----------
        name : str
            Name/identifier for this execution
        execution_data : dict
            Execution data from TrajectoryExecutor
        metadata : dict, optional
            Additional metadata (e.g., control_mode, trajectory_type)
        """
        metrics = self.evaluator.evaluate(execution_data)
        
        comparison = {
            'name': name,
            'execution_data': execution_data,
            'metrics': metrics,
            'metadata': metadata or {}
        }
        
        self.comparisons.append(comparison)
    
    def compare_all(self) -> Dict[str, Any]:
        """Compare all added executions.
        
        Returns
        -------
        comparison_results : dict
            Comparison results with summary statistics
        """
        if len(self.comparisons) < 2:
            raise ValueError("Need at least 2 comparisons")
        
        results = {
            'comparisons': self.comparisons,
            'summary': self._compute_summary()
        }
        
        return results
    
    def _compute_summary(self) -> Dict[str, Any]:
        """Compute summary statistics across comparisons."""
        summary = {
            'tracking_error': {
                'position_rms': [],
                'position_max': [],
                'velocity_rms': [],
                'velocity_max': []
            },
            'smoothness': {
                'jerk_max': [],
                'jerk_rms': []
            },
            'torque': {
                'max': [],
                'rms': [],
                'rate_max': []
            },
            'vibration': {
                'has_vibration': []
            }
        }
        
        for comp in self.comparisons:
            metrics = comp['metrics']
            
            # Tracking error
            tracking = metrics['tracking_error']
            summary['tracking_error']['position_rms'].append(tracking['position']['rms'])
            summary['tracking_error']['position_max'].append(tracking['position']['max'])
            summary['tracking_error']['velocity_rms'].append(tracking['velocity']['rms'])
            summary['tracking_error']['velocity_max'].append(tracking['velocity']['max'])
            
            # Smoothness
            smooth = metrics['smoothness']
            if smooth['jerk']['actual']['max'].ndim == 0:
                summary['smoothness']['jerk_max'].append(smooth['jerk']['actual']['max'])
                summary['smoothness']['jerk_rms'].append(smooth['jerk']['actual']['rms'])
            else:
                summary['smoothness']['jerk_max'].append(np.max(smooth['jerk']['actual']['max']))
                summary['smoothness']['jerk_rms'].append(np.max(smooth['jerk']['actual']['rms']))
            
            # Torque
            torque = metrics['torque']
            if torque['max'].ndim == 0:
                summary['torque']['max'].append(torque['max'])
                summary['torque']['rms'].append(torque['rms'])
                summary['torque']['rate_max'].append(torque['rate']['max'])
            else:
                summary['torque']['max'].append(np.max(torque['max']))
                summary['torque']['rms'].append(np.max(torque['rms']))
                summary['torque']['rate_max'].append(np.max(torque['rate']['max']))
            
            # Vibration
            if 'vibration' in metrics:
                summary['vibration']['has_vibration'].append(
                    metrics['vibration']['has_vibration']
                )
        
        # Convert to numpy arrays
        for key in summary:
            if isinstance(summary[key], dict):
                for subkey in summary[key]:
                    summary[key][subkey] = np.array(summary[key][subkey])
        
        return summary
    
    def generate_comparison_report(
        self,
        output_file: Optional[str] = None
    ) -> str:
        """Generate comparison report.
        
        Parameters
        ----------
        output_file : str, optional
            File path to save report
        
        Returns
        -------
        report : str
            Text report
        """
        if len(self.comparisons) < 2:
            return "Need at least 2 comparisons to generate report."
        
        comparison_results = self.compare_all()
        summary = comparison_results['summary']
        
        lines = []
        lines.append("=" * 70)
        lines.append("Trajectory Execution Comparison Report")
        lines.append("=" * 70)
        lines.append("")
        
        # List all comparisons
        lines.append("Comparisons:")
        for i, comp in enumerate(self.comparisons):
            lines.append(f"  {i+1}. {comp['name']}")
            if comp['metadata']:
                meta_str = ", ".join([f"{k}={v}" for k, v in comp['metadata'].items()])
                lines.append(f"     ({meta_str})")
        lines.append("")
        
        # Summary statistics
        lines.append("Summary Statistics")
        lines.append("-" * 70)
        
        # Tracking error comparison
        lines.append("Position Tracking Error (RMS):")
        for i, comp in enumerate(self.comparisons):
            rms = summary['tracking_error']['position_rms'][i]
            lines.append(f"  {comp['name']}: {rms:.6f}")
        lines.append("")
        
        lines.append("Position Tracking Error (Max):")
        for i, comp in enumerate(self.comparisons):
            max_err = summary['tracking_error']['position_max'][i]
            lines.append(f"  {comp['name']}: {max_err:.6f}")
        lines.append("")
        
        # Smoothness comparison
        lines.append("Jerk (Max):")
        for i, comp in enumerate(self.comparisons):
            jerk = summary['smoothness']['jerk_max'][i]
            lines.append(f"  {comp['name']}: {jerk:.6f}")
        lines.append("")
        
        # Torque comparison
        lines.append("Max Torque:")
        for i, comp in enumerate(self.comparisons):
            max_tau = summary['torque']['max'][i]
            lines.append(f"  {comp['name']}: {max_tau:.6f}")
        lines.append("")
        
        # Vibration
        if len(summary['vibration']['has_vibration']) > 0 and np.any(summary['vibration']['has_vibration']):
            lines.append("Vibration Detected:")
            for i, comp in enumerate(self.comparisons):
                has_vib = summary['vibration']['has_vibration'][i]
                lines.append(f"  {comp['name']}: {'Yes' if has_vib else 'No'}")
            lines.append("")
        
        # Best performer
        lines.append("Best Performers:")
        lines.append("-" * 70)
        
        # Best tracking (lowest RMS error)
        best_tracking_idx = np.argmin(summary['tracking_error']['position_rms'])
        lines.append(f"  Best Tracking: {self.comparisons[best_tracking_idx]['name']}")
        
        # Smoothest (lowest jerk)
        best_smooth_idx = np.argmin(summary['smoothness']['jerk_max'])
        lines.append(f"  Smoothest: {self.comparisons[best_smooth_idx]['name']}")
        
        # Lowest torque
        best_torque_idx = np.argmin(summary['torque']['max'])
        lines.append(f"  Lowest Torque: {self.comparisons[best_torque_idx]['name']}")
        
        lines.append("")
        lines.append("=" * 70)
        
        report = "\n".join(lines)
        
        if output_file:
            with open(output_file, 'w') as f:
                f.write(report)
        
        return report
    
    def clear(self):
        """Clear all comparisons."""
        self.comparisons = []


__all__ = ['ComparisonAnalyzer']

