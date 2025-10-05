"""Singularity analysis for robot manipulators.

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

from typing import Dict, List, Sequence
import math
from robocore.modeling.robot_model import RobotModel
from robocore.kinematics import jacobian
from robocore.utils.beauty_logger import beauty_print


def _svd_simple(A: List[List[float]]):
    """Simplified SVD for small matrices (singular values only).

    For 6xn Jacobian, compute singular values via eigenvalues of A A^T.
    :param A: matrix (m x n).
    :return: singular values (sorted descending).
    """
    m = len(A)
    n = len(A[0]) if A else 0
    # Compute A A^T (m x m)
    AAT = [[sum(A[i][k] * A[j][k] for k in range(n)) for j in range(m)] for i in range(m)]
    # Power iteration for largest eigenvalue (simplified, not full SVD)
    eigenvalues = []
    for _ in range(min(m, n)):
        v = [1.0] * m
        for _ in range(50):
            Av = [sum(AAT[i][j] * v[j] for j in range(m)) for i in range(m)]
            norm = math.sqrt(sum(x * x for x in Av))
            if norm < 1e-14:
                break
            v = [x / norm for x in Av]
        lam = sum(AAT[i][j] * v[i] * v[j] for i in range(m) for j in range(m))
        eigenvalues.append(math.sqrt(max(0, lam)))
        # Deflate (simplified - not accurate for multiple eigenvalues)
        for i in range(m):
            for j in range(m):
                AAT[i][j] -= lam * v[i] * v[j]
    return sorted(eigenvalues, reverse=True)


class SingularityAnalyzer:
    """Analyze robot singularity properties.

    :param model: robot model.
    """

    def __init__(self, model: RobotModel):
        self.model = model

    def analyze_configuration(self, q: Sequence[float]) -> Dict[str, float]:
        """Analyze singularity metrics at configuration.

        :param q: joint configuration.
        :return: dict with manipulability, condition_number, min_singular_value.
        """
        # Prefer numpy Jacobian; if q is torch tensor and torch backend desired, we could extend.
        # For now always use numpy version for stability & speed (convert q if needed).
        import numpy as _np
        if hasattr(q, 'detach'):
            q_np = _np.asarray(q.detach().cpu(), dtype=float)
        else:
            q_np = _np.asarray(q, dtype=float)
        Jn = jacobian(self.model, q_np, backend='numpy', method='numeric', use_central_diff=True)
        # jacobian returns 6xN numpy array; convert to list of lists for existing logic
        J = Jn.tolist()
        # Compute singular values
        sigma = _svd_simple(J)
        sigma_min = sigma[-1] if sigma else 0.0
        sigma_max = sigma[0] if sigma else 0.0
        cond = sigma_max / sigma_min if sigma_min > 1e-12 else 1e12
        # Manipulability (Yoshikawa): sqrt(det(J J^T))
        manipulability = 1.0
        for s in sigma:
            manipulability *= s
        manipulability = math.sqrt(max(0, manipulability))

        return {
            "manipulability": manipulability,
            "condition_number": cond,
            "min_singular_value": sigma_min,
            "max_singular_value": sigma_max,
            "singular_values": sigma,
            "is_singular": sigma_min < 1e-3,
        }

    def sample_workspace(
        self, n_samples: int = 1000, seed: int = 42
    ) -> Dict[str, object]:
        """Sample random configurations and analyze singularity distribution.

        :param n_samples: number of samples.
        :param seed: random seed.
        :return: statistics dict.
        """
        import random

        random.seed(seed)
        metrics = []
        for _ in range(n_samples):
            q = []
            for js in self.model._actuated:
                lo, hi = -3.14, 3.14
                if js.limit:
                    if js.limit[0] is not None:
                        lo = js.limit[0]
                    if js.limit[1] is not None:
                        hi = js.limit[1]
                q.append(random.uniform(lo * 0.8, hi * 0.8))
            try:
                m = self.analyze_configuration(q)
                metrics.append(m)
            except Exception:
                pass

        if not metrics:
            return {"error": "No valid samples"}

        manip_vals = [m["manipulability"] for m in metrics]
        cond_vals = [m["condition_number"] for m in metrics]
        singular_count = sum(1 for m in metrics if m["is_singular"])

        return {
            "n_samples": len(metrics),
            "singular_configs": singular_count,
            "singular_ratio": singular_count / len(metrics),
            "manipulability_mean": sum(manip_vals) / len(manip_vals),
            "manipulability_min": min(manip_vals),
            "manipulability_max": max(manip_vals),
            "condition_number_mean": sum(cond_vals) / len(cond_vals),
            "condition_number_max": max(cond_vals),
        }


__all__ = ["SingularityAnalyzer"]
