from typing import Any, Dict, Sequence
import numpy as np

from robocore.modeling.robot_model import RobotModel


class BiIndependentFKSolverNumpy:
    def __init__(self, left_model: RobotModel, right_model: RobotModel):
        self.left = left_model
        self.right = right_model

    def fk(self, q_left: Sequence[float], q_right: Sequence[float], *, backend: str = 'numpy', return_end: bool = True) -> Dict[str, Any]:
        T_l = self.left.fk(q_left, backend=backend, return_end=return_end)
        T_r = self.right.fk(q_right, backend=backend, return_end=return_end)

        T_l = np.array(T_l)
        T_r = np.array(T_r)
        return {'left': T_l, 'right': T_r}
