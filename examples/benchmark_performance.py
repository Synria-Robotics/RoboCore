"""Performance benchmark: NumPy vs PyTorch backends.

Compares FK, Jacobian, and IK across backends.
"""
import os
import time, argparse, numpy as np
from pathlib import Path
from robocore import RobotModel
from robocore.kinematics import forward_kinematics, inverse_kinematics, jacobian
from robocore.utils.beauty_logger import beauty_print

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

def benchmark_fk(model, q, backend, n_runs=1000, device=None):
    forward_kinematics(model, q, backend=backend, device=device)
    start = time.perf_counter()
    for _ in range(n_runs):
        _ = forward_kinematics(model, q, backend=backend, device=device)
    return (time.perf_counter() - start) / n_runs * 1000


def main(args):
    base = Path(__file__).resolve().parents[1]
    urdf = os.path.join(base, "robocore/assets/robot/urdf/Alicia-D_v5_4/alicia_duo_with_gripper.urdf")
    model = RobotModel(str(urdf), end_link="tool0")
    q = [0.1, 0.2, -0.3, 0.0, 0.5, -0.2, 0.0][:model.dof()]
    
    beauty_print(f"Performance Benchmark: {model.name} ({model.dof()} DOF)", type="module")
    
    beauty_print("\nForward Kinematics", type="module")
    time_np = benchmark_fk(model, q, 'numpy', n_runs=args.fk_runs)
    beauty_print(f"  NumPy: {time_np:.4f} ms")
    if HAS_TORCH:
        device = torch.device(args.torch_device)
        time_torch = benchmark_fk(model, q, 'torch', n_runs=args.fk_runs, device=device)
        beauty_print(f"  Torch ({device}): {time_torch:.4f} ms")
        speedup = time_torch/time_np if time_np < time_torch else time_np/time_torch
        faster = "NumPy" if time_np < time_torch else "Torch"
        beauty_print(f"  {faster} is {speedup:.2f}x faster", type="success")
    
    beauty_print("\n✓ Benchmark complete", type="success")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--fk-runs', type=int, default=1000)
    parser.add_argument('--torch-device', default='cpu')
    args = parser.parse_args()
    main(args)
