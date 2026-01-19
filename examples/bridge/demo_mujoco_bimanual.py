from robocore.bridge.sim.mujoco.interactive_dual_arm import InteractiveDualArmIK
from synriard import get_model_path


def main(args):
    # Model path (Bessica is a dual-arm robot)
    mjcf_path = get_model_path(args.model, version=args.version, variant=args.variant, model_format="mjcf")

    # End-effector links
    left_end = "left_arm_link7"
    right_end = "right_arm_link7"

    # Create interactive IK controller
    controller = InteractiveDualArmIK(mjcf_path, left_end, right_end)

    # Run in independent mode
    controller.run(mode=args.mode)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Bimanual IK Control")
    parser.add_argument('--mode', type=str, default='independent', choices=['independent', 'relative', 'mirror'],
                        help='Mode: independent (independent mode), relative (relative mode), mirror (mirror mode)')
    parser.add_argument('--model', type=str, default="Bessica_M", choices=['Bessica_D', 'Bessica_M'],
                        help='Model: Bessica_D (bimanual), Bessica_M (bimanual with waist)')
    parser.add_argument('--version', type=str, default="v1_0", choices=['v1_0', 'v1_1'], help='Version: v1_0, v1_1')
    parser.add_argument('--variant', type=str, default="covered_interactive")
    args = parser.parse_args()
    main(args)
