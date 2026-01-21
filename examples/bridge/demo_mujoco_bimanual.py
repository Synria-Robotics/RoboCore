from robocore.bridge.sim.mujoco.interactive_ik import InteractiveIK
from synriard import get_model_path


def main(args):
    # Model path (Bessica is a dual-arm robot)
    mjcf_path = get_model_path(args.model, version=args.version, variant=args.variant, model_format="mjcf")

    # Map target names to end-effector links
    # The new interface auto-discovers targets from MJCF based on naming convention:
    # - ik_target_left_gripper -> left_arm_link7
    # - ik_target_right_gripper -> right_arm_link7
    target_end_links = {
        "left_gripper": "left_arm_link7",
        "right_gripper": "right_arm_link7",
    }

    # Create interactive IK controller
    controller = InteractiveIK(mjcf_path, target_end_links=target_end_links)

    # Run in specified mode
    controller.run(mode=args.mode)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Bimanual IK Control")
    parser.add_argument('--mode', type=str, default='independent', choices=['independent', 'relative', 'mirror'],
                        help='Mode: independent (independent mode), relative (relative mode), mirror (mirror mode)')
    parser.add_argument('--model', type=str, default="Bessica_M", choices=['Bessica_D', 'Bessica_M'],
                        help='Model: Bessica_D (bimanual), Bessica_M (bimanual with waist)')
    parser.add_argument('--version', type=str, default="v1_0", choices=['v1_0', 'v1_1'], help='Version: v1_0, v1_1')
    parser.add_argument('--variant', type=str, default="interactive")
    args = parser.parse_args()
    main(args)
