#!/bin/bash
# Run MuJoCo interactive demos with mjpython (required on macOS)
#
# Usage:
#   ./run_mujoco_demo.sh [demo_type] [demo_args...]
#
# Demo types:
#   bimanual  - Dual-arm IK control (default)
#   humanoid  - Humanoid IK control with thumb and toe dragging
#
# Examples:
#   ./run_mujoco_demo.sh bimanual
#   ./run_mujoco_demo.sh humanoid
#   ./run_mujoco_demo.sh bimanual --mode independent
#   ./run_mujoco_demo.sh humanoid --variant g1_body29_hand14_interactive

# Try to find mjpython in common locations
MJPYTHON_PATH=""

# Check conda bin directory
if [ -f "$HOME/anaconda3/envs/synria/bin/mjpython" ]; then
    MJPYTHON_PATH="$HOME/anaconda3/envs/synria/bin/mjpython"
elif [ -f "$HOME/miniconda3/envs/synria/bin/mjpython" ]; then
    MJPYTHON_PATH="$HOME/miniconda3/envs/synria/bin/mjpython"
elif command -v mjpython &> /dev/null; then
    MJPYTHON_PATH="mjpython"
fi

if [ -z "$MJPYTHON_PATH" ]; then
    echo "ERROR: mjpython not found"
    echo ""
    echo "Please install MuJoCo Python bindings:"
    echo "  conda install -c conda-forge mujoco"
    echo "or"
    echo "  pip install mujoco"
    exit 1
fi

# Determine demo type from first argument
DEMO_TYPE="${1:-bimanual}"

# Set demo script based on type
case "$DEMO_TYPE" in
    bimanual)
        DEMO_SCRIPT="examples/bridge/demo_mujoco_bimanual.py"
        shift  # Remove demo_type from arguments
        ;;
    humanoid)
        DEMO_SCRIPT="examples/bridge/demo_mujoco_humanoid.py"
        shift  # Remove demo_type from arguments
        ;;
    *)
        echo "ERROR: Unknown demo type: $DEMO_TYPE"
        echo ""
        echo "Available demo types:"
        echo "  bimanual  - Dual-arm IK control"
        echo "  humanoid  - Humanoid IK control with thumb and toe dragging"
        echo ""
        echo "Usage: $0 [demo_type] [demo_args...]"
        exit 1
        ;;
esac

# Run the demo with remaining arguments passed through
echo "Using mjpython: $MJPYTHON_PATH"
echo "Running demo: $DEMO_SCRIPT"
"$MJPYTHON_PATH" "$DEMO_SCRIPT" "$@"
