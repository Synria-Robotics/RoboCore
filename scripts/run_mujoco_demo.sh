#!/bin/bash
# Run MuJoCo interactive demos with mjpython (required on macOS)

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

# Run the demo with all arguments passed through
echo "Using mjpython: $MJPYTHON_PATH"
"$MJPYTHON_PATH" examples/bridge/demo_mujoco_bimanual.py "$@"
