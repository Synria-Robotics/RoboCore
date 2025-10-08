#!/bin/bash
# Run MuJoCo interactive demos with mjpython (required on macOS)

DEMO=$1

if [ -z "$DEMO" ]; then
    echo "Usage: $0 <demo_name>"
    echo "Available demos:"
    echo "  bi_independent - Independent dual-arm control"
    echo "  bi_relative - relative control with relative constraint"
    echo "  bi_mirror      - Mirror symmetric control"
    exit 1
fi

# Try to find mjpython in common locations
MJPYTHON_PATH=""

# Check conda bin directory
if [ -f "$HOME/anaconda3/bin/mjpython" ]; then
    MJPYTHON_PATH="$HOME/anaconda3/bin/mjpython"
elif [ -f "$HOME/miniconda3/bin/mjpython" ]; then
    MJPYTHON_PATH="$HOME/miniconda3/bin/mjpython"
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

# Run the demo
echo "Using mjpython: $MJPYTHON_PATH"
echo "Running demo: $DEMO"
"$MJPYTHON_PATH" examples/bridge/demo_mujoco_${DEMO}.py
