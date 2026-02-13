#!/bin/bash
source /home/ubuntu/miniconda3/etc/profile.d/conda.sh

# Load environment variables
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

# Activate environment
conda activate embench_man

# Default to :1 if DISPLAY not set
if [ -z "$DISPLAY" ]; then
    export DISPLAY=:1
fi

echo "=== EB-Manipulation Setup ==="
echo "Using DISPLAY=$DISPLAY"

# Check HF_TOKEN
if [ -z "$HF_TOKEN" ]; then
    echo "WARNING: HF_TOKEN is not set."
fi

# Check X Server
if ! xdpyinfo -display $DISPLAY >/dev/null 2>&1; then
    echo "X server not running on $DISPLAY. Attempting to start Xvfb..."
    XVFB_BIN=$(which Xvfb 2>/dev/null || which xvfb 2>/dev/null)
    if [ -n "$XVFB_BIN" ]; then
        rm -f /tmp/.X${DISPLAY#:}-lock
        $XVFB_BIN $DISPLAY -screen 0 1280x1024x24 +extension GLX +extension RANDR +extension RENDER &
        XVFB_PID=$!
        sleep 5
        echo "Started Xvfb (PID $XVFB_PID)"
    else
        echo "ERROR: Xvfb not found. Please start X manually."
    fi
fi

export COPPELIASIM_ROOT=/home/ubuntu/workspace/EmbodiedBench/CoppeliaSim_Pro_V4_1_0_Ubuntu20_04
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/lib/x86_64-linux-gnu:$COPPELIASIM_ROOT:$CONDA_PREFIX/lib:$CONDA_PREFIX/x86_64-conda-linux-gnu/lib:$LD_LIBRARY_PATH
export QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT
export QT_QPA_PLATFORM=xcb
export LIBGL_ALWAYS_SOFTWARE=1
export LIBGL_DRIVERS_PATH=/usr/lib/x86_64-linux-gnu/dri
export MESA_LOADER_DRIVER_OVERRIDE=swrast
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libxcb-dri3.so.0:/usr/lib/x86_64-linux-gnu/libxcb.so.1

echo "Starting Manipulation agent..."
python -u cosmos_agent/run_cosmos_man.py 2>&1 | tee cosmos_outputs/manipulation.log
