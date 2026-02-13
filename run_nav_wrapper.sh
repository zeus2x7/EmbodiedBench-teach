#!/bin/bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source /home/ubuntu/miniconda3/etc/profile.d/conda.sh

# Load environment variables
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

conda activate embench_nav

# Default to :1 if DISPLAY not set
if [ -z "$DISPLAY" ]; then
    export DISPLAY=:1
fi

echo "=== EB-Navigation Setup ==="
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
        $XVFB_BIN $DISPLAY -screen 0 1024x768x24 &
        XVFB_PID=$!
        sleep 2
        echo "Started Xvfb (PID $XVFB_PID)"
    else
        echo "ERROR: Xvfb not found. Please start X manually."
        # Don't exit, might work if env uses internal renderer
    fi
fi

python -u cosmos_agent/run_cosmos_nav.py 2>&1 | tee cosmos_outputs/navigation.log
