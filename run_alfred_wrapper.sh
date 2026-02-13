#!/bin/bash
source /home/ubuntu/miniconda3/etc/profile.d/conda.sh

# Load environment variables
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

# Activate environment
conda activate embench

# Default to :1 if DISPLAY not set
if [ -z "$DISPLAY" ]; then
    export DISPLAY=:1
fi

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

echo "=== EB-ALFRED Setup ==="
echo "Using DISPLAY=$DISPLAY"

# Check HF_TOKEN (Basic check)
if [ -z "$HF_TOKEN" ]; then
    echo "WARNING: HF_TOKEN is not set. Assuming model is cached or public."
    echo "         If this fails, copy .env.template to .env and set your token."
fi

# Check X Server
if ! xdpyinfo -display $DISPLAY >/dev/null 2>&1; then
    echo "X server not running on $DISPLAY. Attempting to start Xvfb..."
    
    # Try to find Xvfb
    XVFB_BIN=$(which Xvfb 2>/dev/null || which xvfb 2>/dev/null)
    
    if [ -n "$XVFB_BIN" ]; then
        rm -f /tmp/.X${DISPLAY#:}-lock
        $XVFB_BIN $DISPLAY -screen 0 1024x768x24 &
        XVFB_PID=$!
        sleep 2
        if ! kill -0 $XVFB_PID >/dev/null 2>&1; then
             echo "ERROR: Failed to start Xvfb."
             exit 1
        fi
        echo "Started Xvfb (PID $XVFB_PID)"
    else
        echo "ERROR: Xvfb not found. Please install xvfb or start X manually."
        echo "       sudo apt-get install xvfb"
        exit 1
    fi
fi

echo "Starting ALFRED agent..."
python -u cosmos_agent/run_cosmos_alfred.py 2>&1 | tee cosmos_outputs/alfred.log
