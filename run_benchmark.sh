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

echo "=== EmbodiedBench Complete Benchmark Run ==="
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

# Define output directory base
OUTPUT_BASE="cosmos_outputs_benchmark"
mkdir -p $OUTPUT_BASE

# Run Logic: Iterate through environments to restart python process each time (clean memory)
# Using -1 for num_episodes to run ALL available episodes in the base set.

echo "---------------------------------------------------"
echo "Running Navigation Benchmark..."
echo "---------------------------------------------------"
python -m cosmos_agent.cosmos_agent --env navigation --num_episodes -1 --model_path "nvidia/Cosmos-Reason2-8B" 2>&1 | tee $OUTPUT_BASE/navigation.log
echo "Running cleanup..."
bash kill_all.sh

echo "---------------------------------------------------"
echo "Running ALFRED Benchmark..."
echo "---------------------------------------------------"
python -m cosmos_agent.cosmos_agent --env alfred --num_episodes -1 --model_path "nvidia/Cosmos-Reason2-8B" 2>&1 | tee $OUTPUT_BASE/alfred.log
echo "Running cleanup..."
bash kill_all.sh

echo "---------------------------------------------------"
echo "Running Habitat Benchmark..."
echo "---------------------------------------------------"
python -m cosmos_agent.cosmos_agent --env habitat --num_episodes -1 --model_path "nvidia/Cosmos-Reason2-8B" 2>&1 | tee $OUTPUT_BASE/habitat.log
echo "Running cleanup..."
bash kill_all.sh

echo "---------------------------------------------------"
echo "Running Manipulation Benchmark..."
echo "---------------------------------------------------"
python -m cosmos_agent.cosmos_agent --env manipulation --num_episodes -1 --model_path "nvidia/Cosmos-Reason2-8B" 2>&1 | tee $OUTPUT_BASE/manipulation.log
echo "Running cleanup..."
bash kill_all.sh

echo "Benchmark Complete."
