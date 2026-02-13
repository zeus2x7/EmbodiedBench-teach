#!/bin/bash
source /home/ubuntu/miniconda3/etc/profile.d/conda.sh
export EMBODIED_BENCH_ROOT=$(pwd)
export DISPLAY=:1

# ========================
# Start X server ONCE
# ========================
echo "=== Setting up X server ==="

# Only start if not already running
if xdpyinfo -display :1 >/dev/null 2>&1; then
    echo "X server is already running on :1"
else
    echo "Starting X server..."
    pkill Xorg 2>/dev/null
    sleep 2
    rm -f /tmp/.X1-lock

    conda activate embench
    python -m embodiedbench.envs.eb_alfred.scripts.startx 1 &
    XPID=$!

    echo "Waiting for X server to initialize..."
    for i in {1..60}; do
        if xdpyinfo -display :1 >/dev/null 2>&1; then
            echo "X server is running."
            break
        fi
        sleep 1
    done

    if ! xdpyinfo -display :1 >/dev/null 2>&1; then
        echo "ERROR: X server failed to start after 60 seconds."
        exit 1
    fi
fi

mkdir -p agent_outputs

# ========================
# EB-ALFRED Random Agent
# ========================
echo ""
echo "========================================="
echo "=== Running EB-ALFRED Random Agent ==="
echo "========================================="
conda activate embench
python random_agent_alfred.py > agent_outputs/alfred.log 2>&1
echo "EB-ALFRED done. Check agent_outputs/alfred.log"

# ========================
# EB-Habitat Random Agent
# ========================
echo ""
echo "========================================="
echo "=== Running EB-Habitat Random Agent ==="
echo "========================================="
conda activate embench
python random_agent_habitat.py > agent_outputs/habitat.log 2>&1
echo "EB-Habitat done. Check agent_outputs/habitat.log"

# ========================
# EB-Navigation Random Agent
# ========================
echo ""  
echo "========================================="
echo "=== Running EB-Navigation Random Agent ==="
echo "========================================="
conda activate embench_nav
python random_agent_nav.py > agent_outputs/navigation.log 2>&1
echo "EB-Navigation done. Check agent_outputs/navigation.log"

# ========================
# EB-Manipulation Random Agent
# ========================
echo ""
echo "========================================="
echo "=== Running EB-Manipulation Random Agent ==="
echo "========================================="
conda activate embench_man
export COPPELIASIM_ROOT=$EMBODIED_BENCH_ROOT/CoppeliaSim_Pro_V4_1_0_Ubuntu20_04
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$COPPELIASIM_ROOT:$CONDA_PREFIX/lib:$CONDA_PREFIX/x86_64-conda-linux-gnu/lib
export QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT
python random_agent_man.py > agent_outputs/manipulation.log 2>&1
echo "EB-Manipulation done. Check agent_outputs/manipulation.log"

# ========================
# Summary
# ========================
echo ""
echo "========================================="
echo "=== Summary ==="
echo "========================================="
for env in alfred habitat navigation manipulation; do
    logfile="agent_outputs/${env}.log"
    if [ -f "$logfile" ]; then
        if grep -q "Completed" "$logfile"; then
            echo "  $env: PASS"
        elif grep -q "Error\|FAIL\|Traceback" "$logfile"; then
            echo "  $env: FAIL (check $logfile)"
        else
            echo "  $env: UNKNOWN (check $logfile)"
        fi
    else
        echo "  $env: NO LOG"
    fi
done

echo ""
echo "All agents completed. Outputs in agent_outputs/"
echo "  Frames: agent_outputs/<env>/episode_*/frame_*.png"
echo "  Videos: agent_outputs/<env>/episode_*/episode.mp4"
echo "  Logs:   agent_outputs/<env>/episode_*/log.json"
