#!/bin/bash
# Run Cosmos-Reason2-8B agent on all 4 EmbodiedBench environments for 2 episodes each.
# Logs saved to complete_cosmos_outputs_for_2episodes/

source /home/ubuntu/miniconda3/etc/profile.d/conda.sh
export EMBODIED_BENCH_ROOT=$(pwd)
export DISPLAY=:1
export OUTPUT_DIR="complete_cosmos_outputs_for_2episodes"
export NUM_EPISODES=2

mkdir -p $OUTPUT_DIR

# Ensure X server is running
if ! xdpyinfo -display :1 >/dev/null 2>&1; then
    echo "Starting X server..."
    conda activate embench
    python -m embodiedbench.envs.eb_alfred.scripts.startx 1 &
    for i in {1..60}; do
        xdpyinfo -display :1 >/dev/null 2>&1 && break
        sleep 1
    done
fi

echo "X server confirmed running."

# ========================
# 1. EB-ALFRED
# ========================
echo ""
echo "========================================="
echo "  1/4: EB-ALFRED (2 Episodes)"
echo "========================================="
conda activate embench
python cosmos_agent/cosmos_agent.py --env alfred --num_episodes $NUM_EPISODES --output_dir $OUTPUT_DIR 2>&1 | tee $OUTPUT_DIR/alfred.log
echo "EB-ALFRED done."

# ========================
# 2. EB-Habitat
# ========================
echo ""
echo "========================================="
echo "  2/4: EB-Habitat (2 Episodes)"
echo "========================================="
conda activate embench
python cosmos_agent/cosmos_agent.py --env habitat --num_episodes $NUM_EPISODES --output_dir $OUTPUT_DIR 2>&1 | tee $OUTPUT_DIR/habitat.log
echo "EB-Habitat done."

# ========================
# 3. EB-Navigation
# ========================
echo ""
echo "========================================="
echo "  3/4: EB-Navigation (2 Episodes)"
echo "========================================="
conda activate embench_nav
python cosmos_agent/cosmos_agent.py --env navigation --num_episodes $NUM_EPISODES --output_dir $OUTPUT_DIR 2>&1 | tee $OUTPUT_DIR/navigation.log
echo "EB-Navigation done."

# ========================
# 4. EB-Manipulation
# ========================
echo ""
echo "========================================="
echo "  4/4: EB-Manipulation (2 Episodes)"
echo "========================================="
conda activate embench_man
export COPPELIASIM_ROOT=$EMBODIED_BENCH_ROOT/CoppeliaSim_Pro_V4_1_0_Ubuntu20_04
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$COPPELIASIM_ROOT:$CONDA_PREFIX/lib:$CONDA_PREFIX/x86_64-conda-linux-gnu/lib
export QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT
python cosmos_agent/cosmos_agent.py --env manipulation --num_episodes $NUM_EPISODES --output_dir $OUTPUT_DIR 2>&1 | tee $OUTPUT_DIR/manipulation.log
echo "EB-Manipulation done."

# ========================
# Summary
# ========================
echo ""
echo "========================================="
echo "  All done! Outputs in $OUTPUT_DIR/"
echo "========================================="
