#!/bin/bash
# Sequential Run of all EmbodiedBench environments for 2 episodes each.
# Logs saved to complete_cosmos_outputs_for_2episodes/

source /home/ubuntu/miniconda3/etc/profile.d/conda.sh
export EMBODIED_BENCH_ROOT=$(pwd)

# Load environment variables from .env if it exists
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

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

echo "1. Navigation"
conda activate embench_nav
python cosmos_agent/cosmos_agent.py --env navigation --num_episodes $NUM_EPISODES --output_dir $OUTPUT_DIR 2>&1 | tee $OUTPUT_DIR/navigation.log
bash kill_all.sh

echo "2. ALFRED"
conda activate embench
python cosmos_agent/cosmos_agent.py --env alfred --num_episodes $NUM_EPISODES --output_dir $OUTPUT_DIR 2>&1 | tee $OUTPUT_DIR/alfred.log
bash kill_all.sh

echo "3. Habitat"
conda activate embench
python cosmos_agent/cosmos_agent.py --env habitat --num_episodes $NUM_EPISODES --output_dir $OUTPUT_DIR 2>&1 | tee $OUTPUT_DIR/habitat.log
bash kill_all.sh

echo "4. Manipulation"
conda activate embench_man
export COPPELIASIM_ROOT=$EMBODIED_BENCH_ROOT/CoppeliaSim_Pro_V4_1_0_Ubuntu20_04
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$COPPELIASIM_ROOT:$CONDA_PREFIX/lib:$CONDA_PREFIX/x86_64-conda-linux-gnu/lib
export QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT
python cosmos_agent/cosmos_agent.py --env manipulation --num_episodes $NUM_EPISODES --output_dir $OUTPUT_DIR 2>&1 | tee $OUTPUT_DIR/manipulation.log
bash kill_all.sh

echo "All runs completed. Outputs in $OUTPUT_DIR/"
