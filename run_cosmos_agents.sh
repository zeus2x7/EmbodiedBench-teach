#!/bin/bash
# Run Cosmos-Reason2-8B agent on all 4 EmbodiedBench environments (single task each).
# Model requires ~32GB GPU memory, split across 2x RTX 5060 Ti (16GB each).

source /home/ubuntu/miniconda3/etc/profile.d/conda.sh
export EMBODIED_BENCH_ROOT=$(pwd)
export DISPLAY=:1

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
mkdir -p cosmos_outputs

# ========================
# 1. EB-ALFRED
# ========================
echo ""
echo "========================================="
echo "  1/4: EB-ALFRED"
echo "========================================="
conda activate embench
python cosmos_agent/run_cosmos_alfred.py 2>&1 | tee cosmos_outputs/alfred.log
echo "EB-ALFRED done."

# ========================
# 2. EB-Habitat
# ========================
echo ""
echo "========================================="
echo "  2/4: EB-Habitat"
echo "========================================="
conda activate embench
python cosmos_agent/run_cosmos_habitat.py 2>&1 | tee cosmos_outputs/habitat.log
echo "EB-Habitat done."

# ========================
# 3. EB-Navigation
# ========================
echo ""
echo "========================================="
echo "  3/4: EB-Navigation"
echo "========================================="
conda activate embench_nav
python cosmos_agent/run_cosmos_nav.py 2>&1 | tee cosmos_outputs/navigation.log
echo "EB-Navigation done."

# ========================
# 4. EB-Manipulation
# ========================
echo ""
echo "========================================="
echo "  4/4: EB-Manipulation"
echo "========================================="
conda activate embench_man
export COPPELIASIM_ROOT=$EMBODIED_BENCH_ROOT/CoppeliaSim_Pro_V4_1_0_Ubuntu20_04
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$COPPELIASIM_ROOT:$CONDA_PREFIX/lib:$CONDA_PREFIX/x86_64-conda-linux-gnu/lib
export QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT
python cosmos_agent/run_cosmos_man.py 2>&1 | tee cosmos_outputs/manipulation.log
echo "EB-Manipulation done."

# ========================
# Summary
# ========================
echo ""
echo "========================================="
echo "  All done! Outputs in cosmos_outputs/"
echo "========================================="
echo ""
echo "Per-environment logs:"
for env in EB-ALFRED EB-Habitat EB-Navigation EB-Manipulation; do
    ep_dir="cosmos_outputs/${env}/episode_1"
    if [ -f "$ep_dir/cosmos_log.json" ]; then
        steps=$(python3 -c "import json; d=json.load(open('$ep_dir/cosmos_log.json')); print(len(d))")
        echo "  $env: $steps steps logged -> $ep_dir/cosmos_log.json"
    else
        echo "  $env: NO LOG (check cosmos_outputs/ for errors)"
    fi
done
