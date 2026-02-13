#!/bin/bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source /home/ubuntu/miniconda3/etc/profile.d/conda.sh
conda activate embench_nav
export DISPLAY=:1
python -u cosmos_agent/run_cosmos_nav.py 2>&1 | tee cosmos_outputs/navigation_retry.log
