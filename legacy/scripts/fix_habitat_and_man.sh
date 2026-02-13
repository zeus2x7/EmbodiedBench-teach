#!/bin/bash
source /home/ubuntu/miniconda3/etc/profile.d/conda.sh

echo "Fixing Habitat Data..."
conda activate embench
python -m habitat_sim.utils.datasets_download --uids rearrange_task_assets
if [ -d "data" ]; then
    mkdir -p embodiedbench/envs/eb_habitat
    mv data embodiedbench/envs/eb_habitat/
    echo "Habitat Data moved successfully."
else
    echo "Habitat Data download failed."
fi

echo "Fixing Manipulation..."
conda activate embench_man
# Install more Qt dependencies
conda install -y -c conda-forge libxkbcommon xorg-libxkbfile xorg-libxau xorg-libxdmcp xorg-libxext xorg-libxrender xorg-libxi xorg-libxtst
