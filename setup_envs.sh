#!/bin/bash
source /home/ubuntu/miniconda3/etc/profile.d/conda.sh

echo "Recreating embench..."
conda env remove -n embench -y || true
conda env create -f conda_envs/environment.yaml

echo "Creating embench_nav..."
conda env create -f conda_envs/environment_eb-nav.yaml

echo "Creating embench_man..."
conda env create -f conda_envs/environment_eb-man.yaml
