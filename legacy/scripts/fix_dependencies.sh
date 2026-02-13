#!/bin/bash
source /home/ubuntu/miniconda3/etc/profile.d/conda.sh

echo "Fixing EB-Habitat dependencies..."
conda activate embench
conda install -y -c conda-forge libopengl libgl

echo "Fixing EB-Navigation dependencies..."
conda activate embench_nav
conda install -y -c conda-forge vulkan-loader mesa-libgl-devel-cos7-x86_64 mesa-dri-drivers-cos7-x86_64

echo "Fixing EB-Manipulation dependencies..."
conda activate embench_man
conda install -y -c conda-forge libxcb xorg-libxcb xorg-libx11 dbus
