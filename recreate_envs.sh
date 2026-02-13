#!/bin/bash
# Script to recreate the 3 specialized Conda environments for EmbodiedBench

# 1. Main Environment: embench (Agent + ALFRED + Habitat)
echo "Creating 'embench' environment..."
conda create -n embench python=3.9 -y
conda run -n embench pip install -r requirements_embench.txt

# 2. Navigation Environment: embench_nav
echo "Creating 'embench_nav' environment..."
conda create -n embench_nav python=3.9 -y
conda run -n embench_nav pip install -r requirements_nav.txt

# 3. Manipulation Environment: embench_man
echo "Creating 'embench_man' environment..."
conda create -n embench_man python=3.9 -y
conda run -n embench_man pip install -r requirements_man.txt

echo "Environment setup complete."
