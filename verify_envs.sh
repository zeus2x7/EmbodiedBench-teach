#!/bin/bash
set -e

echo "========================================"
echo "Verifying Environments inside Docker"
echo "========================================"

echo "[1/3] Testing 'embench' (ALFRED/Habitat)..."
conda run -n embench python -c "import habitat; print('Habitat imported successfully')"
conda run -n embench python -c "from embodiedbench.envs.eb_alfred.EBAlfEnv import EBAlfEnv; print('EBAlfEnv imported successfully')"

echo "[2/3] Testing 'embench_nav' (Navigation)..."
conda run -n embench_nav python -c "from embodiedbench.envs.eb_navigation.EBNavEnv import EBNavigationEnv; print('EBNavigationEnv imported successfully')"

echo "[3/3] Testing 'embench_man' (Manipulation)..."
conda run -n embench_man python -c "import amsolver; print('amsolver imported successfully')"
conda run -n embench_man python -c "from embodiedbench.envs.eb_manipulation.EBManEnv import EBManEnv; print('EBManEnv imported successfully')"

echo "========================================"
echo "All environments verified successfully!"
echo "========================================"
