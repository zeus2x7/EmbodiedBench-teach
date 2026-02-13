#!/bin/bash
# Ensure correct ownership
chown -R $(whoami) /home/ubuntu/workspace/EmbodiedBench || true

# Pre-create the physics config file to avoid permission issues or directory issues
touch embodiedbench/envs/eb_habitat/data/default.physics_config.json || true

bash run_tests.sh
