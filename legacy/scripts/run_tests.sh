#!/bin/bash
source /home/ubuntu/miniconda3/etc/profile.d/conda.sh
export EMBODIED_BENCH_ROOT=$(pwd)
export DISPLAY=:1

# Fix for Habitat
echo "Fixing Habitat Installation..."
conda activate embench
# Ensure habitat-lab is installed correctly
if ! python -c "import habitat" 2>/dev/null; then
    echo "Habitat not found. Installing..."
    cd habitat-lab
    pip install -e habitat-lab
    cd ..
fi

# Start X
python -m embodiedbench.envs.eb_alfred.scripts.startx 1 &
pid=$!
sleep 15

echo "=== Running EB-ALFRED (Automated) ==="
python test_alfred_auto.py > alfred_final.log 2>&1
echo "EB-ALFRED Test completed. Check alfred_final.log"

echo "=== Running EB-Habitat (Automated) ==="
# Create test script for habitat
cat << 'EOF' > test_habitat_auto.py
try:
    from embodiedbench.envs.eb_habitat.EBHabEnv import EBHabEnv
    import sys, random
    print("Init EB-Habitat Env...")
    env = EBHabEnv(eval_set='base', down_sample_ratio=0.01)
    env.reset()
    print("Habitat Reset Done")
    for i in range(5):
        # The code for EBHabEnv might differ, lets assume standard gym + some custom props
        # Checking language skill set
        action_id = random.randint(0, env.action_space.n - 1)
        if hasattr(env, 'language_skill_set'):
             action_str = env.language_skill_set[action_id]
        else:
             action_str = str(action_id)
             
        print(f"Action {i}: {action_str}")
        # Step signature might be standard
        step_result = env.step(action_id)
        # Handle unpacking depending on length
        if len(step_result) == 4:
            obs, reward, done, info = step_result
        else:
            print(f"Unexpected step result length: {len(step_result)}")
            obs, reward, done, info = step_result[0], step_result[1], step_result[2], {}
            
        if done: break
    env.close()
    print("EB-Habitat PASS")
except Exception as e:
    print(f"EB-Habitat FAIL: {e}")
    import traceback; traceback.print_exc()
EOF
python test_habitat_auto.py > habitat_final.log 2>&1
echo "EB-Habitat Test completed. Check habitat_final.log"

echo "=== Running EB-Navigation (Automated) ==="
conda activate embench_nav
# Create test script for navigation
cat << 'EOF' > test_nav_auto.py
try:
    from embodiedbench.envs.eb_navigation.EBNavEnv import EBNavEnv
    import sys, random
    print("Init EB-Navigation Env...")
    env = EBNavEnv(eval_set='base', down_sample_ratio=0.1)
    env.reset()
    for i in range(5):
        action_id = random.randint(0, env.action_space.n - 1)
        action_str = env.language_skill_set[action_id]
        print(f"Action {i}: {action_str}")
        obs, reward, done, info = env.step(action_id)
        if done: break
    env.close()
    print("EB-Navigation PASS")
except Exception as e:
    print(f"EB-Navigation FAIL: {e}")
    import traceback; traceback.print_exc()
EOF
python test_nav_auto.py > navigation_final.log 2>&1
echo "EB-Navigation Test completed. Check navigation_final.log"
echo "=== Running EB-Manipulation ==="
conda activate embench_man
export COPPELIASIM_ROOT=$EMBODIED_BENCH_ROOT/CoppeliaSim_Pro_V4_1_0_Ubuntu20_04
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$COPPELIASIM_ROOT:$CONDA_PREFIX/lib:$CONDA_PREFIX/x86_64-conda-linux-gnu/lib
export QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT
# export QT_DEBUG_PLUGINS=1

# Create test script for manipulation
cat << 'EOF' > test_man_auto.py
from embodiedbench.envs.eb_manipulation.EBManEnv import EBManEnv
import sys, random, faulthandler
import numpy as np
faulthandler.enable()
sys.stdout.reconfigure(line_buffering=True)
try:
    print("Init EB-Manipulation Env...")
    env = EBManEnv(eval_set='base', down_sample_ratio=0.1)
    env.reset()
    print("EB-Manipulation Env reset done")
    action_space = env.action_space
    print(f"Action space: {action_space}")
    
    for i in range(5):
        if hasattr(action_space, 'n'):
            action_id = random.randint(0, action_space.n - 1)
            action_str = env.language_skill_set[action_id]
            print(f"Action {i}: {action_str} (Discrete)")
            action = action_id
        else:
            # Box action space
            action = action_space.sample()
            print(f"Action {i}: Sampled Box action")
            
        obs, reward, done, info = env.step(action)
        if done: break
    env.close()
    print("EB-Manipulation PASS")
except Exception as e:
    print(f"EB-Manipulation FAIL: {e}")
    import traceback; traceback.print_exc()
EOF
python test_man_auto.py > manipulation_final.log 2>&1
echo "EB-Manipulation Test completed. Check manipulation_final.log"

kill $pid
