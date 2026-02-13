#!/bin/bash
source /home/ubuntu/miniconda3/etc/profile.d/conda.sh
conda activate embench
export DISPLAY=:1

# Start X server
python -m embodiedbench.envs.eb_alfred.scripts.startx 1 &
pid=$!
echo "Started X server with PID $pid, waiting 10s..."
sleep 10

# Run test
echo "Running test..."
python test_alfred_auto.py > alfred_test.log 2>&1
result=$?

if [ $result -eq 0 ]; then
    echo "ALFRED Test Passed"
else
    echo "ALFRED Test Failed"
fi

kill $pid
