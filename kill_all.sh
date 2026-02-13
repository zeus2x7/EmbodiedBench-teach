#!/bin/bash
# kill_all.sh - Terminate all relevant processes

echo "Checking for existing python processes..."
pkill -f "python cosmos_agent"
pkill -f "python -u cosmos_agent"
pkill -f "activate emben"

echo "Checking for existing Xvfb processes..."
pkill -f Xvfb
pkill -f Xorg

echo "Removing stale lock files..."
rm -f /tmp/.X1-lock
rm -f /tmp/.X11-unix/X1

echo "Done."
