#!/bin/bash
# fix_display.sh - Kill stale Xvfb and start a new one on :1

echo "Terminating existing Xvfb processes..."
pkill -f Xvfb
pkill -f Xorg

echo "Removing stale lock files..."
rm -f /tmp/.X1-lock
rm -f /tmp/.X11-unix/X1

echo "Starting Xvfb on DISPLAY=:1..."
Xvfb :1 -screen 0 1024x768x24 &
sleep 2

echo "Checking if Xvfb is running..."
if pgrep -f "Xvfb :1" > /dev/null; then
    echo "Xvfb started successfully on :1"
else
    echo "Failed to start Xvfb on :1"
    exit 1
fi

echo "Done."
