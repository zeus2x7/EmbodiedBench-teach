#!/bin/bash
set -e

# Start Xvfb if not running
if ! xdpyinfo -display :1 >/dev/null 2>&1; then
    echo "Starting Xvfb on :1"
    XVFB_BIN=$(which Xvfb)
    rm -f /tmp/.X1-lock
    $XVFB_BIN :1 -screen 0 1024x768x24 &
    sleep 2
fi

export DISPLAY=:1

# Execute the passed command
exec "$@"
