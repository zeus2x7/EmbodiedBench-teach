#!/usr/bin/env python3
"""Pre-download AI2-THOR build for Navigation by initializing controller."""
import ai2thor.controller
import os, signal, sys

# Ignore SIGINT during download
signal.signal(signal.SIGINT, signal.SIG_IGN)

print('Initializing AI2-THOR Controller (this downloads the build if needed)...')
print('This may take several minutes for the first run.')

config = {
    "agentMode": "default",
    "gridSize": 0.1,
    "visibilityDistance": 10,
    "renderDepthImage": False,
    "renderInstanceSegmentation": False,
    "width": 300,
    "height": 300,
    "fieldOfView": 90,
}

try:
    controller = ai2thor.controller.Controller(**config)
    print('Controller initialized successfully!')
    controller.stop()
    print('Build is cached. Future launches will be fast.')
except Exception as e:
    print(f'Error: {e}')
    import traceback
    traceback.print_exc()
