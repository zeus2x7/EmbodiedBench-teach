#!/usr/bin/env python3
"""
Run Cosmos-Reason2-8B on EB-ALFRED (single task).
Usage: python run_cosmos_alfred.py
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cosmos_agent.cosmos_model import CosmosReason2Model
from cosmos_agent.cosmos_agent import CosmosAgent

if __name__ == "__main__":
    model = CosmosReason2Model("nvidia/Cosmos-Reason2-8B")
    agent = CosmosAgent(model)
    agent.run_alfred()
