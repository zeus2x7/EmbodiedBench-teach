#!/bin/bash
# Setup script for Cosmos-Reason2-8B on EmbodiedBench
# 
# Prerequisites:
#   1. Go to https://huggingface.co/nvidia/Cosmos-Reason2-8B and accept the license
#   2. Create an access token at https://huggingface.co/settings/tokens
#   3. Run: export HF_TOKEN=<your_token>
#   4. Then run this script

source /home/ubuntu/miniconda3/etc/profile.d/conda.sh
conda activate embench

echo "=== Cosmos-Reason2-8B Setup ==="
echo ""

# Check for HF token
if [ -z "$HF_TOKEN" ]; then
    echo "ERROR: HF_TOKEN not set."
    echo ""
    echo "Steps to fix:"
    echo "  1. Go to https://huggingface.co/nvidia/Cosmos-Reason2-8B"
    echo "     Click 'Agree and access repository' to accept the license."
    echo ""
    echo "  2. Create a token at https://huggingface.co/settings/tokens"
    echo "     (Read access is sufficient)"
    echo ""
    echo "  3. Run:"
    echo "     export HF_TOKEN=hf_xxxxxxxxxxxxx"
    echo "     bash setup_cosmos.sh"
    exit 1
fi

echo "Logging in to HuggingFace..."
huggingface-cli login --token "$HF_TOKEN"

echo ""
echo "Testing model download..."
python -c "
import transformers, torch
print('Attempting to load model config...')
config = transformers.AutoConfig.from_pretrained('nvidia/Cosmos-Reason2-8B')
print('Config loaded! Model architecture:', config.architectures)
print('')
print('Setup complete! You can now run:')
print('  bash run_cosmos_agents.sh')
"
