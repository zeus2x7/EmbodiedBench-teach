#!/bin/bash
source /home/ubuntu/miniconda3/etc/profile.d/conda.sh

# Function to check if environment exists
check_env() {
    conda env list | grep "$1" > /dev/null
    if [ $? -ne 0 ]; then
        echo "Environment $1 does not exist. Please create it first."
        exit 1
    fi
}

echo "Checking environments..."
check_env "embench"
check_env "embench_nav"
check_env "embench_man"

# Set root
export EMBODIED_BENCH_ROOT=$(pwd)
echo "EMBODIED_BENCH_ROOT is $EMBODIED_BENCH_ROOT"

# Setup embench
echo "=== Setting up embench environment ==="
conda activate embench
pip install -e .

# Install Git LFS
conda install -y -c conda-forge git-lfs
git lfs install
git lfs pull

# Install EB-ALFRED
echo "Installing EB-ALFRED..."
mkdir -p embodiedbench/envs/eb_alfred/data
if [ ! -d "embodiedbench/envs/eb_alfred/data/json_2.1.0" ]; then
    git clone https://huggingface.co/datasets/EmbodiedBench/EB-ALFRED
    mv EB-ALFRED embodiedbench/envs/eb_alfred/data/json_2.1.0
else
    echo "EB-ALFRED already installed."
fi

# Install EB-Habitat
echo "Installing EB-Habitat..."
conda install -y habitat-sim==0.3.0 withbullet headless -c conda-forge -c aihabitat
if [ ! -d "habitat-lab" ]; then
    git clone -b 'v0.3.0' --depth 1 https://github.com/facebookresearch/habitat-lab.git ./habitat-lab
fi

cd habitat-lab
pip install -e .
cd ..

python -m habitat_sim.utils.datasets_download --uids rearrange_task_assets
if [ -d "data" ]; then
    mkdir -p embodiedbench/envs/eb_habitat
    mv data embodiedbench/envs/eb_habitat/
fi


# Setup embench_nav
echo "=== Setting up embench_nav environment ==="
conda activate embench_nav
pip install -e .

# Setup embench_man
echo "=== Setting up embench_man environment ==="
conda activate embench_man
pip install -e .

echo "Installing CoppeliaSim..."
if [ ! -d "CoppeliaSim_Pro_V4_1_0_Ubuntu20_04" ]; then
    cd embodiedbench/envs/eb_manipulation
    wget https://downloads.coppeliarobotics.com/V4_1_0/CoppeliaSim_Pro_V4_1_0_Ubuntu20_04.tar.xz
    tar -xf CoppeliaSim_Pro_V4_1_0_Ubuntu20_04.tar.xz
    rm CoppeliaSim_Pro_V4_1_0_Ubuntu20_04.tar.xz
    mv CoppeliaSim_Pro_V4_1_0_Ubuntu20_04 $EMBODIED_BENCH_ROOT/
    cd $EMBODIED_BENCH_ROOT
fi

export COPPELIASIM_ROOT=$EMBODIED_BENCH_ROOT/CoppeliaSim_Pro_V4_1_0_Ubuntu20_04
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$COPPELIASIM_ROOT
export QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT

echo "Installing PyRep..."
cd embodiedbench/envs/eb_manipulation

if [ ! -d "PyRep" ]; then
    git clone https://github.com/stepjam/PyRep.git
    cd PyRep
    pip install -r requirements.txt
    pip install -e .
    cd ..
else
    echo "PyRep already cloned."
fi

# Install local requirements
pip install -r requirements.txt
pip install -e .

# Copy script to CoppeliaSim
cp ./simAddOnScript_PyRep.lua $COPPELIASIM_ROOT

# Install Data
if [ ! -d "data" ]; then
    echo "Cloning EB-Manipulation data..."
    git clone https://huggingface.co/datasets/EmbodiedBench/EB-Manipulation
    if [ -d "EB-Manipulation/data" ]; then
        mv EB-Manipulation/data/ ./
    fi
    rm -rf EB-Manipulation/
else
    echo "EB-Manipulation data already exists."
fi

cd $EMBODIED_BENCH_ROOT
echo "Dependencies installation complete."
