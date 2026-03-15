#!/bin/bash

conda create -n isaac python=3.8 -y
# make outputs folder for training
mkdir -p outputs
# append to library path
if [ "$CONDA_DEFAULT_ENV" != "base" ]; then
    conda deactivate
fi
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/envs/isaac/lib
export RL_OUTPUT_PATH=$(pwd)/outputs
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/envs/isaac/lib' >> ~/.bashrc
echo "export RL_OUTPUT_PATH=$(pwd)" >> ~/.bashrc

conda activate isaac

pip install gdown

# from current directory
# setup IsaacGym
if [ ! -d "IsaacGym_Preview_TacSL_Package" ]; then
    echo "IsaacGym package not found. Downloading and extracting..."

    gdown 12Sb5IwyP2YGtlmprybgdepWThHhsCmm7
    tar -xf IsaacGym_Preview_TacSL_Package.tar.gz
    rm IsaacGym_Preview_TacSL_Package.tar.gz
else
    echo "IsaacGym_Preview_TacSL_Package already exists. Skipping download."
fi

cd IsaacGym_Preview_TacSL_Package/isaacgym/python
python3 -m pip install -e .
cd ../../..

# setup
pip install -e .