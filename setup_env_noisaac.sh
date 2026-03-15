#!/bin/bash

conda create -n isaac python=3.8 -y

conda activate isaac

# setup
pip install -e .