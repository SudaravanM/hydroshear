#!/bin/bash
# Shared environment for the reproduction scripts. Source it; do not run it.
#   HYDROSHEAR_ROOT   repo root            default: four levels above this file
#   CONDA_SH          conda hook           default: ~/miniconda3/etc/profile.d/conda.sh
#   CONDA_ENV         env name             default: isaac
#   RL_OUTPUT_PATH    outputs directory    default: $HYDROSHEAR_ROOT/outputs
HYDROSHEAR_ROOT="${HYDROSHEAR_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)}"
source "${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"; conda activate "${CONDA_ENV:-isaac}"
cd "$HYDROSHEAR_ROOT" || exit 1
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$CONDA_PREFIX/lib"
export RL_OUTPUT_PATH="${RL_OUTPUT_PATH:-$HYDROSHEAR_ROOT/outputs}"; mkdir -p "$RL_OUTPUT_PATH"
ulimit -c 0
DRAWER_RUNS="$RL_OUTPUT_PATH/1_hydroshear/DrawerTaskPulling"
# best checkpoint of a run family by the NUMBER in its name, not by mtime
best_ckpt () { ls "$DRAWER_RUNS"/$1_*/nn/best_sr_*.pth 2>/dev/null | sed -E 's/(.*best_sr_)([0-9.]+)\.pth/\2 \0/' | sort -rn | head -1 | cut -d' ' -f2-; }
