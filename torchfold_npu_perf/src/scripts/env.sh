# NOTE: This script defines the environment variables used by run.sh.
#       Modify the defaults below as needed, or override them before running.

# Activate the conda environment if needed.
# conda activate torchfold

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export TORCHFOLD_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Basic path settings.
export JSON_PATH="${JSON_PATH:-/path/to/json/dir}"
export DB_DIR="${DB_DIR:-/path/to/database/dir}"
export OUTPUT_DIR="${OUTPUT_DIR:-$(pwd)/output}"
export LOG_DIR="${LOG_DIR:-${OUTPUT_DIR}/log}"

# Weights: set exactly one of the following (they are mutually exclusive).
# CHECKPOINT_PATH: TorchFold trained checkpoint file (.pt)
# MODEL_DIR: official AlphaFold 3 parameters directory
# Option 1: TorchFold trained weights
# export CHECKPOINT_PATH="${CHECKPOINT_PATH:-/path/to/torchfold_checkpoint.pt}"
# Option 2: official AlphaFold 3 parameters
export MODEL_DIR="${MODEL_DIR:-/path/to/AlphaFold3/parameters}"

# Single-node device settings.
# Single card: DEVICES=0, NPUS_PER_NODE=1, USE_DIST=0
# Multi-card: DEVICES=0,1,2,3, NPUS_PER_NODE=4, USE_DIST=1
export DEVICES="${DEVICES:-0}"
export NPUS_PER_NODE="${NPUS_PER_NODE:-1}"
export USE_DIST="${USE_DIST:-0}"

export CPU_PER_NUMA="${CPU_PER_NUMA:-32}"
export NCORES="${NCORES:-${CPU_PER_NUMA}}"
