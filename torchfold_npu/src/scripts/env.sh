# NOTE: This script is the template for the environment variables used in the inference script.
#       You may copy this file to `env.sh` and modify the variables as needed.

# cd $(dirname "$0")

# Activate the conda environment
# conda activate torchfold

# Basic path settings
export JSON_PATH=/path/to/json/dir
export DB_DIR=/path/to/database/dir
export OUTPUT_DIR=$(pwd)/output
export LOG_DIR=${OUTPUT_DIR}/log
mkdir -p ${OUTPUT_DIR}
mkdir -p ${LOG_DIR}

# Weights: set exactly one of the following (they are mutually exclusive).
# CHECKPOINT_PATH: TorchFold trained checkpoint file (.pt)
# MODEL_DIR: official AlphaFold 3 parameters directory
#
# Option 1: TorchFold trained weights
# export CHECKPOINT_PATH=/path/to/torchfold_checkpoint.pt
#
# Option 2: official AlphaFold 3 parameters
export MODEL_DIR=/path/to/AlphaFold3/parameters
export CPU_PER_NUMA=32 # $(lscpu | grep socket | cut -d ' ' -f 22)

# Distributed settings
export MASTER_PORT=9539
export MASTER_ADDR=localhost
export NNODES=1
export NODE_RANK=0

# Single node settings
export NCORES=${CPU_PER_NUMA}
export RANK=0
