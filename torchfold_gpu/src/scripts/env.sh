#!/bin/bash

# =============================================================================
# Basic Path Settings
# =============================================================================
export PYTHON_BIN=/path/to/torchx_env/bin/python
export JSON_PATH=/path/to/json/dir
export OUTPUT_DIR=/path/to/output/dir
export LOG_DIR=${OUTPUT_DIR}/log
mkdir -p ${OUTPUT_DIR}
mkdir -p ${LOG_DIR}



# =============================================================================
# Database Settings
# =============================================================================
# Configure the following lines according to the paths set in README:
# - DB_DIR: Set when RUN_DATA_PIPELINE=true (see README "Optional: Download AlphaFold Databases")
export DB_DIR=/path/to/database/dir

# Whether to run data pipeline (true/false)
export RUN_DATA_PIPELINE=true



# =============================================================================
# Inference Parameters
# =============================================================================
# Number of diffusion samples for inference
export NUM_DIFFUSION_SAMPLES=2



# =============================================================================
# Inference Weights Configuration
# =============================================================================
# Set exactly one of the following (they are mutually exclusive).
#
# CHECKPOINT_PATH: TorchFold trained checkpoint file (.pt)
# MODEL_DIR: official AlphaFold 3 parameters directory
#
# Option 1: TorchFold trained weights
#   - Set CHECKPOINT_PATH to the .pt file
#   - Comment out MODEL_DIR
# export CHECKPOINT_PATH=/path/to/torchfold_checkpoint.pt

# Option 2: official AlphaFold 3 parameters
#   - Set MODEL_DIR to the AF3 parameters directory
#   - Comment out CHECKPOINT_PATH
export MODEL_DIR=/path/to/AlphaFold3/parameters