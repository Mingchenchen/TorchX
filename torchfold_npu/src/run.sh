#!/bin/bash
# This script is used to run TorchFold on a single node.

# cd $(dirname "$0")
source scripts/env.sh || { \
    echo 'Please place your `env.sh` under `scripts` directory.'; \
    echo 'You can refer to `env.sh.example` for the content of `env.sh`.'; \
    exit 1; \
}

cur_path=$(pwd)
INPUT_NAME=${1:-37aa_2JO9.json}

export CPU_AFFINITY_CONF=1
export TASK_QUEUE_ENABLE=2
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export LD_PRELOAD=/usr/local/lib/lib/libtcmalloc.so.4
export USE_DIST=0

# model_dir and checkpoint_path are mutually exclusive: set exactly one in env.sh.
# MODEL_DIR: official AlphaFold 3 parameters directory
# CHECKPOINT_PATH: TorchFold trained checkpoint file (.pt)
if [ -n "${CHECKPOINT_PATH}" ] && [ -n "${MODEL_DIR}" ]; then
    echo "Error: set only one of CHECKPOINT_PATH or MODEL_DIR in env.sh (they are mutually exclusive)."
    exit 1
fi
if [ -z "${CHECKPOINT_PATH}" ] && [ -z "${MODEL_DIR}" ]; then
    echo "Error: set one of CHECKPOINT_PATH or MODEL_DIR in env.sh."
    exit 1
fi

MODEL_ARG=""
CKPT_ARG=""
if [ -n "${CHECKPOINT_PATH}" ]; then
    CKPT_ARG="--checkpoint_path=${CHECKPOINT_PATH}"
    echo "Using TorchFold checkpoint: ${CHECKPOINT_PATH}"
else
    MODEL_ARG="--model_dir=${MODEL_DIR}"
    echo "Using official AlphaFold 3 parameters: ${MODEL_DIR}"
fi

# dot_product_attention, options: ["torch", "Fusion_Attention"]
echo "Running TorchFold on single node for ${INPUT_NAME}, NCORES=${NCORES}, RANK=${RANK}"
python run_torchfold.py \
    --run_data_pipeline=False \
    --json_path=${JSON_PATH}/${INPUT_NAME} \
    --output_dir=${OUTPUT_DIR} \
    --db_dir=${DB_DIR} \
    --dot_product_attention=torch \
    ${MODEL_ARG} ${CKPT_ARG} \
    | tee -a ${LOG_DIR}/${INPUT_NAME}.log
