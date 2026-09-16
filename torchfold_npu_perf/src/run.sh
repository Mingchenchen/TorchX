#!/usr/bin/env bash
set -eo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"
source scripts/env.sh || {
    echo 'Please place your `env.sh` under `scripts` directory.' >&2
    exit 1
}

# Single card: DEVICES=0, NPUS_PER_NODE=1, USE_DIST=0
# Multi-card: DEVICES=0,1,2,3, NPUS_PER_NODE=4, USE_DIST=1
DEVICES=${DEVICES:-0}
NPUS_PER_NODE=${NPUS_PER_NODE:-1}
USE_DIST=${USE_DIST:-0}
INPUT_NAME=${1:-37aa_2JO9.json}
JSON_PATH=${JSON_PATH:-..}
DB_DIR=${DB_DIR:-databases}
OUTPUT_DIR=${OUTPUT_DIR:-output}
LOG_DIR=${LOG_DIR:-${OUTPUT_DIR}/log}

export CPU_AFFINITY_CONF=1 TASK_QUEUE_ENABLE=2
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export ASCEND_RT_VISIBLE_DEVICES="${DEVICES}"
export USE_DIST
RUNNER=(python)
if [[ "${USE_DIST}" == "1" ]]; then
    RUNNER+=(
        -m torch.distributed.run
        --standalone
        "--nproc_per_node=${NPUS_PER_NODE}"
    )
fi

mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"

if [ -n "${CHECKPOINT_PATH}" ] && [ -n "${MODEL_DIR}" ]; then
    echo "Error: set only one of CHECKPOINT_PATH or MODEL_DIR in env.sh (they are mutually exclusive)." >&2
    exit 1
fi
if [ -z "${CHECKPOINT_PATH}" ] && [ -z "${MODEL_DIR}" ]; then
    echo "Error: set one of CHECKPOINT_PATH or MODEL_DIR in env.sh." >&2
    exit 1
fi

WEIGHT_ARGS=()
if [ -n "${CHECKPOINT_PATH}" ]; then
    WEIGHT_ARGS+=(--checkpoint_path="${CHECKPOINT_PATH}")
    echo "Using TorchFold checkpoint: ${CHECKPOINT_PATH}"
else
    WEIGHT_ARGS+=(--model_dir="${MODEL_DIR}")
    echo "Using official AlphaFold 3 parameters: ${MODEL_DIR}"
fi

echo "Running TorchFold on ${NPUS_PER_NODE} card(s) for ${INPUT_NAME}, NCORES=${NCORES}, USE_DIST=${USE_DIST}"
# Default torch: runs without Fusion_Attention installed.
# For better performance, install Fusion_Attention (see ../torchx_npu/README.md,
# "Accelerate using CANN Fusion_Attention") then set Fusion_Attention below.
"${RUNNER[@]}" run_torchfold.py \
    --run_data_pipeline=False \
    --json_path="${JSON_PATH}/${INPUT_NAME}" \
    --output_dir="${OUTPUT_DIR}" \
    --db_dir="${DB_DIR}" \
    --diffusion_sample_parallel=True \
    --dot_product_attention=torch \
    "${WEIGHT_ARGS[@]}" \
    2>&1 | tee -a "${LOG_DIR}/$(basename "${INPUT_NAME}" .json).log"
