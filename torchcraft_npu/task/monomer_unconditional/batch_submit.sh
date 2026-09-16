#!/bin/bash

# Batch submit TorchFold design tasks - Use different random seeds
# You can adjust the number of tasks by modifying the NUM_TASKS variable
export CPU_AFFINITY_CONF=1
export TASK_QUEUE_ENABLE=2
export COMBINED_ENABLE=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

# ===================Configuration Parameters===================
NUM_TASKS=1   # Number of tasks, you can modify this value
BASE_SEED=1   # The starting seed value
SEED_STEP=1   # Seed value interval

# Set base directory
BASE_DIR="/<path_to_torchcraft_repo>/torchcraft_npu" # Needs modification by the user
# Python with torchx installed. Edit the path below, or pass it when running:
#   PYTHON_BIN=/path/to/torchx_env/bin/python bash batch_submit.sh
PYTHON_BIN="${PYTHON_BIN:-/path/to/torchx_env/bin/python}"
ORIGINAL_JSON="${BASE_DIR}/task/monomer_unconditional/test-demo.json"
BATCH_OUTPUT_DIR="${BASE_DIR}/task/monomer_unconditional/batch_runs_$(date +%Y%m%d_%H%M%S)"

export PYTHON_BIN
# CANN toolkit. Edit below, or pass it when running:
#   ASCEND_TOOLKIT_HOME=/path/to/ascend-toolkit bash batch_submit.sh
ASCEND_TOOLKIT_HOME="${ASCEND_TOOLKIT_HOME:-/usr/local/Ascend/ascend-toolkit}"
source "${ASCEND_TOOLKIT_HOME}/set_env.sh"

# ===============================================

# Check if the original JSON file exists
if [ ! -f "${ORIGINAL_JSON}" ]; then
    echo "ERROR: Cannot find the original JSON file: ${ORIGINAL_JSON}"
    exit 1
fi
if [ ! -x "${PYTHON_BIN}" ] && [ ! -f "${PYTHON_BIN}" ]; then
    echo "ERROR: Python not found: ${PYTHON_BIN}"
    exit 1
fi

# Create batch output directory
mkdir -p "${BATCH_OUTPUT_DIR}"
echo "Created batch output directory: ${BATCH_OUTPUT_DIR}"

echo "Starting to submit ${NUM_TASKS} tasks"

# Loop to submit tasks
for i in $(seq 0 $((NUM_TASKS-1))); do
    SEED=$((BASE_SEED + i * SEED_STEP))
    TASK_NAME="seed_${SEED}"
    TASK_DIR="${BATCH_OUTPUT_DIR}/${TASK_NAME}"
    
    echo "Preparing to submit task $((i+1))/${NUM_TASKS}: ${TASK_NAME} (seed=${SEED})"
    
    # Create task-specific directory
    mkdir -p "${TASK_DIR}"
    
    # Copy original JSON file to the task directory
    cp "${ORIGINAL_JSON}" "${TASK_DIR}/test-demo.json"
    echo "  Copied JSON file to: ${TASK_DIR}/test-demo.json"

    # Switch to the task directory
    cd "${TASK_DIR}"
    
    # Run TorchCraft design task
    ${PYTHON_BIN} ${BASE_DIR}/task/run_task.py \
        --outdir="${TASK_DIR}" \
        --task_type=monomer \
        --config="${BASE_DIR}/task/config" \
        --random_seed="${SEED}" \
        --design_length=100 \
        --json_path="${TASK_DIR}/test-demo.json"

    echo "Task ${TASK_NAME} completed"
    rm -f "${TASK_DIR}/test-demo.json"
    echo "Removed copied file ${TASK_DIR}/test-demo.json"
done

echo ""
echo "=========================================="
echo "Batch submission completed!"
echo "=========================================="
echo "Total tasks submitted: ${NUM_TASKS}"
echo "Batch output directory: ${BATCH_OUTPUT_DIR}"
echo "Seed range: ${BASE_SEED} to $((BASE_SEED + (NUM_TASKS-1) * SEED_STEP)) (interval ${SEED_STEP})"
echo ""
echo "Use the following command to view task output:"
echo "  ls ${BATCH_OUTPUT_DIR}/seed_*"
