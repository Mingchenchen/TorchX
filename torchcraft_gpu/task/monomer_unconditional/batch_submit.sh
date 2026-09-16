#!/bin/bash
# Batch run TorchCraft monomer design tasks (GPU, no Slurm)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# =================== Configuration ===================
REPO_ROOT_DIR="/<path_to_repo>/torchcraft_gpu" # Needs modification by the user
JSON_PATH="${REPO_ROOT_DIR}/task/monomer_unconditional/test-demo.json"
OUT_DIR="${REPO_ROOT_DIR}/task/monomer_unconditional/output"
MAIN_PATH="${REPO_ROOT_DIR}/task/run_task.py"
CONFIG_DIR="${REPO_ROOT_DIR}/task/config"
# Python with torchx installed. Edit the path below, or pass it when running:
#   PYTHON_BIN=/path/to/torchx_env/bin/python bash batch_submit.sh
PYTHON_BIN="${PYTHON_BIN:-/path/to/torchx_env/bin/python}"

NUM_TASKS=1
BASE_SEED=1
SEED_STEP=1
DESIGN_LENGTH=100

export PYTHON_BIN

# =================== Validation ===================
if [ ! -f "${JSON_PATH}" ]; then
    echo "Error: JSON not found: ${JSON_PATH}" >&2
    exit 1
fi
if [ ! -f "${MAIN_PATH}" ]; then
    echo "Error: Main program not found: ${MAIN_PATH}" >&2
    exit 1
fi
if [ ! -x "${PYTHON_BIN}" ] && [ ! -f "${PYTHON_BIN}" ]; then
    echo "Error: Python not found: ${PYTHON_BIN}" >&2
    exit 1
fi

BATCH_OUTPUT_DIR="${OUT_DIR}/batch_runs_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${BATCH_OUTPUT_DIR}"

echo "Output: ${BATCH_OUTPUT_DIR}"
echo "Tasks: ${NUM_TASKS}, design_length=${DESIGN_LENGTH}"

for i in $(seq 0 $((NUM_TASKS - 1))); do
    SEED=$((BASE_SEED + i * SEED_STEP))
    TASK_NAME="seed_${SEED}"
    TASK_DIR="${BATCH_OUTPUT_DIR}/${TASK_NAME}"

    echo "Starting task $((i + 1))/${NUM_TASKS}: ${TASK_NAME} (seed=${SEED})"

    mkdir -p "${TASK_DIR}"
    cp "${JSON_PATH}" "${TASK_DIR}/test-demo.json"

    ${PYTHON_BIN} ${MAIN_PATH} \
        --outdir="${TASK_DIR}" \
        --task_type=monomer \
        --config="${CONFIG_DIR}" \
        --random_seed="${SEED}" \
        --design_length="${DESIGN_LENGTH}" \
        --json_path="${TASK_DIR}/test-demo.json"

    rm -f "${TASK_DIR}/test-demo.json"
    echo "Task ${TASK_NAME} completed"
done

echo "Done. Results: ${BATCH_OUTPUT_DIR}"
