#!/bin/bash
# Batch run TorchCraft mini_binder design tasks (GPU)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# =================== Configuration ===================
REPO_ROOT_DIR="/<path_to_repo>/torchcraft_gpu" # Needs modification by the user
PDB_DIR="${REPO_ROOT_DIR}/task/mini_binder/input/pdl1_8znl.pdb"
HOTSPOT_INDICES="31,55,99,107"
PDB_FULL_SEQ="${REPO_ROOT_DIR}/task/mini_binder/input/input_sequence.csv"
OUT_DIR="${REPO_ROOT_DIR}/task/mini_binder/output"
MAIN_PATH="${REPO_ROOT_DIR}/task/run_task.py"
CONFIG_DIR="${REPO_ROOT_DIR}/task/config"
# Python with torchx installed. Edit the path below, or pass it when running:
#   PYTHON_BIN=/path/to/torchx_env/bin/python bash mini_binder_design.sh
PYTHON_BIN="${PYTHON_BIN:-/path/to/torchx_env/bin/python}"

NUM_TASKS=10
DESIGN_LENGTH_RANGE="60-130"
TASKS_PER_CARD=1
NUM_GPUS=1
BASE_SEED=1
SEED_STEP=1

export PYTHON_BIN

# =================== Validation ===================
if [ ! -f "${PDB_DIR}" ]; then
    echo "Error: PDB not found: ${PDB_DIR}" >&2
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

IFS='-' read -r LEN_MIN LEN_MAX <<< "$DESIGN_LENGTH_RANGE"
if [ -z "$LEN_MAX" ]; then
    LEN_MAX=$LEN_MIN
fi

BATCH_OUTPUT_DIR="${OUT_DIR}/batch_runs_$(date +%Y%m%d_%H%M%S)"
PDB_FILE_NAME=$(basename "${PDB_DIR}")
NUM_JOBS=$(( (NUM_TASKS + TASKS_PER_CARD - 1) / TASKS_PER_CARD ))

echo "Output: ${BATCH_OUTPUT_DIR}"
echo "Tasks: ${NUM_TASKS}, length ${LEN_MIN}-${LEN_MAX}"

run_one_gpu_job() {
    local GLOBAL_JOB_IDX=$1
    local GPU_SLOT=$2
    export CUDA_VISIBLE_DEVICES="${GPU_SLOT}"

    local start_task_idx=$((GLOBAL_JOB_IDX * TASKS_PER_CARD))
    local PIDS=()
    local k

    for ((k=0; k<TASKS_PER_CARD; k++)); do
        local current_task_idx=$((start_task_idx + k))
        if [ "${current_task_idx}" -ge "${NUM_TASKS}" ]; then
            break
        fi

        local CURRENT_SEED=$(( BASE_SEED + current_task_idx * SEED_STEP ))
        if [ "${LEN_MIN}" -eq "${LEN_MAX}" ]; then
            CURRENT_LENGTH=${LEN_MIN}
        else
            DIFF=$(( ${LEN_MAX} - ${LEN_MIN} + 1 ))
            CURRENT_LENGTH=$(( ${LEN_MIN} + RANDOM % DIFF ))
        fi

        local TASK_NAME="seed_${CURRENT_SEED}"
        local TASK_DIR="${BATCH_OUTPUT_DIR}/${TASK_NAME}"
        local LOG_BASE="${TASK_DIR}/gpu${GPU_SLOT}_g${GLOBAL_JOB_IDX}_k${k}"

        mkdir -p "${TASK_DIR}"
        cp "${PDB_DIR}" "${TASK_DIR}/${PDB_FILE_NAME}"

        echo "Starting ${TASK_NAME} (length=${CURRENT_LENGTH}, gpu=${GPU_SLOT})"

        ${PYTHON_BIN} ${MAIN_PATH} \
            --pdb_path="${TASK_DIR}/${PDB_FILE_NAME}" \
            --outdir="${TASK_DIR}" \
            --task_type=mini_binder \
            --config="${CONFIG_DIR}" \
            --design_length=${CURRENT_LENGTH} \
            --hotspot_indices="${HOTSPOT_INDICES}" \
            --target_full_seq=${PDB_FULL_SEQ} \
            --random_seed=${CURRENT_SEED} > "${LOG_BASE}.out" 2> "${LOG_BASE}.err" &

        PIDS+=($!)
    done

    for pid in "${PIDS[@]}"; do
        wait "${pid}"
    done
}

for ((wave_start=0; wave_start<NUM_JOBS; wave_start+=NUM_GPUS)); do
    remaining=$((NUM_JOBS - wave_start))
    (( wave_count = remaining < NUM_GPUS ? remaining : NUM_GPUS ))

    wave_pids=()
    for ((slot=0; slot<wave_count; slot++)); do
        g=$((wave_start + slot))
        run_one_gpu_job "${g}" "${slot}" &
        wave_pids+=($!)
    done
    for pid in "${wave_pids[@]}"; do
        wait "${pid}"
    done
done

echo "Done. Results: ${BATCH_OUTPUT_DIR}"
