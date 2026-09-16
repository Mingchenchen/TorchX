#!/bin/bash
# Batch run TorchCraft VHH design tasks (GPU)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# =================== Configuration ===================
REPO_ROOT_DIR="/<path_to_torchcraft_repo>/torchcraft_gpu" # Needs modification by the user
PDB_DIR="${REPO_ROOT_DIR}/task/vhh/input/bhrf1_2wh6.pdb"
HOTSPOT_INDICES="63,70,79,86,105"
PDB_FULL_SEQ="${REPO_ROOT_DIR}/task/vhh/input/input_sequence.csv"
OUT_DIR="${REPO_ROOT_DIR}/task/vhh/output"
MAIN_PATH="${REPO_ROOT_DIR}/task/run_task.py"
CONFIG_DIR="${REPO_ROOT_DIR}/task/config"
# Python with torchx installed. Edit the path below, or pass it when running:
#   PYTHON_BIN=/path/to/torchx_env/bin/python bash vhh_design.sh
PYTHON_BIN="${PYTHON_BIN:-/path/to/torchx_env/bin/python}"

FRAMEWORK_DIR="${REPO_ROOT_DIR}/task/vhh/input/frameworks"
FRAMEWORKS=(
    "${FRAMEWORK_DIR}/7eow_nanobody_framework.cif"
    "${FRAMEWORK_DIR}/5jds_nanobody_framework.cif"
    "${FRAMEWORK_DIR}/7xl0_nanobody_framework.cif"
    "${FRAMEWORK_DIR}/8coh_nanobody_framework.cif"
    "${FRAMEWORK_DIR}/8z8v_nanobody_framework.cif"
)

NUM_TASKS=100
TASKS_PER_FRAMEWORK=20
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

NUM_FRAMEWORKS=${#FRAMEWORKS[@]}
BATCH_OUTPUT_DIR="${OUT_DIR}/batch_runs_$(date +%Y%m%d_%H%M%S)"
PDB_FILE_NAME=$(basename "${PDB_DIR}")
NUM_JOBS=$(( (NUM_TASKS + TASKS_PER_CARD - 1) / TASKS_PER_CARD ))

echo "Output: ${BATCH_OUTPUT_DIR}"
echo "Tasks: ${NUM_TASKS} (${NUM_FRAMEWORKS} frameworks x ${TASKS_PER_FRAMEWORK})"

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
        local task_idx_from_zero=$((CURRENT_SEED - BASE_SEED))
        local framework_idx=$((task_idx_from_zero / TASKS_PER_FRAMEWORK))
        local CURRENT_FRAMEWORK_PATH="${FRAMEWORKS[$framework_idx]}"

        if (( framework_idx < 0 || framework_idx >= NUM_FRAMEWORKS )); then
            echo "Skip seed=${CURRENT_SEED}: framework index out of range" >&2
            continue
        fi
        if [ ! -f "${CURRENT_FRAMEWORK_PATH}" ]; then
            echo "Skip seed=${CURRENT_SEED}: framework not found: ${CURRENT_FRAMEWORK_PATH}" >&2
            continue
        fi

        local FW_FILENAME
        FW_FILENAME=$(basename "${CURRENT_FRAMEWORK_PATH}")
        local FW_ID=${FW_FILENAME%%_*}
        local TASK_NAME="seed_${CURRENT_SEED}_${FW_ID}"
        local TASK_DIR="${BATCH_OUTPUT_DIR}/${TASK_NAME}"
        local LOG_BASE="${TASK_DIR}/gpu${GPU_SLOT}_g${GLOBAL_JOB_IDX}_k${k}"

        mkdir -p "${TASK_DIR}"
        cp "${PDB_DIR}" "${TASK_DIR}/${PDB_FILE_NAME}"

        echo "Starting ${TASK_NAME} (framework=${FW_ID}, gpu=${GPU_SLOT})"

        ${PYTHON_BIN} ${MAIN_PATH} \
            --pdb_path="${TASK_DIR}/${PDB_FILE_NAME}" \
            --outdir="${TASK_DIR}" \
            --random_seed=${CURRENT_SEED} \
            --task_type=vhh \
            --config="${CONFIG_DIR}" \
            --hotspot_indices="${HOTSPOT_INDICES}" \
            --target_full_seq=${PDB_FULL_SEQ} \
            --framework_cif_path="${CURRENT_FRAMEWORK_PATH}" > "${LOG_BASE}.out" 2> "${LOG_BASE}.err" &

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
