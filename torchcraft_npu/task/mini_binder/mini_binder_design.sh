#!/bin/bash
# Batch run TorchFold design tasks
# Huawei Ascend 910B NPU, no Slurm

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# ---------- Terminal Output ----------
_ts() { date '+%H:%M:%S'; }
_ui_hr() { printf '%s\n' '────────────────────────────────────────'; }
_ui_title() { _ui_hr; printf '  %s\n' "$1"; _ui_hr; }
_ui_kv() { printf '  %-16s %s\n' "$1" "$2"; }
# In parallel job: Time | NPU | Global Job | Message
_ui_w() { printf '[%s][NPU%s][g%s] %s\n' "$(_ts)" "$1" "$2" "$3"; }
_ui_ok() { printf '[%s] Completed %s\n' "$(_ts)" "$1"; }
_ui_err() { printf '[%s] Error %s\n' "$(_ts)" "$1" >&2; }

# =================== Configuration Parameters ===================
# Adjust for each task >>>
REPO_ROOT_DIR="/<path_to_repo>/torchcraft_npu" # Needs modification by the user
PDB_DIR="${REPO_ROOT_DIR}/task/mini_binder/input/pdl1_8znl.pdb"
HOTSPOT_INDICES="31,55,99,107"
PDB_FULL_SEQ="${REPO_ROOT_DIR}/task/mini_binder/input/input_sequence.csv"
OUT_DIR="${REPO_ROOT_DIR}/task/mini_binder/output"
MAIN_PATH="${REPO_ROOT_DIR}/task/run_task.py"
CONFIG_DIR="${REPO_ROOT_DIR}/task/config"

# Python with torchx installed. Edit the path below, or pass it when running:
#   PYTHON_BIN=/path/to/torchx_env/bin/python bash mini_binder_design.sh
PYTHON_BIN="${PYTHON_BIN:-/path/to/torchx_env/bin/python}"

NUM_TASKS=60           # Sampling count
DESIGN_LENGTH_RANGE="90-150"
# Adjust for each task <<<

# Adjust according to environment >>>
TASKS_PER_CARD=2        # 910b=2 910c=4
NUM_NPUS=2              # Number of NPU cards, use npu-smi info to get
BASE_SEED=80000         # Starting seed value
SEED_STEP=1             # Seed step size

export PYTHON_BIN

# CANN toolkit. Edit below, or pass it when running:
#   ASCEND_TOOLKIT_HOME=/path/to/ascend-toolkit bash mini_binder_design.sh
ASCEND_TOOLKIT_HOME="${ASCEND_TOOLKIT_HOME:-/usr/local/Ascend/ascend-toolkit}"
source "${ASCEND_TOOLKIT_HOME}/set_env.sh"
# Adjust according to environment <<<

# --- Startup validation and summary ---
if [ ! -f "${PDB_DIR}" ]; then
    _ui_err "PDB not found: ${PDB_DIR}"
    exit 1
fi
if [ ! -f "${MAIN_PATH}" ]; then
    _ui_err "Main program not found: ${MAIN_PATH}"
    exit 1
fi
if [ ! -x "${PYTHON_BIN}" ] && [ ! -f "${PYTHON_BIN}" ]; then
    _ui_err "Python not found: ${PYTHON_BIN}"
    exit 1
fi
# Parse design length range
IFS='-' read -r LEN_MIN LEN_MAX <<< "$DESIGN_LENGTH_RANGE"
if [ -z "$LEN_MAX" ]; then
    LEN_MAX=$LEN_MIN
fi
echo "Design length range: ${LEN_MIN} - ${LEN_MAX}"

BATCH_OUTPUT_DIR="${OUT_DIR}/batch_runs_$(date +%Y%m%d_%H%M%S)"

NUM_JOBS=$(( (NUM_TASKS + TASKS_PER_CARD - 1) / TASKS_PER_CARD ))
NUM_WAVES=$(( (NUM_JOBS + NUM_NPUS - 1) / NUM_NPUS ))
PDB_FILE_NAME=$(basename "${PDB_DIR}")

SECONDS=0
_ui_title "TorchFold Batch Design · Startup"
_ui_kv "Host" "$(hostname)"
_ui_kv "Output" "${BATCH_OUTPUT_DIR}"
_ui_kv "Task" "${NUM_TASKS} designs · ${NUM_JOBS} NPU jobs · ${NUM_WAVES} waves"
_ui_kv "Parallel" "${TASKS_PER_CARD} subprocesses per NPU · Max ${NUM_NPUS} cards per wave"
_ui_kv "PDB" "${PDB_FILE_NAME}"
_ui_kv "Python" "${PYTHON_BIN}"
printf '\n'

run_one_npu_job() {
    local GLOBAL_JOB_IDX=$1
    local NPU_SLOT=$2
    export ASCEND_RT_VISIBLE_DEVICES="${NPU_SLOT}"

    _ui_w "${NPU_SLOT}" "${GLOBAL_JOB_IDX}" "Start · Slot ${TASKS_PER_CARD}"

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
        local LOG_BASE="${TASK_DIR}/npu${NPU_SLOT}_g${GLOBAL_JOB_IDX}_k${k}"

        mkdir -p "${TASK_DIR}"
        cp "${PDB_DIR}" "${TASK_DIR}/${PDB_FILE_NAME}"


        ${PYTHON_BIN} ${MAIN_PATH} \
            --pdb_path="${TASK_DIR}/${PDB_FILE_NAME}" \
            --outdir="${TASK_DIR}" \
            --task_type=mini_binder \
            --config="${CONFIG_DIR}" \
            --design_length=${CURRENT_LENGTH} \
            --hotspot_indices="${HOTSPOT_INDICES}" \
            --target_full_seq="${PDB_FULL_SEQ}" \
            --random_seed=${CURRENT_SEED}  > "${LOG_BASE}.out" 2> "${LOG_BASE}.err" &



        PIDS+=($!)
        _ui_w "${NPU_SLOT}" "${GLOBAL_JOB_IDX}" "Started ${TASK_NAME} · pid=$!"
    done

    if ((${#PIDS[@]} > 0)); then
        _ui_w "${NPU_SLOT}" "${GLOBAL_JOB_IDX}" "Waiting for ${#PIDS[@]} subprocesses..."
        for pid in "${PIDS[@]}"; do
            wait "${pid}"
        done
    fi
    _ui_w "${NPU_SLOT}" "${GLOBAL_JOB_IDX}" "NPU job finished"
}

wave_idx=1
for ((wave_start=0; wave_start<NUM_JOBS; wave_start+=NUM_NPUS)); do
    remaining=$((NUM_JOBS - wave_start))
    (( wave_count = remaining < NUM_NPUS ? remaining : NUM_NPUS ))
    wave_end=$((wave_start + wave_count - 1))

    _ui_title "Wave ${wave_idx}/${NUM_WAVES} · Global Jobs ${wave_start}-${wave_end}(${wave_count} parallel)"
    wave_pids=()
    for ((slot=0; slot<wave_count; slot++)); do
        g=$((wave_start + slot))
        run_one_npu_job "${g}" "${slot}" &
        wave_pids+=($!)
    done
    for pid in "${wave_pids[@]}"; do
        wait "${pid}"
    done
    _ui_ok "Wave ${wave_idx}/${NUM_WAVES} completed"
    printf '\n'
    ((wave_idx++))
done

_ui_title "All tasks completed"
_ui_kv "Time elapsed" "${SECONDS}s"
_ui_kv "Result directory" "${BATCH_OUTPUT_DIR}"
_ui_kv "Logs" "in each task directory npu*_g*_k*.out / .err"
_ui_hr
