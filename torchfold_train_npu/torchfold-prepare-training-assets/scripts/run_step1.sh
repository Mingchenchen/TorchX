#!/usr/bin/env bash
# Step1: CIF -> bioassembly PKL + indices.csv
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/common.sh"
PIPELINE_ROOT="${PIPELINE_ROOT:-$(abspath "${SCRIPT_DIR}/..")}"

PROTENIX_PYTHON="${PROTENIX_PYTHON:-${PYTHON_BIN:-python3}}"
CIF_LIST="${CIF_LIST:?CIF_LIST not set}"
OUTPUT_RUN_ROOT="${OUTPUT_RUN_ROOT:?OUTPUT_RUN_ROOT not set}"
CLUSTER_TXT="${CLUSTER_TXT:?CLUSTER_TXT not set}"
LOG_DIR="${LOG_DIR:-${OUTPUT_RUN_ROOT}/logs}"
DISTILLATION="${DISTILLATION:-1}"

if [[ ! -s "${CIF_LIST}" ]]; then
  echo "Error: CIF list missing/empty: ${CIF_LIST}" >&2
  exit 1
fi
if [[ ! -s "${CLUSTER_TXT}" ]]; then
  echo "Error: cluster file missing: ${CLUSTER_TXT}" >&2
  echo "Finish clustering first, or run scripts/run_all.sh" >&2
  exit 1
fi

if [[ -n "${PROTENIX_ROOT:-}" ]]; then
  export PYTHONPATH="${PROTENIX_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
fi
if [[ -n "${PROTENIX_ROOT_DIR:-}" ]]; then
  export PROTENIX_ROOT_DIR
fi

mkdir -p "${LOG_DIR}" "${OUTPUT_RUN_ROOT}/output"

RUN_ID="$(date +%Y%m%d-%H%M%S)-${SLURM_JOB_ID:-local}"
RUN_ROOT="${OUTPUT_RUN_ROOT}/output/${RUN_ID}"
STEP1_ROOT="${RUN_ROOT}/step1_prepare_training_data"
STEP1_BIO_DIR="${STEP1_ROOT}/bioassembly"
STEP1_INDICES_CSV="${STEP1_ROOT}/indices.csv"
mkdir -p "${STEP1_BIO_DIR}"

TOTAL_CPUS="$(ncpus)"
NUM_WORKERS=$(( TOTAL_CPUS * 8 / 10 ))
(( NUM_WORKERS < 1 )) && NUM_WORKERS=1

DISTILL_FLAG=()
if [[ "${DISTILLATION}" == "1" ]]; then
  DISTILL_FLAG=(-d)
fi

echo "[INFO] host=$(hostname)"
echo "[INFO] RUN_ROOT=${RUN_ROOT}"
echo "[INFO] CIF_LIST=${CIF_LIST}"
echo "[INFO] CLUSTER_TXT=${CLUSTER_TXT}"
echo "[INFO] workers=${NUM_WORKERS}"
echo "[INFO] distillation=${DISTILLATION}"
echo "[INFO] PROTENIX_ROOT_DIR=${PROTENIX_ROOT_DIR:-}"
echo "[INFO] start=$(date '+%F %T')"

"${PROTENIX_PYTHON}" "${SCRIPT_DIR}/prepare_step1.py" \
  -i "${CIF_LIST}" \
  -o "${STEP1_INDICES_CSV}" \
  -b "${STEP1_BIO_DIR}" \
  -n "${NUM_WORKERS}" \
  -c "${CLUSTER_TXT}" \
  "${DISTILL_FLAG[@]}"

echo "[INFO] end=$(date '+%F %T')"
echo "OK: ${STEP1_ROOT}"
print_pipeline_result_paths
