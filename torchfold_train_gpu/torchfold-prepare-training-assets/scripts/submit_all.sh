#!/usr/bin/env bash
# Submit the full pipeline via one coordinator Slurm job.
# Coordinator waits for each stage (squeue + sacct), then submits the next.
#
# Usage:
#   bash scripts/submit_all.sh \
#     --cif-list /path/to/all_cif_path.txt \
#     --json-dir /path/to/json_w_msa \
#     --output-dir /path/to/run_output \
#     --log-dir /path/to/logs \
#     --partition PARTITION
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/common.sh"
PIPELINE_ROOT="$(abspath "${SCRIPT_DIR}/..")"
export PIPELINE_ROOT

PARTITION="${PARTITION:-}"
MSA_CPUS="${MSA_CPUS:-32}"
CLUSTER_CPUS="${CLUSTER_CPUS:-32}"
STEP1_CPUS="${STEP1_CPUS:-32}"
MSA_MEM="${MSA_MEM:-200G}"
CLUSTER_MEM="${CLUSTER_MEM:-64G}"
STEP1_MEM="${STEP1_MEM:-80G}"
DISTILLATION="${DISTILLATION:-1}"
CDHIT_IDENTITY=""

usage() {
  cat <<EOF
Usage: $0 --cif-list PATH --json-dir PATH --output-dir PATH --log-dir PATH --identity FLOAT [options]

Required:
  --cif-list PATH     CIF list file (one path per line)
  --json-dir PATH     TorchFold JSON dir with inline pairedMsa/unpairedMsa
  --output-dir PATH   Output root
  --log-dir PATH      Slurm log directory
  --identity FLOAT    CD-HIT sequence identity, e.g. 0.4 or 0.5

Optional:
  --partition PART    Slurm partition
  --msa-cpus N        CPUs for MSA job (default: ${MSA_CPUS})
  --cluster-cpus N    CPUs for cluster job (default: ${CLUSTER_CPUS})
  --step1-cpus N      CPUs for Step1 job (default: ${STEP1_CPUS})
  --msa-mem MEM       Memory for MSA job (default: ${MSA_MEM})
  --cluster-mem MEM   Memory for cluster job (default: ${CLUSTER_MEM})
  --step1-mem MEM     Memory for Step1 job (default: ${STEP1_MEM})
  --no-distillation   Step1 WeightedPDB filters
  -h, --help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --cif-list)   CIF_LIST="$2"; shift 2 ;;
    --json-dir)   JSON_MSA_DIR="$2"; shift 2 ;;
    --output-dir) OUTPUT_RUN_ROOT="$2"; shift 2 ;;
    --log-dir)    LOG_DIR="$2"; shift 2 ;;
    --identity)   CDHIT_IDENTITY="$2"; shift 2 ;;
    --partition)  PARTITION="$2"; shift 2 ;;
    --msa-cpus)   MSA_CPUS="$2"; shift 2 ;;
    --cluster-cpus) CLUSTER_CPUS="$2"; shift 2 ;;
    --step1-cpus) STEP1_CPUS="$2"; shift 2 ;;
    --msa-mem)    MSA_MEM="$2"; shift 2 ;;
    --cluster-mem) CLUSTER_MEM="$2"; shift 2 ;;
    --step1-mem)  STEP1_MEM="$2"; shift 2 ;;
    --no-distillation) DISTILLATION=0; shift ;;
    -h|--help)    usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 1 ;;
  esac
done

if [[ -z "${CIF_LIST:-}" || -z "${JSON_MSA_DIR:-}" || -z "${OUTPUT_RUN_ROOT:-}" || -z "${LOG_DIR:-}" || -z "${CDHIT_IDENTITY}" ]]; then
  echo "Error: --cif-list, --json-dir, --output-dir, --log-dir, --identity are required." >&2
  usage
  exit 1
fi

OUTPUT_RUN_ROOT="$(abspath "${OUTPUT_RUN_ROOT}")"
LOG_DIR="$(abspath "${LOG_DIR}")"
CIF_LIST="$(abspath "${CIF_LIST}")"
JSON_MSA_DIR="$(abspath "${JSON_MSA_DIR}")"
MSA_OUTDIR="${OUTPUT_RUN_ROOT}/msa_torchfold"
CLUSTER_OUT_DIR="${OUTPUT_RUN_ROOT}/cluster_information"
IDENTITY_PCT="$(identity_to_pct "${CDHIT_IDENTITY}")"
CLUSTER_TXT="${CLUSTER_OUT_DIR}/clusters-by-entity-${IDENTITY_PCT}.txt"
CDHIT_WORD_SIZE="$(cdhit_word_size "${CDHIT_IDENTITY}")"
PYTHON_BIN="${PYTHON_BIN:-python3}"
PROTENIX_PYTHON="${PROTENIX_PYTHON:-${PYTHON_BIN}}"
CDHIT_BIN="${CDHIT_BIN:-$(find_cdhit || true)}"
PROTENIX_ROOT="${PROTENIX_ROOT:-}"
PROTENIX_ROOT_DIR="${PROTENIX_ROOT_DIR:-}"

mkdir -p "${LOG_DIR}" "${OUTPUT_RUN_ROOT}" "${MSA_OUTDIR}" "${CLUSTER_OUT_DIR}"

if [[ ! -s "${CIF_LIST}" ]]; then
  echo "Error: CIF list missing or empty: ${CIF_LIST}" >&2
  exit 1
fi
if [[ ! -d "${JSON_MSA_DIR}" ]]; then
  echo "Error: JSON dir not found: ${JSON_MSA_DIR}" >&2
  exit 1
fi

EXPORT_COORD="ALL,CIF_LIST=${CIF_LIST},JSON_MSA_DIR=${JSON_MSA_DIR},OUTPUT_RUN_ROOT=${OUTPUT_RUN_ROOT},LOG_DIR=${LOG_DIR},MSA_OUTDIR=${MSA_OUTDIR},CLUSTER_OUT_DIR=${CLUSTER_OUT_DIR},CLUSTER_TXT=${CLUSTER_TXT},SCRIPT_DIR=${SCRIPT_DIR},PIPELINE_ROOT=${PIPELINE_ROOT},PARTITION=${PARTITION},MSA_CPUS=${MSA_CPUS},CLUSTER_CPUS=${CLUSTER_CPUS},STEP1_CPUS=${STEP1_CPUS},MSA_MEM=${MSA_MEM},CLUSTER_MEM=${CLUSTER_MEM},STEP1_MEM=${STEP1_MEM},PYTHON_BIN=${PYTHON_BIN},PROTENIX_PYTHON=${PROTENIX_PYTHON},CDHIT_BIN=${CDHIT_BIN},PROTENIX_ROOT=${PROTENIX_ROOT},PROTENIX_ROOT_DIR=${PROTENIX_ROOT_DIR},DISTILLATION=${DISTILLATION},CDHIT_IDENTITY=${CDHIT_IDENTITY},CDHIT_WORD_SIZE=${CDHIT_WORD_SIZE},IDENTITY_PCT=${IDENTITY_PCT}"

MANIFEST="${LOG_DIR}/submit_all_$(date +%Y%m%d-%H%M%S).manifest"
{
  echo "# submit_all manifest $(date '+%F %T')"
  echo "CIF_LIST=${CIF_LIST}"
  echo "JSON_MSA_DIR=${JSON_MSA_DIR}"
  echo "OUTPUT_RUN_ROOT=${OUTPUT_RUN_ROOT}"
  echo "LOG_DIR=${LOG_DIR}"
} > "${MANIFEST}"

SBATCH_P=()
if [[ -n "${PARTITION}" ]]; then
  SBATCH_P=(-p "${PARTITION}")
fi

echo "[INFO] Submitting coordinator (MSA -> Cluster -> Step1) ..."
echo "[INFO] CIF_LIST=${CIF_LIST}"
echo "[INFO] OUTPUT_RUN_ROOT=${OUTPUT_RUN_ROOT}"
echo "[INFO] LOG_DIR=${LOG_DIR}"

JOB_COORD="$(
  sbatch --parsable \
    --export="${EXPORT_COORD}" \
    "${SBATCH_P[@]}" \
    -J tfprep-coord \
    --cpus-per-task=1 \
    --mem=4G \
    -o "${LOG_DIR}/coordinator_%j.out" \
    -e "${LOG_DIR}/coordinator_%j.err" \
    "${SCRIPT_DIR}/submit_coordinator.sh"
)"

echo "JOB_COORDINATOR=${JOB_COORD}" >> "${MANIFEST}"

echo "[INFO] Coordinator submitted (safe to close terminal):"
echo "  job=${JOB_COORD}"
echo "  log=${LOG_DIR}/coordinator_${JOB_COORD}.out"
echo "  manifest=${MANIFEST}"
echo
echo "Monitor:"
echo "  tail -f ${LOG_DIR}/coordinator_${JOB_COORD}.out"
echo "  squeue -j ${JOB_COORD}"
