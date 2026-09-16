#!/usr/bin/env bash
# Coordinator: submit 1 -> 2 -> 3 sequentially after each job_id completes.
set -euo pipefail

SCRIPT_DIR="${SCRIPT_DIR:?SCRIPT_DIR not set}"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/common.sh"
PIPELINE_ROOT="${PIPELINE_ROOT:?PIPELINE_ROOT not set}"

PARTITION="${PARTITION:-}"
MSA_CPUS="${MSA_CPUS:-32}"
CLUSTER_CPUS="${CLUSTER_CPUS:-32}"
STEP1_CPUS="${STEP1_CPUS:-32}"
MSA_MEM="${MSA_MEM:-200G}"
CLUSTER_MEM="${CLUSTER_MEM:-64G}"
STEP1_MEM="${STEP1_MEM:-80G}"

SBATCH_P=()
if [[ -n "${PARTITION}" ]]; then
  SBATCH_P=(-p "${PARTITION}")
fi

EXPORT_COMMON="ALL,CIF_LIST=${CIF_LIST},JSON_MSA_DIR=${JSON_MSA_DIR},OUTPUT_RUN_ROOT=${OUTPUT_RUN_ROOT},LOG_DIR=${LOG_DIR},MSA_OUTDIR=${MSA_OUTDIR},CLUSTER_OUT_DIR=${CLUSTER_OUT_DIR},CLUSTER_TXT=${CLUSTER_TXT},SCRIPT_DIR=${SCRIPT_DIR},PIPELINE_ROOT=${PIPELINE_ROOT},PYTHON_BIN=${PYTHON_BIN:-python3},PROTENIX_PYTHON=${PROTENIX_PYTHON:-${PYTHON_BIN:-python3}},CDHIT_BIN=${CDHIT_BIN:-},PROTENIX_ROOT=${PROTENIX_ROOT:-},PROTENIX_ROOT_DIR=${PROTENIX_ROOT_DIR:-},DISTILLATION=${DISTILLATION:-1},CDHIT_IDENTITY=${CDHIT_IDENTITY},CDHIT_WORD_SIZE=${CDHIT_WORD_SIZE:-},IDENTITY_PCT=${IDENTITY_PCT:-}"

wait_job() {
  local jid="$1"
  local name="$2"
  echo "[WAIT] ${name} job_id=${jid} start=$(date '+%F %T')"
  while squeue -j "${jid}" -h 2>/dev/null | grep -q .; do
    sleep 30
  done
  local state exitcode
  read -r state exitcode < <(
    sacct -j "${jid}" -X --format=State,ExitCode -n -P 2>/dev/null \
      | awk -F'|' 'NF>=2 {print $1, $2; exit}'
  )
  state="${state:-UNKNOWN}"
  exitcode="${exitcode:-?}"
  local ec="${exitcode%%:*}"
  if [[ "${state}" != "COMPLETED" || ( -n "${ec}" && "${ec}" != "0" ) ]]; then
    echo "[ERROR] ${name} job_id=${jid} failed state=${state} exitcode=${exitcode}" >&2
    echo "[ERROR] check logs under: ${LOG_DIR}/" >&2
    exit 1
  fi
  echo "[OK] ${name} job_id=${jid} state=${state} exitcode=${exitcode} end=$(date '+%F %T')"
}

check_file() {
  local path="$1"
  local label="$2"
  if [[ ! -s "${path}" ]]; then
    echo "[ERROR] ${label} missing or empty: ${path}" >&2
    exit 1
  fi
  echo "[CHECK] ${label} ok: ${path}"
}

echo "[INFO] coordinator host=$(hostname) slurm_job=${SLURM_JOB_ID:-local}"
echo "[INFO] pipeline: MSA -> Cluster -> Step1"
echo "[INFO] OUTPUT_RUN_ROOT=${OUTPUT_RUN_ROOT}"
echo "[INFO] LOG_DIR=${LOG_DIR}"

JOB_MSA="$(
  sbatch --parsable \
    --export="${EXPORT_COMMON}" \
    "${SBATCH_P[@]}" \
    -J tfprep-msa \
    --cpus-per-task="${MSA_CPUS}" \
    --mem="${MSA_MEM}" \
    -o "${LOG_DIR}/convert_msa_%j.out" \
    -e "${LOG_DIR}/convert_msa_%j.err" \
    "${SCRIPT_DIR}/run_convert_msa.sh"
)"
echo "JOB_MSA=${JOB_MSA}"
wait_job "${JOB_MSA}" "MSA"
check_file "${MSA_OUTDIR}/common/seq_to_pdb_index.json" "SEQ_TO_PDB_INDEX"

JOB_CLUSTER="$(
  sbatch --parsable \
    --export="${EXPORT_COMMON}" \
    "${SBATCH_P[@]}" \
    -J tfprep-clu40 \
    --cpus-per-task="${CLUSTER_CPUS}" \
    --mem="${CLUSTER_MEM}" \
    -o "${LOG_DIR}/cluster_%j.out" \
    -e "${LOG_DIR}/cluster_%j.err" \
    "${SCRIPT_DIR}/run_cluster.sh"
)"
echo "JOB_CLUSTER=${JOB_CLUSTER}"
wait_job "${JOB_CLUSTER}" "Cluster"
check_file "${CLUSTER_TXT}" "CLUSTERS_BY_ENTITY_40"

JOB_STEP1="$(
  sbatch --parsable \
    --export="${EXPORT_COMMON}" \
    "${SBATCH_P[@]}" \
    -J tfprep-step1 \
    --cpus-per-task="${STEP1_CPUS}" \
    --mem="${STEP1_MEM}" \
    -o "${LOG_DIR}/step1_%j.out" \
    -e "${LOG_DIR}/step1_%j.err" \
    "${SCRIPT_DIR}/run_step1.sh"
)"
echo "JOB_STEP1=${JOB_STEP1}"
wait_job "${JOB_STEP1}" "Step1"

STEP1_ROOT="$(
  ls -td "${OUTPUT_RUN_ROOT}/output"/*/step1_prepare_training_data 2>/dev/null \
    | head -1 || true
)"
if [[ -z "${STEP1_ROOT}" || ! -d "${STEP1_ROOT}" ]]; then
  echo "[ERROR] Step1 output not found under ${OUTPUT_RUN_ROOT}/output" >&2
  exit 1
fi
STEP1_BIO_DIR="${STEP1_ROOT}/bioassembly"
STEP1_INDICES_CSV="${STEP1_ROOT}/indices.csv"
check_file "${STEP1_INDICES_CSV}" "INDICES_CSV"
print_pipeline_result_paths

echo "[DONE] pipeline completed"
echo "JOB_MSA=${JOB_MSA} JOB_CLUSTER=${JOB_CLUSTER} JOB_STEP1=${JOB_STEP1}"
