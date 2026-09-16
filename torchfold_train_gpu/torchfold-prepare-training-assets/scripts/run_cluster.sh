#!/usr/bin/env bash
# Polymer entity clustering with CD-HIT (identity from CDHIT_IDENTITY).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/common.sh"
PIPELINE_ROOT="${PIPELINE_ROOT:-$(abspath "${SCRIPT_DIR}/..")}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
CIF_LIST="${CIF_LIST:?CIF_LIST not set}"
CLUSTER_OUT_DIR="${CLUSTER_OUT_DIR:?CLUSTER_OUT_DIR not set}"
CDHIT_IDENTITY="${CDHIT_IDENTITY:?CDHIT_IDENTITY not set (pass --identity to run_all.sh)}"
IDENTITY_PCT="${IDENTITY_PCT:-$(identity_to_pct "${CDHIT_IDENTITY}")}"
CDHIT_WORD_SIZE="${CDHIT_WORD_SIZE:-$(cdhit_word_size "${CDHIT_IDENTITY}")}"
CLUSTER_TXT="${CLUSTER_TXT:-${CLUSTER_OUT_DIR}/clusters-by-entity-${IDENTITY_PCT}.txt}"
LOG_DIR="${LOG_DIR:-$(dirname "${CLUSTER_OUT_DIR}")/logs}"

mkdir -p "${LOG_DIR}" "${CLUSTER_OUT_DIR}"
CPUS="$(ncpus)"
JOBS=$(( CPUS * 9 / 10 ))
(( JOBS < 1 )) && JOBS=1
CDHIT_THREADS=$(( CPUS * 9 / 10 ))
(( CDHIT_THREADS < 1 )) && CDHIT_THREADS=1

if ! CDHIT_BIN="$(find_cdhit)"; then
  echo "Error: cd-hit not found. Set CDHIT_BIN or install cd-hit." >&2
  exit 1
fi
if [[ ! -s "${CIF_LIST}" ]]; then
  echo "Error: CIF list missing/empty: ${CIF_LIST}" >&2
  exit 1
fi

FASTA="${CLUSTER_OUT_DIR}/all_entities.fasta"
NR_FASTA="${CLUSTER_OUT_DIR}/entities_nr${IDENTITY_PCT}.fasta"

echo "[INFO] host=$(hostname) jobs=${JOBS} cdhit_T=${CDHIT_THREADS}"
echo "[INFO] CIF_LIST=${CIF_LIST}"
echo "[INFO] CLUSTER_OUT_DIR=${CLUSTER_OUT_DIR}"
echo "[INFO] CLUSTER_TXT=${CLUSTER_TXT}"
echo "[INFO] identity=${CDHIT_IDENTITY} word=${CDHIT_WORD_SIZE}"
echo "[INFO] CDHIT_BIN=${CDHIT_BIN}"
echo "[INFO] start=$(date '+%F %T')"

echo "[Step1] Extract entity FASTA ..."
"${PYTHON_BIN}" -u "${SCRIPT_DIR}/cif_to_entity_fasta.py" \
  --cif-list "${CIF_LIST}" \
  --output-fasta "${FASTA}" \
  --jobs "${JOBS}"

if [[ ! -s "${FASTA}" ]]; then
  echo "Error: empty FASTA: ${FASTA}" >&2
  exit 1
fi

echo "[Step2] CD-HIT -c ${CDHIT_IDENTITY} -n ${CDHIT_WORD_SIZE} ..."
CDHIT_CMD=("${CDHIT_BIN}"
  -i "${FASTA}" -o "${NR_FASTA}"
  -c "${CDHIT_IDENTITY}" -n "${CDHIT_WORD_SIZE}"
  -T "${CDHIT_THREADS}" -M 0 -d 0)
if command -v stdbuf >/dev/null 2>&1; then
  stdbuf -oL -eL "${CDHIT_CMD[@]}"
else
  "${CDHIT_CMD[@]}"
fi

echo "[Step3] Convert .clstr -> TorchFold cluster txt ..."
"${PYTHON_BIN}" -u "${SCRIPT_DIR}/clstr_to_torchfold_cluster.py" \
  --input-clstr "${NR_FASTA}.clstr" \
  --output-txt "${CLUSTER_TXT}"

echo "[INFO] end=$(date '+%F %T')"
echo "OK: ${CLUSTER_TXT}"
print_cluster_result_paths
