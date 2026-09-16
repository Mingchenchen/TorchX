#!/usr/bin/env bash
# Convert TorchFold JSON inline MSA -> TorchFold mmcif_msa_template layout.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/common.sh"
PIPELINE_ROOT="${PIPELINE_ROOT:-$(abspath "${SCRIPT_DIR}/..")}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
CIF_LIST="${CIF_LIST:?CIF_LIST not set}"
JSON_MSA_DIR="${JSON_MSA_DIR:?JSON_MSA_DIR not set}"
MSA_OUTDIR="${MSA_OUTDIR:?MSA_OUTDIR not set}"
LOG_DIR="${LOG_DIR:-$(dirname "${MSA_OUTDIR}")/logs}"

mkdir -p "${LOG_DIR}" "${MSA_OUTDIR}"
WORKERS="${WORKERS:-$(ncpus)}"

echo "[INFO] host=$(hostname) workers=${WORKERS}"
echo "[INFO] CIF_LIST=${CIF_LIST}"
echo "[INFO] JSON_MSA_DIR=${JSON_MSA_DIR}"
echo "[INFO] MSA_OUTDIR=${MSA_OUTDIR}"
echo "[INFO] start=$(date '+%F %T')"

"${PYTHON_BIN}" -u "${SCRIPT_DIR}/convert_torchfold_json_msa.py" \
  --cif-list "${CIF_LIST}" \
  --json-dir "${JSON_MSA_DIR}" \
  --outdir "${MSA_OUTDIR}" \
  --workers "${WORKERS}" \
  --skip-existing

echo "[INFO] end=$(date '+%F %T')"
echo "OK: ${MSA_OUTDIR}"
print_msa_result_paths
