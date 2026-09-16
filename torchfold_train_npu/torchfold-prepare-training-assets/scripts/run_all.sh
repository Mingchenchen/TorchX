#!/usr/bin/env bash
# Run the full pipeline locally: MSA convert -> cluster -> Step1.
#
# Usage:
#   bash scripts/run_all.sh --example --identity 0.4
#   bash scripts/run_all.sh \
#     --cif-list /path/to/all_cif_path.txt \
#     --json-dir /path/to/json_w_msa \
#     --output-dir /path/to/run_output \
#     --identity 0.4
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/common.sh"
PIPELINE_ROOT="$(abspath "${SCRIPT_DIR}/..")"
export PIPELINE_ROOT

SKIP_STEP1=0
USE_EXAMPLE=0
WORKERS="${WORKERS:-}"
DISTILLATION="${DISTILLATION:-1}"
CDHIT_IDENTITY=""

usage() {
  cat <<EOF
Usage: $0 --cif-list PATH --json-dir PATH --output-dir PATH --identity FLOAT [options]
   or: $0 --example --identity FLOAT [options]

Required (unless --example):
  --cif-list PATH     CIF list (one path per line)
  --json-dir PATH     TorchFold JSON dir with inline pairedMsa/unpairedMsa
  --output-dir PATH   Output root (msa_torchfold/, cluster_information/, output/)
  --identity FLOAT    CD-HIT sequence identity, e.g. 0.4 or 0.5

Optional:
  --log-dir PATH      Log directory (default: OUTPUT/logs)
  --workers N         CPU workers (default: all cores)
  --skip-step1        Stop after MSA + clustering (skip Step1)
  --no-distillation   Step1 WeightedPDB filters (RCSB mmCIF)
  --example           Run bundled tiny examples under examples/
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
    --workers)    WORKERS="$2"; shift 2 ;;
    --skip-step1) SKIP_STEP1=1; shift ;;
    --no-distillation) DISTILLATION=0; shift ;;
    --example)    USE_EXAMPLE=1; shift ;;
    -h|--help)    usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 1 ;;
  esac
done

if [[ -z "${CDHIT_IDENTITY}" ]]; then
  echo "Error: --identity is required (e.g. --identity 0.4)." >&2
  usage
  exit 1
fi

if [[ "${USE_EXAMPLE}" == "1" ]]; then
  CIF_LIST="${PIPELINE_ROOT}/examples/cif_list.txt"
  JSON_MSA_DIR="${PIPELINE_ROOT}/examples/json_w_msa"
  OUTPUT_RUN_ROOT="${OUTPUT_RUN_ROOT:-${PIPELINE_ROOT}/example_run}"
  LOG_DIR="${LOG_DIR:-${OUTPUT_RUN_ROOT}/logs}"
  if [[ -z "${WORKERS}" ]]; then
    WORKERS=4
  fi
fi

if [[ -z "${CIF_LIST:-}" || -z "${JSON_MSA_DIR:-}" || -z "${OUTPUT_RUN_ROOT:-}" ]]; then
  echo "Error: --cif-list, --json-dir, --output-dir are required (or pass --example)." >&2
  usage
  exit 1
fi

OUTPUT_RUN_ROOT="$(abspath "${OUTPUT_RUN_ROOT}")"
LOG_DIR="$(abspath "${LOG_DIR:-${OUTPUT_RUN_ROOT}/logs}")"
CIF_LIST="$(abspath "${CIF_LIST}")"
JSON_MSA_DIR="$(abspath "${JSON_MSA_DIR}")"
MSA_OUTDIR="${OUTPUT_RUN_ROOT}/msa_torchfold"
CLUSTER_OUT_DIR="${OUTPUT_RUN_ROOT}/cluster_information"
IDENTITY_PCT="$(identity_to_pct "${CDHIT_IDENTITY}")"
CLUSTER_TXT="${CLUSTER_OUT_DIR}/clusters-by-entity-${IDENTITY_PCT}.txt"
CDHIT_WORD_SIZE="$(cdhit_word_size "${CDHIT_IDENTITY}")"

mkdir -p "${LOG_DIR}" "${OUTPUT_RUN_ROOT}" "${MSA_OUTDIR}" "${CLUSTER_OUT_DIR}"

if [[ ! -s "${CIF_LIST}" ]]; then
  echo "Error: CIF list missing or empty: ${CIF_LIST}" >&2
  exit 1
fi
if [[ ! -d "${JSON_MSA_DIR}" ]]; then
  echo "Error: JSON dir not found: ${JSON_MSA_DIR}" >&2
  exit 1
fi

# Rewrite relative CIF paths against the repo root (used by --example).
ABS_CIF_LIST="${OUTPUT_RUN_ROOT}/cif_list.abs.txt"
"${PYTHON_BIN:-python3}" - "${CIF_LIST}" "${PIPELINE_ROOT}" "${ABS_CIF_LIST}" <<'PY'
import sys
from pathlib import Path
src, root, dst = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
lines = []
for raw in src.read_text(encoding="utf-8").splitlines():
    p = raw.strip().split("#", 1)[0].strip()
    if not p:
        continue
    path = Path(p)
    if not path.is_absolute():
        path = (root / path).resolve()
    else:
        path = path.resolve()
    if not path.exists():
        raise SystemExit(f"CIF not found: {path} (from {p})")
    lines.append(str(path))
if not lines:
    raise SystemExit(f"No CIF paths in {src}")
dst.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"[INFO] resolved {len(lines)} CIF paths -> {dst}")
PY
CIF_LIST="${ABS_CIF_LIST}"

export PIPELINE_ROOT CIF_LIST JSON_MSA_DIR OUTPUT_RUN_ROOT LOG_DIR
export MSA_OUTDIR CLUSTER_OUT_DIR CLUSTER_TXT DISTILLATION
export CDHIT_IDENTITY CDHIT_WORD_SIZE IDENTITY_PCT
if [[ -n "${WORKERS}" ]]; then
  export WORKERS
  export SLURM_CPUS_PER_TASK="${WORKERS}"
fi

echo "[INFO] pipeline: MSA -> Cluster (${CDHIT_IDENTITY})$([ "${SKIP_STEP1}" = 1 ] && echo '' || echo ' -> Step1')"
echo "[INFO] OUTPUT_RUN_ROOT=${OUTPUT_RUN_ROOT}"
echo "[INFO] LOG_DIR=${LOG_DIR}"

echo "[STAGE 1/3] MSA convert"
bash "${SCRIPT_DIR}/run_convert_msa.sh" 2>&1 | tee "${LOG_DIR}/convert_msa.log"
if [[ ! -s "${MSA_OUTDIR}/common/seq_to_pdb_index.json" ]]; then
  echo "[ERROR] seq_to_pdb_index.json missing" >&2
  exit 1
fi

echo "[STAGE 2/3] Cluster identity=${CDHIT_IDENTITY}"
bash "${SCRIPT_DIR}/run_cluster.sh" 2>&1 | tee "${LOG_DIR}/cluster.log"
if [[ ! -s "${CLUSTER_TXT}" ]]; then
  echo "[ERROR] cluster file missing: ${CLUSTER_TXT}" >&2
  exit 1
fi

if [[ "${SKIP_STEP1}" == "1" ]]; then
  echo "[DONE] skipped Step1"
  print_msa_result_paths
  print_cluster_result_paths
  exit 0
fi

echo "[STAGE 3/3] Step1"
bash "${SCRIPT_DIR}/run_step1.sh" 2>&1 | tee "${LOG_DIR}/step1.log"

STEP1_ROOT="$(
  ls -td "${OUTPUT_RUN_ROOT}/output"/*/step1_prepare_training_data 2>/dev/null | head -1 || true
)"
if [[ -z "${STEP1_ROOT}" || ! -s "${STEP1_ROOT}/indices.csv" ]]; then
  echo "[ERROR] Step1 indices.csv not found under ${OUTPUT_RUN_ROOT}/output" >&2
  exit 1
fi
echo "[DONE] pipeline completed"
echo "[DONE] indices: ${STEP1_ROOT}/indices.csv"
