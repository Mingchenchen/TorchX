#!/usr/bin/env bash
# Single-GPU JSON inference (no Slurm binding).
# Usage:
#   INPUT_JSON=/path/to/inputs.json bash scripts/infer/predict_json.sh
#   USE_TEMPLATE=true TEMPLATE_MMCIF_DIR=/path/to/mmcif bash scripts/infer/predict_json.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

AF3_PARAMS_DIR="${AF3_PARAMS_DIR:-./Alphafold3params}"
INPUT_JSON="${INPUT_JSON:-}"
OUT_DIR="${OUT_DIR:-${REPO_ROOT}/outputs/infer_json/$(date +%Y%m%d_%H%M%S)}"
DATASET_NAME="${DATASET_NAME:-json}"

NUM_SAMPLES="${NUM_SAMPLES:-5}"
NUM_SEEDS="${NUM_SEEDS:-1}"
BASE_SEED="${BASE_SEED:-42}"
NUM_RECYCLES="${NUM_RECYCLES:-10}"
DIFFUSION_STEPS="${DIFFUSION_STEPS:-200}"

USE_MSA="${USE_MSA:-false}"
USE_TEMPLATE="${USE_TEMPLATE:-false}"
TEMPLATE_MMCIF_DIR="${TEMPLATE_MMCIF_DIR:-${MMCIF_DIR:-}}"

if [[ -z "${INPUT_JSON}" ]]; then
  echo "ERROR: set INPUT_JSON=/path/to/inputs.json" >&2
  exit 1
fi

MSA_FLAG=(--no-use_msa)
TEMPLATE_FLAG=(--no-use_template)
if [[ "${USE_MSA}" == "true" ]]; then
  MSA_FLAG=(--use_msa)
fi
if [[ "${USE_TEMPLATE}" == "true" ]]; then
  TEMPLATE_FLAG=(--use_template)
fi

CKPT_ARGS=()
if [[ -n "${CKPT:-}" ]]; then
  CKPT_ARGS+=(--ckpt "${CKPT}")
fi

TEMPLATE_ARGS=()
if [[ -n "${TEMPLATE_MMCIF_DIR}" ]]; then
  TEMPLATE_ARGS+=(--template_mmcif_dir "${TEMPLATE_MMCIF_DIR}")
fi

mkdir -p "${OUT_DIR}"

echo "----------------------------------------------"
echo "TorchFold GPU predict_json"
echo "INPUT_JSON=${INPUT_JSON}"
echo "OUT_DIR=${OUT_DIR}"
echo "AF3_PARAMS_DIR=${AF3_PARAMS_DIR} CKPT=${CKPT:-<none>}"
echo "USE_MSA=${USE_MSA} USE_TEMPLATE=${USE_TEMPLATE}"
echo "----------------------------------------------"

python -m torchfold.runner.predict_json \
  --input_json "${INPUT_JSON}" \
  --out_dir "${OUT_DIR}" \
  --dataset_name "${DATASET_NAME}" \
  --af3_params_dir "${AF3_PARAMS_DIR}" \
  "${CKPT_ARGS[@]}" \
  "${MSA_FLAG[@]}" \
  "${TEMPLATE_FLAG[@]}" \
  "${TEMPLATE_ARGS[@]}" \
  --num_samples "${NUM_SAMPLES}" \
  --num_seeds "${NUM_SEEDS}" \
  --base_seed "${BASE_SEED}" \
  --num_recycles "${NUM_RECYCLES}" \
  --diffusion_steps "${DIFFUSION_STEPS}" \
  ${EXTRA_ARGS:-}
