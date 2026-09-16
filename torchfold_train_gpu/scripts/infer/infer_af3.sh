#!/usr/bin/env bash
# Multi-GPU sharded inference over a configured test set (no Slurm binding).
# Usage:
#   bash scripts/infer/infer_af3.sh
#   CKPT=/path/to/step.pt NUM_TARGETS=32 bash scripts/infer/infer_af3.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

AF3_PARAMS_DIR="${AF3_PARAMS_DIR:-./Alphafold3params}"
TORCHFOLD_ROOT_DIR="${TORCHFOLD_ROOT_DIR:-./train_data}"
export TORCHFOLD_ROOT_DIR

TEST_SET="${TEST_SET:-recentPDB_1536_sample384_0925}"
NUM_TARGETS="${NUM_TARGETS:-8}"
NUM_SAMPLES="${NUM_SAMPLES:-5}"
NUM_SEEDS="${NUM_SEEDS:-1}"
NUM_RECYCLES="${NUM_RECYCLES:-10}"
DIFFUSION_STEPS="${DIFFUSION_STEPS:-200}"
MAX_N_TOKEN="${MAX_N_TOKEN:-384}"
BASE_SEED="${BASE_SEED:-42}"

NGPU="$(python - <<'PY'
import torch
print(torch.cuda.device_count() if torch.cuda.is_available() else 1)
PY
)"
NUM_SHARDS="${NUM_SHARDS:-${NGPU}}"
OUT_DIR="${OUT_DIR:-${REPO_ROOT}/outputs/infer/$(date +%Y%m%d_%H%M%S)}"
mkdir -p "${OUT_DIR}"

CKPT_ARGS=()
if [[ -n "${CKPT:-}" ]]; then
  CKPT_ARGS+=(--ckpt "${CKPT}")
fi

echo "----------------------------------------------"
echo "TorchFold GPU inference (test set)"
echo "REPO_ROOT=${REPO_ROOT}"
echo "NUM_SHARDS=${NUM_SHARDS} TEST_SET=${TEST_SET} OUT_DIR=${OUT_DIR}"
echo "AF3_PARAMS_DIR=${AF3_PARAMS_DIR} CKPT=${CKPT:-<none>}"
echo "----------------------------------------------"

pids=()
fail=0
for ((i = 0; i < NUM_SHARDS; i++)); do
  GPU_ID=$((i % NGPU))
  LOG="${OUT_DIR}/infer_shard${i}.log"
  echo "launch shard ${i}/${NUM_SHARDS} on GPU ${GPU_ID} -> ${LOG}"
  CUDA_VISIBLE_DEVICES="${GPU_ID}" python -m torchfold.runner.inference \
    --af3_params_dir "${AF3_PARAMS_DIR}" \
    "${CKPT_ARGS[@]}" \
    --test_set "${TEST_SET}" \
    --num_targets "${NUM_TARGETS}" \
    --num_samples "${NUM_SAMPLES}" \
    --num_seeds "${NUM_SEEDS}" \
    --num_recycles "${NUM_RECYCLES}" \
    --diffusion_steps "${DIFFUSION_STEPS}" \
    --max_n_token "${MAX_N_TOKEN}" \
    --base_seed "${BASE_SEED}" \
    --num_shards "${NUM_SHARDS}" \
    --shard_idx "${i}" \
    --out_dir "${OUT_DIR}" \
    ${EXTRA_ARGS:-} \
    >"${LOG}" 2>&1 &
  pids+=("$!")
done

for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    fail=1
  fi
done

if [[ "${fail}" -ne 0 ]]; then
  echo "ERROR: one or more inference shards failed; see ${OUT_DIR}/infer_shard*.log" >&2
  exit 1
fi
echo "All shards finished. Outputs under ${OUT_DIR}"
