#!/usr/bin/env bash
# Generic multi-GPU training launcher (no Slurm / cluster binding).
# Usage:
#   bash scripts/ft_abag_full_data.sh
#   NNODES=1 NPROC_PER_NODE=8 RESUME_CKPT=/path/to/step.pt bash scripts/ft_abag_full_data.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

NNODES="${NNODES:-1}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29501}"

AF3_PARAMS_DIR="${AF3_PARAMS_DIR:-./Alphafold3params}"
TORCHFOLD_ROOT_DIR="${TORCHFOLD_ROOT_DIR:-./train_data}"
export TORCHFOLD_ROOT_DIR

RUN_DATE="${RUN_DATE:-$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="${OUT_DIR:-${REPO_ROOT}/outputs/abag_full/${RUN_DATE}}"
mkdir -p "${OUT_DIR}"

TRAIN_SETS="${TRAIN_SETS:-sabdab,weightedPDB_before210930_v20260101,distill_20260609}"
TRAIN_SAMPLE_WEIGHTS="${TRAIN_SAMPLE_WEIGHTS:-0.34,0.33,0.33}"
TEST_SETS="${TEST_SETS:-abag_2025}"
NUM_DL_WORKERS="${NUM_DL_WORKERS:-2}"
SAMPLE_DIFFUSION_CHUNK_SIZE="${SAMPLE_DIFFUSION_CHUNK_SIZE:-5}"
WANDB_PROJECT="${WANDB_PROJECT:-torchfold-abag-full}"

RESUME_FLAGS=()
JAX_DUMP_FLAGS=()
if [[ -n "${RESUME_CKPT:-}" ]]; then
  RESUME_FLAGS+=(--resume_ckpt "${RESUME_CKPT}")
  if [[ "${LOAD_PARAMS_ONLY:-false}" == "true" ]]; then
    RESUME_FLAGS+=(--load_params_only)
  fi
elif [[ -n "${RESUME_DIR:-}" ]]; then
  # Pick the largest numeric <step>.pt under RESUME_DIR (skip *ema*).
  CKPT="$(ls -1 "${RESUME_DIR}"/*.pt 2>/dev/null | grep -v ema | sort -V | tail -n 1 || true)"
  if [[ -z "${CKPT}" ]]; then
    echo "ERROR: no checkpoint found under RESUME_DIR=${RESUME_DIR}" >&2
    exit 1
  fi
  RESUME_FLAGS+=(--resume_ckpt "${CKPT}")
else
  JAX_DUMP_FLAGS+=(--af3_params_dir "${AF3_PARAMS_DIR}")
fi

# Optional kernel / allocator knobs (override from env if needed).
export LAYERNORM_TYPE="${LAYERNORM_TYPE:-fast_layernorm}"
export TRIANGLE_MULTIPLICATIVE="${TRIANGLE_MULTIPLICATIVE:-torch}"
export TRIANGLE_ATTENTION="${TRIANGLE_ATTENTION:-torch}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "----------------------------------------------"
echo "TorchFold GPU train"
echo "REPO_ROOT=${REPO_ROOT}"
echo "NNODES=${NNODES} NPROC_PER_NODE=${NPROC_PER_NODE} NODE_RANK=${NODE_RANK}"
echo "MASTER_ADDR=${MASTER_ADDR} MASTER_PORT=${MASTER_PORT}"
if [[ ${#RESUME_FLAGS[@]} -gt 0 ]]; then
  echo "RESUME_CKPT=${RESUME_CKPT:-${CKPT:-}}"
else
  echo "AF3_PARAMS_DIR=${AF3_PARAMS_DIR}"
fi
echo "TORCHFOLD_ROOT_DIR=${TORCHFOLD_ROOT_DIR}"
echo "OUT_DIR=${OUT_DIR}"
echo "----------------------------------------------"

# shellcheck disable=SC2086
torchrun \
  --nnodes="${NNODES}" \
  --nproc_per_node="${NPROC_PER_NODE}" \
  --node_rank="${NODE_RANK}" \
  --master_addr="${MASTER_ADDR}" \
  --master_port="${MASTER_PORT}" \
  -m torchfold.runner.train \
  "${JAX_DUMP_FLAGS[@]}" \
  --model_name torchfold_af3_default \
  --train_crop_size 768 --grad_checkpoint \
  --num_recycles 9 --num_diffusion_samples_training 48 --mini_rollout_steps 20 \
  --train_mode structure_only \
  --pair_dropout 0.25 --msa_dropout 0.15 \
  --accumulation_steps 1 --amp_dtype bf16 --num_dl_workers "${NUM_DL_WORKERS}" \
  --loss_alpha_diffusion 1.0 --loss_alpha_bond 1.0 --loss_smooth_lddt 1.0 \
  --loss_alpha_pae 1.0 --loss_alpha_distogram 3e-1 --loss_alpha_confidence 1e-2 \
  --loss_interface_mse_weight 10.0 --loss_interface_mse_distance_threshold 10.0 \
  --loss_interface_mse_mode ca \
  --lr 1e-4 --warmup_steps 500 --ema_decay 0.999 \
  --num_steps 50000 --epoch_size 10000 \
  --eval_interval 200 --ckpt_interval 200 --eval_batches 0 \
  --eval_diffusion_steps 20 --test_max_n_token 1024 \
  --sample_diffusion_chunk_size "${SAMPLE_DIFFUSION_CHUNK_SIZE}" \
  --data.train_sets "${TRAIN_SETS}" \
  --data.train_sampler.train_sample_weights "${TRAIN_SAMPLE_WEIGHTS}" \
  --data.test_sets "${TEST_SETS}" \
  "${RESUME_FLAGS[@]}" \
  --run_dir "${OUT_DIR}" --wandb_dir "${OUT_DIR}" \
  --wandb_project "${WANDB_PROJECT}" --save_final_ckpt \
  ${EXTRA_ARGS:-}
