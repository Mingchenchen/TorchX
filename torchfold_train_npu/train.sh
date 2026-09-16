#!/usr/bin/env bash
# TorchFold NPU training launcher (no hard-coded host paths).
# Edit config.yaml paths, then:
#   bash train.sh
# Optional overrides:
#   NPUS_PER_NODE=8 NNODES=1 MASTER_ADDR=127.0.0.1 bash train.sh
set -euo pipefail

BASE_DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${BASE_DIR}:${BASE_DIR}/loss_function${PYTHONPATH:+:${PYTHONPATH}}"

export CPU_AFFINITY_CONF="${CPU_AFFINITY_CONF:-1}"
export TASK_QUEUE_ENABLE="${TASK_QUEUE_ENABLE:-2}"
export COMBINED_ENABLE="${COMBINED_ENABLE:-1}"
export PYTORCH_NPU_ALLOC_CONF="${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export PYTHONUNBUFFERED=1

# Optional TCMalloc (set LD_PRELOAD yourself if installed; see README).
# export LD_PRELOAD=/path/to/libtcmalloc.so.4

export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-29501}"
export NODE_RANK="${NODE_RANK:-${SLURM_NODEID:-0}}"

# Larger p_mean → higher noise; smaller p_mean → lower noise
# Data source 0: low noise; sources 1,2: higher noise
if [[ -z "${NOISE_CONFIGS:-}" ]]; then
  export NOISE_CONFIGS='{"0": {"p_mean": -2.5, "p_std": 1.2}, "1": {"p_mean": 0.5, "p_std": 1.5}, "2": {"p_mean": 0.5, "p_std": 1.5}}'
fi

NPUS_PER_NODE="${NPUS_PER_NODE:-2}"
NNODES="${NNODES:-1}"
WORLD_SIZE=$((NNODES * NPUS_PER_NODE))

echo "----------------------------------------------"
echo "TorchFold NPU training"
echo "BASE_DIR=${BASE_DIR}"
echo "Nodes: ${NNODES}"
echo "NPUs per node: ${NPUS_PER_NODE}"
echo "Total NPUs (world_size): ${WORLD_SIZE}"
echo "NODE_RANK=${NODE_RANK}"
echo "Master hostname: $(hostname)"
echo "MASTER_ADDR=${MASTER_ADDR}"
echo "MASTER_PORT=${MASTER_PORT}"
if [[ -n "${SLURM_JOB_NODELIST:-}" ]]; then
  echo "SLURM_JOB_NODELIST=${SLURM_JOB_NODELIST}"
fi
echo "----------------------------------------------"

torchrun \
  --nnodes="${NNODES}" \
  --nproc_per_node="${NPUS_PER_NODE}" \
  --node_rank="${NODE_RANK}" \
  --master_addr="${MASTER_ADDR}" \
  --master_port="${MASTER_PORT}" \
  "${BASE_DIR}/train_torchfold.py" \
  --config "${BASE_DIR}/config.yaml"
