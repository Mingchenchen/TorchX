# Training

torchfold trains/fine-tunes with
`python -m torchfold.runner.train`, launched under **`torchrun` (DDP)** on one or
more nodes. Weights can be a **JAX parameter dump** or a **PyTorch checkpoint**.

The canonical recipe is **`scripts/ft_abag_full_data.sh`** (Ab–Ag fine-tune on
`sabdab + weightedPDB + distill`). Read it alongside this page — it is the
source of truth for the default hyper-parameters.

---

## 1. Launch

```bash
# from a JAX parameter dump (1 node x 8 GPUs by default)
NNODES=1 NPROC_PER_NODE=8 \
TORCHFOLD_ROOT_DIR=./train_data \
bash scripts/ft_abag_full_data.sh

# full resume (model + optimizer + EMA + step)
RESUME_CKPT=/abs/.../checkpoints/<step>.pt \
  bash scripts/ft_abag_full_data.sh

# auto-find the largest <step>.pt under a run dir and resume it
RESUME_DIR=/abs/.../abag_full/<RUN_DATE> \
  bash scripts/ft_abag_full_data.sh

# params-only warm start (fresh optimizer + LR schedule)
LOAD_PARAMS_ONLY=true RESUME_CKPT=/abs/.../<step>.pt \
  bash scripts/ft_abag_full_data.sh

# land outputs back in an existing run dir basename
RUN_DATE=<existing-timestamp> \
  bash scripts/ft_abag_full_data.sh
```

You can also call the module directly:

```bash
torchrun --nnodes=1 --nproc_per_node=8 --node_rank=0 \
  --master_addr=127.0.0.1 --master_port=29501 \
  -m torchfold.runner.train \
  --resume_ckpt /path/to/checkpoint.pt --load_params_only \
  --num_steps 50000 ...
```

**Env overrides** (read by the script, all optional):

| Env | Meaning | Default |
|-----|---------|---------|
| `NNODES` / `NPROC_PER_NODE` / `NODE_RANK` | `torchrun` topology | `1` / `8` / `0` |
| `MASTER_ADDR` / `MASTER_PORT` | rendezvous | `127.0.0.1` / `29501` |
| `AF3_PARAMS_DIR` | JAX parameter dump directory | `./Alphafold3params` |
| `TORCHFOLD_ROOT_DIR` | training data root | `./train_data` |
| `RESUME_CKPT` | optional PyTorch `.pt`; if unset, start from the JAX dump | unset |
| `RESUME_DIR` | resume the largest `<step>.pt` under this dir (excludes `*_ema*`) | `<none>` |
| `LOAD_PARAMS_ONLY` | with a resume ckpt, load **params only** (fresh optim/schedule) | `false` |
| `RUN_DATE` | output dir basename (`OUT_DIR=.../abag_full/<RUN_DATE>`) | timestamp |
| `OUT_DIR` | full output directory | `./outputs/abag_full/<RUN_DATE>` |
| `TRAIN_SETS` | comma-sep dataset names (must match weights length) | `sabdab,weightedPDB_before210930_v20260101,distill_20260609` |
| `TRAIN_SAMPLE_WEIGHTS` | per-dataset sampling proportion | `0.34,0.33,0.33` |
| `TEST_SETS` | eval sets | `abag_2025` |
| `NUM_DL_WORKERS` | dataloader workers per rank | `2` |
| `SAMPLE_DIFFUSION_CHUNK_SIZE` | diffusion samples per step at eval (lower if OOM) | `5` |
| `WANDB_PROJECT` | wandb project name (example) | `torchfold-abag-full` |
| `EXTRA_ARGS` | appended **after** all flags; argparse last-wins | `<none>` |

```bash
EXTRA_ARGS="--lr 1e-4 --num_steps 20000 --loss_interface_mse_weight 6.0" \
  bash scripts/ft_abag_full_data.sh
```

---

## 2. What the script does

- Resolves the repo root from the script location; puts it on `PYTHONPATH`.
- Launches `torchrun` with `NNODES` / `NPROC_PER_NODE` / `NODE_RANK` /
  `MASTER_ADDR` / `MASTER_PORT` (one process per GPU).
- Defaults triangle kernels to the portable `torch` path; override with env if
  you have optional fused kernels installed (see [configuration.md](configuration.md)).
- Writes checkpoints and wandb artifacts under `--run_dir = OUT_DIR`.

Abridged command from the script:

```bash
torchrun --nnodes=$NNODES --nproc_per_node=$NPROC_PER_NODE \
    --node_rank=$NODE_RANK --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
  -m torchfold.runner.train \
    --resume_ckpt $RESUME_CKPT \
    --model_name torchfold_af3_default \
    --train_crop_size 768 --grad_checkpoint \
    --num_recycles 9 --num_diffusion_samples_training 48 --mini_rollout_steps 20 \
    --train_mode structure_only \
    --pair_dropout 0.25 --msa_dropout 0.15 \
    --accumulation_steps 1 --amp_dtype bf16 --num_dl_workers $NUM_DL_WORKERS \
    --loss_alpha_diffusion 1.0 --loss_alpha_bond 1.0 --loss_smooth_lddt 1.0 \
    --loss_alpha_pae 1.0 --loss_alpha_distogram 3e-1 --loss_alpha_confidence 1e-2 \
    --loss_interface_mse_weight 10.0 --loss_interface_mse_distance_threshold 10.0 \
    --loss_interface_mse_mode ca \
    --lr 1e-4 --warmup_steps 500 --ema_decay 0.999 \
    --num_steps 50000 --epoch_size 10000 \
    --eval_interval 200 --ckpt_interval 200 --eval_batches 0 \
    --eval_diffusion_steps 20 --test_max_n_token 1024 \
    --sample_diffusion_chunk_size $SAMPLE_DIFFUSION_CHUNK_SIZE \
    --data.train_sets $TRAIN_SETS \
    --data.train_sampler.train_sample_weights $TRAIN_SAMPLE_WEIGHTS \
    --data.test_sets $TEST_SETS \
    --run_dir $OUT_DIR --wandb_dir $OUT_DIR \
    --wandb_project $WANDB_PROJECT --save_final_ckpt
```

---

## 3. Initialisation & resume (`runner/train.py`)

| Mode | Flags | Loads |
|------|-------|-------|
| JAX parameter dump | `--af3_params_dir <dir>` | model params only |
| Full resume (PyTorch checkpoint) | `--resume_ckpt <step>.pt` | model + optimizer + EMA + step |
| Params-only warm start (PyTorch checkpoint) | `--resume_ckpt <step>.pt --load_params_only` | model params only; fresh optim + LR schedule |

`--model_name` selects the architecture from the registry in
`configs/configs_model_type.py` (default `torchfold_af3_default`). See
[configuration.md](configuration.md).

---

## 4. Flag reference (`python -m torchfold.runner.train`)

**Model / forward**

| Flag | Meaning |
|------|---------|
| `--model_name` | config registry key (default `torchfold_af3_default`) |
| `--train_crop_size` | token crop size for training (e.g. 768) |
| `--num_recycles` | trunk recycles during training |
| `--num_diffusion_samples_training` | diffusion samples per step (e.g. 48) |
| `--mini_rollout_steps` | mini-rollout diffusion steps feeding the confidence head |
| `--train_mode` | e.g. `structure_only` |
| `--grad_checkpoint` | activation checkpointing (memory) |
| `--pair_dropout`, `--msa_dropout` | dropout rates |
| `--amp_dtype` | autocast dtype (`bf16`) |

**Loss weights** (AF3 stage-3 style)

| Flag | Meaning |
|------|---------|
| `--loss_alpha_diffusion` | diffusion (MSE) weight |
| `--loss_alpha_bond` | bond-length loss weight |
| `--loss_smooth_lddt` | smooth-LDDT weight |
| `--loss_alpha_pae` | PAE head weight |
| `--loss_alpha_distogram` | distogram weight |
| `--loss_alpha_confidence` | confidence head weight |
| `--loss_interface_mse_weight` / `--loss_interface_mse_distance_threshold` / `--loss_interface_mse_mode` | interface-weighted MSE (e.g. weight 10.0, 10 A, `ca`) |
| `--loss_smooth_lddt_interface_weight` / `--loss_smooth_lddt_interface_distance_threshold` | interface-weighted smooth-LDDT |
| `--diffusion_sparse_loss` / `--diffusion_lddt_loss_dense` (`--no-...`) | sparse vs dense LDDT loss path |

**Optimisation / schedule**

| Flag | Meaning |
|------|---------|
| `--lr`, `--warmup_steps` | learning rate + warmup |
| `--ema_decay` | EMA decay (e.g. 0.999); EMA ckpts saved as `<step>_ema.pt` |
| `--num_steps` | total training steps |
| `--epoch_size` | steps per "epoch" (logging/scheduling unit) |
| `--accumulation_steps` | gradient accumulation |
| `--grad_clip` | gradient clipping norm |
| `--decay_every_n_steps`, `--decay_factor` | step-wise LR decay |
| `--seed` | RNG seed |

**Eval / checkpoint / logging**

| Flag | Meaning |
|------|---------|
| `--eval_interval`, `--ckpt_interval` | eval / checkpoint every N steps |
| `--eval_batches` | batches per eval set (0 = whole set) |
| `--eval_diffusion_steps`, `--eval_num_samples` | sampling budget at eval |
| `--eval_ema_only` | evaluate only the EMA weights |
| `--test_max_n_token` | skip eval targets larger than this |
| `--log_interval` | logging cadence |
| `--num_dl_workers` | dataloader workers per rank |
| `--sample_diffusion_chunk_size` | diffusion samples batched per step |
| `--run_dir`, `--wandb_dir`, `--wandb_project`, `--save_final_ckpt` | outputs (checkpoints + wandb) |

**Data** (nested overrides; see §5)

| Flag | Meaning |
|------|---------|
| `--data.train_sets` | comma-sep training dataset names |
| `--data.test_sets` | comma-sep eval dataset names |
| `--data.train_sampler.train_sample_weights` | per-dataset sampling proportion |
| `--data.<set>.base_info.max_n_token` / `.max_release_date` | per-dataset filters |
| `--data.<set>.base_info.exclusion.mol_1_type` / `.mol_2_type` / `.mol_type_group` | exclude modalities |
| `--data.<set>.cropping_configs.method_weights` | `[contiguous,spatial,interface]` crop mix |

---

## 5. Data & config overrides

Dataset names (`sabdab`, `weightedPDB_before210930_v20260101`, `distill_*`,
eval sets `recentPDB_1536_sample384_0925`, `posebusters_0925`, `abag_2025`, ...)
are defined in `configs/configs_data.py`. Anything in the nested config tree is
overridable from the CLI with **dotted `--key value`** flags (parsed by
`config/config.py::parse_configs`), e.g.:

```bash
--data.sabdab.cropping_configs.method_weights 0.1,0.1,0.8     # contig,spatial,iface
--data.sabdab.base_info.max_release_date 2025-01-01
--data.sabdab.base_info.exclusion.mol_1_type ions,ligand,nuc
```

`train_sample_weights` length **must** equal `train_sets` length (asserted in
`data/pipeline/dataset.py`).

Set the data root before launch:

```bash
export TORCHFOLD_ROOT_DIR=./train_data
```

---

## 6. Outputs

Under `--run_dir` (= `OUT_DIR`):

```
OUT_DIR/
  checkpoints/<step>.pt        # full checkpoint (model + optim + EMA + step)
  checkpoints/<step>_ema.pt    # EMA-only weights
  wandb/                       # wandb run dir (often WANDB_MODE=offline)
```

Resume a run by pointing `RESUME_CKPT` (or `RESUME_DIR`) at one of these.

---

## 7. Paths & environment

Recommended:

```bash
cd <this-repo>          # directory that contains pyproject.toml
pip install -e .
```

That installs the tree as the `torchfold` package. The launch scripts also set
`PYTHONPATH` to the repo root as a fallback.

Defaults used throughout:

```
TORCHFOLD_ROOT_DIR=./train_data
```

Runtime knobs (`LAYERNORM_TYPE`, `TRIANGLE_*`, allocator, NCCL) are documented in
[configuration.md](configuration.md).
