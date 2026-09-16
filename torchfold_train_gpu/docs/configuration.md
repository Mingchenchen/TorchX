# Configuration

torchfold uses an AF3-style nested config tree that is **fully overridable
from the command line**. This page covers the model registry, the `--data.*` /
`--model.*` override mechanism, the runtime env knobs, and JAX dump /
PyTorch checkpoint loading.

---

## 1. Config sources

| File | Role |
|------|------|
| `configs/configs_base.py` | the full default config (model dims, diffusion, loss, data defaults) |
| `configs/configs_model_type.py` | **model registry**: `model_name -> spec`, deep-merged onto base |
| `configs/configs_data.py` | dataset definitions (train sets, eval sets, cropping, filters) |
| `config/config.py` | `parse_configs(...)` — turns the nested dict into an argparse-driven `ConfigDict` so any nested key becomes a `--dotted.key` flag |
| `config/extend_types.py` | typed/required config value wrappers |

Resolution order: **`configs_base` -> deep-merge `model_configs[model_name]`
-> apply CLI `--dotted.key` overrides** (last wins).

---

## 2. Model registry (`--model_name`)

`configs/configs_model_type.py` holds `model_configs = { name -> spec }`. The
spec is deep-merged onto `configs_base`. Currently:

| `--model_name` | Notes |
|----------------|-------|
| `torchfold_af3_default` | default; the AF3-faithful config (used by both train & infer) |

Adding a variant is just another registry entry, e.g. (commented template in the
file):

```python
model_configs = {
    "torchfold_af3_default": { ... },
    "my_model_v1": {
        "model": {
            "pairformer": {"c_z": 256},
            "diffusion_module": {"c_z": 256},
        },
    },
}
```
Then train/infer with `--model_name my_model_v1`.

---

## 3. Nested CLI overrides (`--data.*`, `--model.*`)

`config/config.py::parse_configs` registers **one `--<dotted.key>` argument per
leaf** of the nested config. So you can override anything without editing config
files. Examples (from the training recipe):

```bash
--data.train_sets sabdab,weightedPDB_before210930_v20260101,distill_20260609
--data.train_sampler.train_sample_weights 0.34,0.33,0.33
--data.test_sets abag_2025
--data.sabdab.base_info.max_n_token -1
--data.sabdab.base_info.max_release_date 2025-01-01
--data.sabdab.base_info.exclusion.mol_1_type ions,ligand,nuc
--data.sabdab.cropping_configs.method_weights 0.1,0.1,0.8     # [contiguous,spatial,interface]
```

Rules:
- Values are passed as strings and coerced to the leaf's declared type.
- Comma-separated lists map to list-typed leaves (e.g. `method_weights`,
  `train_sample_weights`, `mol_1_type`).
- `train_sample_weights` length **must** match `train_sets` length (asserted in
  `data/pipeline/dataset.py`).

> Many top-level training knobs (`--lr`, `--num_steps`, loss weights, etc.) are
> **plain argparse flags** on `runner/train.py` (see [training.md](training.md)),
> not nested config keys. The `--data.*` / `--model.*` dotted form is for the
> nested config tree.

---

## 4. Datasets (`configs/configs_data.py`)

Names referenced by `--data.train_sets` / `--data.test_sets` / `--test_set`:

| Kind | Examples |
|------|----------|
| Train | `sabdab`, `weightedPDB_before210930_v20260101`, `distill_20260609` |
| Eval  | `recentPDB_1536_sample384_0925`, `posebusters_0925`, `abag_2025` |

Per-dataset config (overridable as `--data.<name>.<...>`):
`base_info.max_n_token`, `base_info.max_release_date`,
`base_info.exclusion.{mol_1_type,mol_2_type,mol_type_group}`,
`cropping_configs.method_weights` (`[contiguous, spatial, interface]` mix).

Data root (paths under each dataset resolve relative to this):

```bash
export TORCHFOLD_ROOT_DIR=./train_data
```

---

## 5. Runtime environment knobs

Set by the launch scripts; override via env before running.

| Env var | Effect | Suggested default |
|---------|--------|-------------------|
| `PYTHONPATH` | usually unnecessary after `pip install -e .`; otherwise include the repo root | repo root |
| `TORCHFOLD_ROOT_DIR` | training / eval data root | `./train_data` |
| `AF3_PARAMS_DIR` | JAX parameter dump directory | `./Alphafold3params` |
| `LAYERNORM_TYPE` | LayerNorm kernel | `fast_layernorm` or `torch` |
| `TRIANGLE_MULTIPLICATIVE` | triangle multiplicative-update kernel | `torch` (portable) |
| `TRIANGLE_ATTENTION` | triangle-attention kernel | `torch` (portable) |
| `PYTORCH_CUDA_ALLOC_CONF` | allocator | `expandable_segments:True` |
| `OMP_NUM_THREADS` | CPU threads | e.g. `8` |
| `WANDB_MODE` / `WANDB_DIR` | wandb (training) | `offline` / `OUT_DIR` |
| NCCL: `NCCL_IB_DISABLE`, `NCCL_SOCKET_IFNAME`, `TORCH_NCCL_*` | multi-node DDP | set only if needed on your fabric |

`TRIANGLE_*=torch` is the default in the shipped scripts so a plain install
runs without optional fused packages. If you install cuequivariance (see
`pip install -e ".[cuequivariance]"`), you may set:

```bash
export TRIANGLE_MULTIPLICATIVE=cuequivariance
export TRIANGLE_ATTENTION=cuequivariance
```

---

## 6. Weight loading

Two formats:

- **JAX parameter dump.** Directory of JAX weights (`AF3_PARAMS_DIR`).
  `import_jax_weights_` maps them into the torch module.
- **PyTorch checkpoint.** `RESUME_CKPT` / `--resume_ckpt` (train; optional
  `--load_params_only`) or `CKPT` / `--ckpt` (eval infer).

| Init path | How |
|-----------|-----|
| JAX parameter dump | `AF3_PARAMS_DIR` / `--af3_params_dir` |
| PyTorch checkpoint | train: `--resume_ckpt` (optional `--load_params_only`); eval infer: `--ckpt` |

---

## 7. See also

- [training.md](training.md) — training flag reference + the canonical recipe
- [inference.md](inference.md) — inference CLI + output layout
- [../README.md](../README.md) — install, data layout, quickstart
