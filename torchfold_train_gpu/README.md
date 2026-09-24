# TorchFold (GPU)

GPU training and inference code for TorchFold (AlphaFold3-style structure prediction).

> **Compatibility:** This is a **standalone GPU implementation**, not a device
> switch on a shared codebase. Entry points, configs, data pipelines, and
> **checkpoint formats are not compatible with `torchfold_train_npu`**.

## Do not install into the inference env

This package installs as `torchfold` (`pip install -e .`). TorchFold / TorchScore
inference also does `import torchfold` from its own `src/`. If you install
training into the **same env** used for `torchfold_gpu` (or torchx inference),
`import torchfold` resolves to this training tree and inference breaks.

Use a **separate conda env** for training. Do **not** run
`pip install -e .` inside the torchx / fold / score / craft environment.

The other option (not done in this tree) is to rename the training package to
`torchfold-train` and `import torchfold_train`, so the two can share an env.

## Requirements

- Python >= 3.11
- CUDA-capable GPU(s)
- PyTorch with CUDA matching your driver
- **wandb** (required): `scripts/ft_abag_full_data.sh` / `runner/train.py` call `wandb.init`. Set `WANDB_MODE=offline` if you do not want to sync to the wandb cloud.

## Install

```bash
cd torchfold_train_gpu
pip install -e .
```

Optional fused triangle kernels (otherwise keep the default torch path):

```bash
pip install -e ".[cuequivariance]"
```

## Data & weights

Training loads **either** a PyTorch checkpoint **or** a JAX parameter dump.

- **PyTorch checkpoint (`.pt`)**: Hugging Face release, or a `{step}.pt` from a run you trained. Train with `RESUME_CKPT` (optional `LOAD_PARAMS_ONLY=true`); eval infer with `CKPT`.
- **JAX parameter dump**: a directory of JAX weights. See [docs/training.md](docs/training.md) if you use this path.

Public `.pt` weights and public training data are on Hugging Face (gated; request access on the dataset page):

[https://huggingface.co/datasets/TorchX-CPL/TorchFold](https://huggingface.co/datasets/TorchX-CPL/TorchFold)

```bash
hf download TorchX-CPL/TorchFold --repo-type dataset --local-dir ./TorchFold
```

To build training assets from your own CIF files and TorchFold JSON (with MSA), follow [`torchfold-prepare-training-assets/README.md`](torchfold-prepare-training-assets/README.md).

| Item | Convention |
|------|------------|
| Training data root | `TORCHFOLD_ROOT_DIR=./train_data` (see `configs/configs_data.py`); public data from Hugging Face, or assets from `torchfold-prepare-training-assets` |

Point `TORCHFOLD_ROOT_DIR` at the public training data after download, or at the assets you built.

## Entrypoints

| Task | Module / script |
|------|-----------------|
| Train (DDP) | `torchrun ... -m torchfold.runner.train` or `bash scripts/ft_abag_full_data.sh` |
| Inference (test sets) | `python -m torchfold.runner.inference` or `bash scripts/infer/infer_af3.sh` |
| Inference (JSON) | `python -m torchfold.runner.predict_json` or `bash scripts/infer/predict_json.sh` |

This tree ships `scripts/infer/infer_af3.sh` and `scripts/infer/predict_json.sh`.
They **overlap** with [`../torchfold_gpu`](../torchfold_gpu/README.md) in this
repo. Use these scripts only to evaluate a training run. For released /
production inference, use `torchfold_gpu` instead — do not run both stacks for
the same prediction job.

Scripts under `scripts/` are plain `bash` + `torchrun` helpers. Override paths and hyper-parameters via environment variables . See:

- [docs/training.md](docs/training.md)
- [docs/inference.md](docs/inference.md)
- [docs/configuration.md](docs/configuration.md)

## Quick examples

```bash
# single-node 8-GPU train from a PyTorch checkpoint (params only)
NNODES=1 NPROC_PER_NODE=8 \
LOAD_PARAMS_ONLY=true RESUME_CKPT=/path/to/checkpoint.pt \
TORCHFOLD_ROOT_DIR=./train_data \
WANDB_MODE=offline \
bash scripts/ft_abag_full_data.sh

# same recipe from a JAX parameter dump instead (see docs/training.md)
NNODES=1 NPROC_PER_NODE=8 \
TORCHFOLD_ROOT_DIR=./train_data \
WANDB_MODE=offline \
bash scripts/ft_abag_full_data.sh

# curated test-set inference
CKPT=/path/to/checkpoint.pt \
bash scripts/infer/infer_af3.sh

# JSON inference (single GPU)
INPUT_JSON=/path/to/inputs.json \
CKPT=/path/to/checkpoint.pt \
bash scripts/infer/predict_json.sh
```

## License

Released under the [Apache License 2.0](../LICENSE) at the repository root.
