#### Prerequisite

> This directory contains only the runtime scripts for **Dataset Preparation** and **Model** (model inference/generation).  
> **Please first follow [`../torchx_gpu/README.md`](../torchx_gpu/README.md) and install from `../torchx_gpu/torchx`**, including:
> - Conda environment creation and installation of torch, etc.
> - Torchx installation, chemical data building
>
> After completing those steps, return here for the actual design run.

Recommended runtime: **Linux x86_64 + Python 3.11 + PyTorch 2.8.0 (CUDA 12 / cu128)**.  
`src/checkpoint_reentrant_function.cpython-311-x86_64-linux-gnu.so` is required for gradient checkpointing and only matches this Python/ABI. After the torchx env is active:

```shell
cd src
python -c "from checkpoint_reentrant_function import CheckpointReentrantFunction; print('ok')"
```

#### Optional: cueq acceleration

Default backends in `task/config/base.yaml` are `"torch"` and do not need extra packages.

To enable cuEquivariance triangle kernels, install the pinned CUDA 12 packages **after** torchx is installed. Do not run an unpinned `pip install cuequivariance-torch` — that can replace torch with a CUDA 13 build.

```shell
pip install -r requirements-cueq.txt
python -c "import torch; print(torch.__version__)"   # still 2.8.0
```

Then set `TRIANGLE_MULTIPLICATIVE` and `TRIANGLE_ATTENTION` to `"cuequivariance"` in `task/config/base.yaml`.


#### Design

Before running, modify paths in the design script and in `task/config/base.yaml`.

In `task/config/base.yaml`, set **exactly one** of:

- `model_dir`: directory of official AlphaFold 3 parameters
- `checkpoint_path`: TorchFold trained checkpoint file (`.pt`)

If both are set, `checkpoint_path` takes priority. Comment out the unused one.

```shell
cd task/monomer_unconditional
bash batch_submit.sh
```

VHH and mini-binder design entry points: `task/vhh/vhh_design.sh` and `task/mini_binder/mini_binder_design.sh`.
