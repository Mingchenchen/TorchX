#### Prerequisite

> This directory contains only the runtime scripts for **Dataset Preparation** and **Model** (model inference/generation).  
> **Please first follow [`../torchx_gpu/README.md`](../torchx_gpu/README.md) and install from `../torchx_gpu/torchx`**, including:
> - Conda environment creation and installation of torch, etc.
> - Torchx installation, chemical data building
>
> After completing those steps, return here for the actual design run.

If you want to use the cueq acceleration operators, you need to additionally install `cuequivariance-torch` and `cuequivariance-ops-torch-cu12` (CUDA 12; use `cuequivariance-ops-torch-cu13` for CUDA 13). Then set `TRIANGLE_MULTIPLICATIVE` and `TRIANGLE_ATTENTION` to `"cuequivariance"` in `task/config/base.yaml`. The default is `"torch"`.

```shell
pip install cuequivariance-torch cuequivariance-ops-torch-cu12
```


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