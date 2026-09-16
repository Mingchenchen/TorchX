#### Prerequisite

> This directory contains only the runtime scripts for **Dataset Preparation** and **Model** (model inference/generation).  
> **Please first follow [`../torchx_npu/README.md`](../torchx_npu/README.md) and install from `../torchx_npu/torchx`**, including:
> - Conda environment creation and installation of torch, torch_npu, mx_driving, tcmalloc, etc.
> - Torchx installation, chemical data building, and model weights download
> - (Optional, recommended for performance) CANN Fusion_Attention operator acceleration setup — see **Accelerate using CANN Fusion_Attention** in [`../torchx_npu/README.md`](../torchx_npu/README.md)
>
> After completing those steps, return here for the actual fold run.

```shell
mkdir -p models
# Place your downloaded model weights in the models/ directory
```

#### Dataset Preparation

Please refer to the project documentation for downloading databases.

Set the database path in `src/scripts/env.sh` before running inference.

#### Inference

Before running, edit the paths in `src/scripts/env.sh`. Set **exactly one** of:

- `MODEL_DIR`: directory of official AlphaFold 3 parameters
- `CHECKPOINT_PATH`: TorchFold trained checkpoint file (`.pt`)

Comment out the one you are not using. `src/run.sh` will error if both or neither is set.

```shell
bash run.sh
```

`--dot_product_attention` defaults to `torch` so a direct run does not fail when Fusion_Attention is not installed. For better performance, use **Fusion_Attention** instead of `torch`. Fusion_Attention must be installed first; follow **Accelerate using CANN Fusion_Attention** in [`../torchx_npu/README.md`](../torchx_npu/README.md). Then set `--dot_product_attention=Fusion_Attention` in `src/run.sh`.

#### Multi-card Inference

Set `DEVICES`, `NPUS_PER_NODE`, and `USE_DIST` before running. The number of
device IDs in `DEVICES` must match `NPUS_PER_NODE`, and multi-card inference
requires `USE_DIST=1`.

Two-card inference:

```shell
DEVICES=0,1 NPUS_PER_NODE=2 USE_DIST=1 bash run.sh
```

Four-card inference:

```shell
DEVICES=0,1,2,3 NPUS_PER_NODE=4 USE_DIST=1 bash run.sh
```
