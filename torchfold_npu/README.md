#### Prerequisite

> This directory contains only the runtime scripts for **Dataset Preparation** and **Model** (model inference/generation).  
> **Please first follow [`../torchx_npu/README.md`](../torchx_npu/README.md) and install from `../torchx_npu/torchx`**, including:
> - Conda environment creation and installation of torch, torch_npu, mx_driving, tcmalloc, etc.
> - Torchx installation, chemical data building, and model weights download
> - (Optional) CANN Fusion_Attention operator acceleration setup
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

The default attention implementation is `torch`. To use CANN Fusion_Attention, install the operator (see [`../torchx_npu/README.md`](../torchx_npu/README.md)), then set `--dot_product_attention=Fusion_Attention` in `src/run.sh`.
