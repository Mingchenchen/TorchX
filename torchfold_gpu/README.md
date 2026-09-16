#### Prerequisite

> This directory contains only the runtime scripts for **Dataset Preparation** and **Model** (model inference/generation).  
> **Please first follow [`../torchx_gpu/README.md`](../torchx_gpu/README.md) and install from `../torchx_gpu/torchx`**, including:
> - Conda environment creation and installation of torch, etc.
> - Torchx installation, chemical data building
>
> After completing those steps, return here for the actual fold run.


**Optional: Download AlphaFold Databases**

Only needed when `RUN_DATA_PIPELINE=true` in `src/scripts/env.sh` (default). Skip this step if you already have the databases elsewhere, or set `RUN_DATA_PIPELINE=false` to run inference on preprocessed inputs only.

```bash
bash fetch_databases.sh /path/to/database/dir
```

## Running Inference

1. **Configure Paths**: Edit `src/scripts/env.sh` according to your environment.

   Set **exactly one** of the following (they are mutually exclusive):

   - `MODEL_DIR`: directory of official AlphaFold 3 parameters
   - `CHECKPOINT_PATH`: TorchFold trained checkpoint file (`.pt`)

   Comment out the one you are not using. `src/run.sh` will error if both or neither is set.

2. **Run Inference**:

```bash
bash run.sh
```


