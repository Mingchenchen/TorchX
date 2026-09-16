#### Prerequisite

> This directory contains only the runtime scripts for **Dataset Preparation** and **Model** (model inference/generation).  
> **Please first follow [`../torchx_npu/README.md`](../torchx_npu/README.md) and install from `../torchx_npu/torchx`**, including:
> - Conda environment creation and installation of torch, torch_npu, mx_driving, tcmalloc, etc.
> - Torchx installation, chemical data building, and model weights download
> - (Optional) CANN Fusion_Attention operator acceleration setup
>
> After completing those steps, return here for the actual design run.

The default attention implementation is `"torch"`. To use CANN Fusion_Attention, install the operator (see [`../torchx_npu/README.md`](../torchx_npu/README.md)), then set `dot_product_attention` to `"Fusion_Attention"` in `task/config/base.yaml`.

#### Dataset Preparation

Design takes a preprocessed JSON as input. Keep `run_data_pipeline: False` in `task/config/base.yaml` so MSA / template search is skipped.

The initial binder sequence is filled automatically at the start of design by `src/custom_function/fill_sequences.py` (imported by `run_torchcraft.py`; you do not run this script yourself). The fill strategy is `design_initial_sequence` in `task/config/base.yaml` (default `gumbel_sequence`).

#### Design

Before running, modify paths in the corresponding design script and in `task/config/base.yaml`.

In `task/config/base.yaml`, set **exactly one** of:

- `model_dir`: directory of official AlphaFold 3 parameters
- `checkpoint_path`: TorchFold trained checkpoint file (`.pt`)

If both are set, `checkpoint_path` takes priority. Comment out the unused one.

```shell
cd task/monomer_unconditional
bash batch_submit.sh
```

VHH and mini-binder design entry points: `task/vhh/vhh_design.sh` and `task/mini_binder/mini_binder_design.sh`.


The inference script contains code for multi-task parallel inference on a single card. The table below shows the NPU memory occupied by different sequence lengths during single-card inference, as well as the maximum achievable task parallelism.
(Test environment: Ascend 910B3 chip, 64GB (65536MB) NPU memory, with approximately 3400MB of memory usage when idle)

**Table 1: Single-card Single-task Memory Usage and Maximum Parallelism by Sequence Length**

| Sequence Length(aa) | Memory Usage(MB) | Memory Utilization(%) | Maximum Number of Parallel Tasks per Card |
|-------------|-------------|--------------|---------------|
| 100 | 6553 | 10 | 19 |
| 200 | 9175 | 14 | 10 |
| 300 | 13762 | 21 | 5 |
| 400 | 20316 | 31 | 3 |
| 500 | 28180 | 43 | 2 |
| 600 | 38666 | 59 | 1 |
| 700 | 49807 | 76 | 1 |
| 800 | 64225 | 98 | 1 |

**Table 2: Maximum Supported Sequence Length for Multi-Task on a Single Card**

| Number of Parallel Tasks per Card | Maximum Supported Sequence Length(aa) |
|---------------|---------------------|
| 1 | 800 |
| 2 | 550 |
| 3 | 435 |
| 4 | 380 |
| 5 | 330 |
| 6 | 280 |
| 10 | 210 |
| 19 | 100 |
