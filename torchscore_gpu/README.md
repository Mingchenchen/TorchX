#### Prerequisite

> This directory contains only the runtime scripts for **Dataset Preparation** and **Model** (model inference/generation).  
> **Please first follow [`../torchx_gpu/README.md`](../torchx_gpu/README.md) and install from `../torchx_gpu/torchx`**, including:
> - Conda environment creation and installation of torch, etc.
> - Torchx installation, chemical data building
>
> After completing those steps, return here for the actual score run.


#### Scoring Pipeline (TorchScore)

TorchScore evaluates protein complex structures on GPU using a unified pipeline script. The pipeline takes a directory of `.pdb` files as input and produces per-chain and inter-chain confidence metrics.

**Step 1 — Configure the pipeline script**

Edit `src/pipeline_scripts/TorchScore_pipeline.sh` and set the following variables:

| Variable         | Description                                       | Example                            |
|------------------|---------------------------------------------------|------------------------------------|
| `MODEL_DIR`      | Official AlphaFold 3 parameters directory (set this **or** `CHECKPOINT_PATH`, not both) | `/path/to/AlphaFold3/parameters` |
| `PYTHON_EXEC`    | Python executable (must have torchx installed)    | `/<path_to_torchx_env>/bin/python` |
| `TOTAL_CARDS`    | Number of GPU cards available                     | `4`                                |
| `TASKS_PER_CARD` | Parallel inference tasks per card                 | `2`                                |

**Step 2 — Run the pipeline**

```shell
bash src/pipeline_scripts/TorchScore_pipeline.sh \
    <input_pdb_dir> \
    <output_dir> \
    <num_jobs>
```

- `<input_pdb_dir>`: Directory containing input `.pdb` files (one file per complex).
- `<output_dir>`: Root output directory. All intermediate and final results are written here.
- `<num_jobs>`: Number of parallel CPU pre-processing jobs (Step 1–2 of the pipeline).

**Notice**

Set **exactly one** weight source:

- `MODEL_DIR`: official AlphaFold 3 parameters directory (default in the pipeline script)
- `--checkpoint_path`: TorchFold trained checkpoint file (`.pt`)

To use a TorchFold checkpoint, edit `src/pipeline_scripts/TorchScore_pipeline.sh` and replace `--model_dir="$MODEL_DIR"` with `--checkpoint_path="/path/to/torchfold_checkpoint.pt"`.



**What the pipeline does internally**

| Step | Script                     | Description                                                                     |
|------|----------------------------|---------------------------------------------------------------------------------|
| 1    | `01_prepare_get_json.py`   | PDB → CIF conversion, JSON input generation, batch splitting by sequence length |
| 2    | `02_prepare_pdb2jax.sh/py` | PDB → H5 format conversion on CPU (JAX)                                         |
| 3    | `TorchScore_pipeline.sh`   | Runs `run_torchscore.py` on each batch across GPU cards               |
| 4    | `03_get_metrics.py`        | Extracts confidence metrics from inference outputs                              |

**Output layout**

```
<output_dir>/
├── torchscore_outputs/          # Raw inference outputs (per-complex subdirectories)
│   └── <complex_name>/
│       └── seed-10_sample-0/
│           ├── confidences.json
│           └── summary_confidences.json
└── torchscore_metrics.csv       # Final metrics table (one row per complex)
```

**Output metrics**

| Metric                         | Description                                  |
|--------------------------------|----------------------------------------------|
| `ptm`                          | Global predicted TM-score                    |
| `iptm`                         | Global interface pTM (length-weighted)       |
| `chain_X_plddt`                | Per-chain mean pLDDT                         |
| `chain_X_pae`                  | Per-chain mean PAE                           |
| `chain_X_ptm` / `chain_X_iptm` | Per-chain pTM / ipTM                         |
| `ipsae_X_Y`                    | Interface ipSAE between chain X and chain Y  |
| `iptm_X_Y`                     | Inter-chain ipTM between chain X and chain Y |
