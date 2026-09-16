# vhh input

The program **reads the target sequence from the input PDB automatically**. In most cases you only need the target PDB and nanobody framework CIFs; a sequence file is not required.

Provide `input_sequence.csv` only when the PDB has chain breaks / missing residues, or when the sequence used for design should not fully match the PDB structure. In those cases the CSV sequence overrides the sequence parsed from the PDB.

## Input layout

```text
vhh/
├── vhh_design.sh
├── input/
│   ├── bhrf1_2wh6.pdb                 # target structure (required)
│   ├── input_sequence.csv             # full target sequence (optional; see below)
│   ├── frameworks/                    # nanobody framework CIFs (required)
│   │   ├── 7eow_nanobody_framework.cif
│   │   ├── 5jds_nanobody_framework.cif
│   │   ├── 7xl0_nanobody_framework.cif
│   │   ├── 8coh_nanobody_framework.cif
│   │   └── 8z8v_nanobody_framework.cif
│   └── readme.md
└── output/                            # created at runtime
```

Paths you may need to edit in `vhh_design.sh`:

- `REPO_ROOT_DIR`: root of the `torchcraft_gpu` repo
- `PYTHON_BIN`: python from the conda env
- `PDB_DIR`: target PDB, default `${REPO_ROOT_DIR}/task/vhh/input/bhrf1_2wh6.pdb`
- `HOTSPOT_INDICES`: hotspot residue indices (PDB numbering), default `63,70,79,86,105`
- `PDB_FULL_SEQ`: optional sequence CSV, default `${REPO_ROOT_DIR}/task/vhh/input/input_sequence.csv`
- `FRAMEWORK_DIR`: framework CIF directory, default `${REPO_ROOT_DIR}/task/vhh/input/frameworks`
- `OUT_DIR`: default `${REPO_ROOT_DIR}/task/vhh/output`

Also configure `task/config/base.yaml` (`model_dir`, `db_dir`, `checkpoint_path`) before running.

To switch targets, place the new PDB under `input/` and update `PDB_DIR` and `HOTSPOT_INDICES` in the script. To change frameworks, update the `FRAMEWORKS` array and `NUM_TASKS` / `TASKS_PER_FRAMEWORK`.

## `input_sequence.csv` (optional)

If you do not need to override the PDB sequence, omit this file or do not pass `--target_full_seq`. When you do need it, use:

```csv
target,real_sequence
"bhrf1_2wh6",AYSTREILLALCIRDSRVHGNGTLHPVLELAARETPLRLSPEDTVVLRYHVLLEEIIERNSETFTETWNRFITHTEHVDLDFNSVFLEIFHRGDPSLGRALAWMAWCMHACRTLCCNQSTPYYVVDLSVRGMLEASEGLDGWIHQQGGWSTLIEDNI
```

| Column | Description |
| --- | --- |
| `target` | Must match the PDB filename without extension, e.g. `bhrf1_2wh6.pdb` → `bhrf1_2wh6` |
| `real_sequence` | Full amino-acid sequence used for design |

Lookup uses the PDB filename. If no matching `target` is found, the sequence parsed from the PDB is used.

## How to run

```bash
cd task/vhh
bash vhh_design.sh
```

The script runs on GPU directly (no Slurm). Tasks rotate through the five framework CIFs; parallel execution is controlled by `NUM_GPUS`, `TASKS_PER_CARD`, and `CUDA_VISIBLE_DEVICES`.

## Output layout

Running `vhh_design.sh` writes:

```text
output/batch_runs_YYYYMMDD_HHMMSS/
└── seed_{SEED}_{FW_ID}/
    ├── bhrf1_2wh6.pdb                             # target PDB copied into the task dir
    ├── gpu{slot}_g{job}_k{k}.out                  # stdout log
    ├── gpu{slot}_g{job}_k{k}.err                  # stderr log
    ├── design_results.csv                         # per-epoch sequences and scores
    └── {target}_last_stage_epoch_{n}_model.cif    # last-stage structures
```

`{FW_ID}` is the framework filename prefix (e.g. `7eow`, `5jds`, `8z8v`). `design_results.csv` includes `Epoch`, `Sequence` (designed VHH), `pTM`, `ipTM`, `pLDDT`, loss terms (including paratope / IgLM), and `Stage`. Structure files are mmCIF models of the target–nanobody complex.
