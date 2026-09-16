# mini_binder input

The program **reads the target sequence from the input PDB automatically**. In most cases you only need the target PDB; a sequence file is not required.

Provide `input_sequence.csv` only when the PDB has chain breaks / missing residues, or when the sequence used for design should not fully match the PDB structure. In those cases the CSV sequence overrides the sequence parsed from the PDB.

## Input layout

```text
mini_binder/
├── mini_binder_design.sh
├── input/
│   ├── pdl1_8znl.pdb          # target structure (required)
│   ├── input_sequence.csv     # full target sequence (optional; see below)
│   └── readme.md
└── output/                    # created at runtime
```

Paths you may need to edit in the script:

- `REPO_ROOT_DIR`: root of the `torchcraft_npu` repo
- `PYTHON_BIN`: python from the conda env
- `PDB_DIR`: target PDB, default `input/pdl1_8znl.pdb`
- `PDB_FULL_SEQ`: optional sequence CSV, default `input/input_sequence.csv`

To switch targets, place the new PDB under `input/` and update `PDB_DIR` and `--hotspot_indices` in the script.

## `input_sequence.csv` (optional)

If you do not need to override the PDB sequence, omit this file or do not pass `--target_full_seq`. When you do need it, use:

```csv
target,real_sequence
"pdl1_8znl",FTVTVPKDLYVVEYGSNMTIECKFPVEKQLDLAALIVYWEMEDKNIIQFVHGEEDLKVQHSSYRQRARLLKDQLSLGNAALQITDVKLQDAGVYRCMISYGGADYKRITVKVNK
```

| Column | Description |
| --- | --- |
| `target` | Must match the PDB filename without extension, e.g. `pdl1_8znl.pdb` → `pdl1_8znl` |
| `real_sequence` | Full amino-acid sequence used for design |

Lookup uses the PDB filename. If no matching `target` is found, the sequence parsed from the PDB is used.

## Output layout

Running `mini_binder_design.sh` writes:

```text
output/batch_runs_YYYYMMDD_HHMMSS/
└── seed_{SEED}/
    ├── pdl1_8znl.pdb                              # target PDB copied into the task dir
    ├── npu{slot}_g{job}_k{k}.out                  # stdout log
    ├── npu{slot}_g{job}_k{k}.err                  # stderr log
    ├── design_results.csv                         # per-epoch sequences and scores
    └── {target}_last_stage_epoch_{n}_model.cif    # last-stage structures
```

`design_results.csv` includes `Epoch`, `Sequence` (designed binder), `pTM`, `ipTM`, `pLDDT`, loss terms, and `Stage`. Structure files are mmCIF models of the target–binder complex.
