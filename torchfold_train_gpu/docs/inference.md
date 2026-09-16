# Inference

torchfold runs inference with `python -m torchfold.runner.inference`
(`runner/inference.py`): it loads a **JAX parameter dump** or a trained
torchfold `{step}.pt` checkpoint, runs diffusion sampling on a **configured
test set**, and dumps **torchfold-format** output — ranked mmCIF files
(per-atom pLDDT in the B-factor column) plus `summary_confidence` /
`full_data` JSONs.

`runner/inference.py` itself is **single-GPU**. The launcher
**`scripts/infer/infer_af3.sh`** fans the work across GPUs on a machine by
starting one process per GPU, each handling a round-robin **shard** of the
targets (`--num_shards` / `--shard_idx`).

---

## 1. Launch (configured test set)

```bash
# from a JAX parameter dump (default), recentPDB eval set, 8 targets:
AF3_PARAMS_DIR=./Alphafold3params \
TORCHFOLD_ROOT_DIR=./train_data \
bash scripts/infer/infer_af3.sh

# a trained checkpoint instead of a JAX dump:
CKPT=/abs/.../checkpoints/42000.pt \
  bash scripts/infer/infer_af3.sh

# Ab-Ag set, more targets/samples, atom-level confidence, custom out dir:
OUT_DIR=./outputs/eval/step42000 CKPT=/abs/.../42000.pt \
  TEST_SET=abag_2025 NUM_TARGETS=100000 NUM_SAMPLES=20 MAX_N_TOKEN=1024 \
  EXTRA_ARGS="--need_atom_confidence" \
  bash scripts/infer/infer_af3.sh
```

**Env tunables** (all optional):

| Env | Meaning | Default |
|-----|---------|---------|
| `CKPT` | torchfold `{step}.pt`; if set, overrides the JAX dump | `<none>` |
| `AF3_PARAMS_DIR` | JAX parameter dump directory | `./Alphafold3params` |
| `TORCHFOLD_ROOT_DIR` | data root for configured test sets | `./train_data` |
| `TEST_SET` | `recentPDB_1536_sample384_0925` \| `posebusters_0925` \| `abag_2025` | `recentPDB_1536_sample384_0925` |
| `NUM_TARGETS` | # targets to predict (across all shards) | `8` |
| `NUM_SAMPLES` | diffusion samples per seed | `5` |
| `NUM_SEEDS` | independent diffusion seeds per target (each -> its own `seed_{seed}/`) | `1` |
| `NUM_RECYCLES` | trunk recycles | `10` |
| `DIFFUSION_STEPS` | reverse-diffusion steps | `200` |
| `MAX_N_TOKEN` | skip targets larger than this (tokens) | `384` |
| `BASE_SEED` | first diffusion seed | `42` |
| `NUM_SHARDS` | # parallel shards | # GPUs on the machine |
| `OUT_DIR` | output dir | `./outputs/infer/<timestamp>` |
| `EXTRA_ARGS` | appended to every shard's argv (argparse last-wins) | `<none>` |

The launcher loops `for i in 0..NUM_SHARDS-1`, pinning shard `i` to GPU `i % NGPU`
with `CUDA_VISIBLE_DEVICES`, each writing `infer_shard<i>.log`; it `wait`s on all
shards and fails if any shard fails.

---

## 2. CLI reference (`python -m torchfold.runner.inference`)

| Flag | Meaning | Default |
|------|---------|---------|
| `--af3_params_dir` | JAX parameter dump directory (used unless `--ckpt`) | `./Alphafold3params` |
| `--ckpt` | torchfold `{step}.pt`; overrides the JAX dump | `None` |
| `--test_set` | configured test set name (from `configs/configs_data.py`) | `recentPDB_1536_sample384_0925` |
| `--dataset_name` | output dataset dir name | `None` -> use `--test_set` |
| `--num_targets` | # targets to predict (pre-shard cap) | `2` |
| `--max_n_token` | skip targets larger than this | `384` |
| `--num_samples` | diffusion samples per seed | `5` |
| `--num_seeds` | independent seeds per target (one forward each) | `1` |
| `--base_seed` | first diffusion seed | `42` |
| `--num_recycles` | trunk recycles | `10` |
| `--diffusion_steps` | reverse-diffusion steps | `200` |
| `--need_atom_confidence` | also dump per-atom `full_data` JSON | off |
| `--num_shards` / `--shard_idx` | split targets across parallel jobs | `1` / `0` |
| `--out_dir` | output directory | `./infer_out` |

> A single seed produces `--num_samples` structures; `--num_seeds` independent
> seeds each run **one** forward and write their own `seed_{seed}/` directory.

---

## 3. Output layout

```
{out_dir}/{dataset_name}/{pdb_id}/seed_{seed}/predictions/
    {pdb_id}_sample_{rank}.cif                       # rank0 = best ranking_score; B-factor = pLDDT
    {pdb_id}_summary_confidence_sample_{rank}.json   # always
    {pdb_id}_full_data_sample_{rank}.json            # only with --need_atom_confidence
```

- **`*_sample_{rank}.cif`** — predicted structure, samples ranked **best-first**
  by `ranking_score`; the per-atom pLDDT is written into the **B-factor** column.
- **`summary_confidence`** — per-sample scalar/aggregate scores:
  `ranking_score`, `ptm`, `iptm`, `plddt`, `gpde`, and per-chain
  `chain_ptm` / `chain_iptm` / `chain_pair_iptm` / `chain_plddt` /
  `chain_pair_pae` / `chain_gpde`
  (see `runner/confidence_summary.py` and `runner/confidence_perchain.py`).
- **`full_data`** (opt-in) — per-atom / per-token-pair arrays:
  `atom_plddt [N_atom]`, `token_pair_pae [N,N]`, `token_pair_pde [N,N]`,
  `contact_probs [N,N]`.

> Note: `atom_plddt` from `full_data` is also what fills the CIF B-factor, so the
> inference driver always computes `full_data` internally even when
> `--need_atom_confidence` is off (only the JSON write is gated by that flag).

The launcher prints a per-seed one-line summary, e.g.
`seed 42: 5 samples predict_s=... rank0 ranking_score=... iptm=... ptm=... plddt=...`.

---

## 4. Environment

See [../README.md](../README.md) and [configuration.md](configuration.md).

Recommended: `pip install -e .` from the repo root. Launch scripts also put the
repo root on `PYTHONPATH`. Defaults:

```
AF3_PARAMS_DIR=./Alphafold3params
TORCHFOLD_ROOT_DIR=./train_data
```

Kernel env defaults in the scripts use the portable `torch` path; optional fused
backends are documented in [configuration.md](configuration.md).

---

## 5. JSON input (single sequence, optional MSA/template)

Beyond a configured **test set**, torchfold can fold raw sequences from a JSON
file via `python -m torchfold.runner.predict_json`, launched by
**`scripts/infer/predict_json.sh`** (1 GPU). No test-set registration needed.

### Input format

A list of entries, each with a `name` and a `sequences` list. Example:

```json
[
  {
    "name": "example_protein",
    "sequences": [
      { "proteinChain": { "sequence": "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG", "count": 1 } }
    ]
  }
]
```

Each top-level list element is folded independently and gets its own output
directory keyed by `name`.

### Optional per-chain fields (chain id / MSA / template)

Every field below is **optional**. They follow the AlphaFold3 JSON style.

| Field | Where | Meaning | Honored when |
|-------|-------|---------|--------------|
| `id` | any entity | explicit chain IDs: a `list[str]` of length `count`, unique across the whole input. Omit to auto-assign. | always |
| `unpairedMsa` / `pairedMsa` | proteinChain | inline a3m string | `--use_msa` |
| `unpairedMsaPath` / `pairedMsaPath` | proteinChain | path to an a3m file (read at load) | `--use_msa` |
| `templatesPath` | proteinChain | path to a `.hhr` or `.a3m` template-hits file (protein only) | `--use_template` |

Example with optional paths:

```json
[{ "name": "example_with_msa_template",
   "sequences": [
     { "proteinChain": {
         "sequence": "MQIFVK...LRLRGG",
         "count": 1,
         "id": ["A"],
         "unpairedMsaPath": "/path/to/chainA.unpaired.a3m",
         "pairedMsaPath":   "/path/to/chainA.paired.a3m",
         "templatesPath":   "/path/to/chainA.hhr"
     } }
   ]
}]
```

> **Inference does NOT search for MSAs/templates.** `--use_msa` only consumes
> the a3m you supply in the JSON; with `--use_msa` set but no a3m provided, the MSA
> is empty and the run is effectively single-sequence. Likewise `--use_template`
> needs a `templatesPath`. To fold a bare sequence, omit these fields and leave both
> toggles `false` (the default).

### Launch

```bash
# required: INPUT_JSON
INPUT_JSON=/path/to/inputs.json \
AF3_PARAMS_DIR=./Alphafold3params \
bash scripts/infer/predict_json.sh

# with template, no msa:
INPUT_JSON=/path/to/inputs.json USE_TEMPLATE=true \
  TEMPLATE_MMCIF_DIR=/path/to/mmcif \
  bash scripts/infer/predict_json.sh

# with msa + template and a trained checkpoint:
INPUT_JSON=/path/to/my.json \
USE_MSA=true USE_TEMPLATE=true CKPT=/path/to/42000.pt \
  bash scripts/infer/predict_json.sh
```

`USE_MSA` / `USE_TEMPLATE` map to `--use_msa` / `--no-use_msa` and
`--use_template` / `--no-use_template`. Both default to `false` (single-sequence
mode; no external databases required).

**Env tunables** (all optional except `INPUT_JSON`):

| Env | Meaning | Default |
|-----|---------|---------|
| `INPUT_JSON` | input JSON (list of `{name,sequences}`) | **required** |
| `USE_MSA` | `true`\|`false` -> `--use_msa` / `--no-use_msa` | `false` |
| `USE_TEMPLATE` | `true`\|`false` -> `--use_template` / `--no-use_template` | `false` |
| `TEMPLATE_MMCIF_DIR` | mmCIF library for templates (also accepts `MMCIF_DIR`) | empty |
| `CKPT` | torchfold `{step}.pt`; if set, overrides the JAX dump | `<none>` |
| `AF3_PARAMS_DIR` | JAX parameter dump directory | `./Alphafold3params` |
| `NUM_SAMPLES` | diffusion samples per seed | `5` |
| `NUM_SEEDS` | independent seeds per entry | `1` |
| `BASE_SEED` | first diffusion seed | `42` |
| `NUM_RECYCLES` | trunk recycles | `10` |
| `DIFFUSION_STEPS` | reverse-diffusion steps | `200` |
| `DATASET_NAME` | output dataset dir name | `json` |
| `OUT_DIR` | output dir | `./outputs/infer_json/<timestamp>` |
| `EXTRA_ARGS` | appended verbatim to argv | `<none>` |

### Output layout

Same format as the test-set path (§3), keyed by each entry's `name`:

```
{out_dir}/{dataset_name}/{name}/seed_{seed}/predictions/
    {name}_sample_{rank}.cif
    {name}_summary_confidence_sample_{rank}.json
    {name}_full_data_sample_{rank}.json            # only with --need_atom_confidence
```
