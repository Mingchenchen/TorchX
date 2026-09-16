# CIF + TorchFold JSON (with MSA) → TorchFold training assets

TorchFold training data follow the [Protenix](https://www.biorxiv.org/content/10.1101/2025.01.08.631967) training-data format.

## Setup

```bash
conda install -c bioconda cd-hit
pip install protenix

export PROTENIX_ROOT_DIR=/path/to/protenix_data
mkdir -p $PROTENIX_ROOT_DIR/common
cd $PROTENIX_ROOT_DIR/common
wget https://protenix.tos-cn-beijing.volces.com/common/components.cif
wget https://protenix.tos-cn-beijing.volces.com/common/components.cif.rdkit_mol.pkl
```

If `cd-hit` is not on `PATH`: `export CDHIT_BIN=/path/to/cd-hit`

## Run

```bash
# example (--identity is required, e.g. 0.4 / 0.5)
bash scripts/run_all.sh --example --identity 0.4

# your data
# cif_list.txt: one CIF path per line
# json_w_msa/: {cif_stem}.json TorchFold JSON with inline pairedMsa / unpairedMsa
bash scripts/run_all.sh \
  --cif-list /path/to/cif_list.txt \
  --json-dir /path/to/json_w_msa \
  --output-dir /path/to/run_output \
  --identity 0.4
```

## Slurm (optional)

On a Slurm cluster, submit the same pipeline as sequential jobs (MSA → cluster → Step1):

```bash
bash scripts/submit_all.sh \
  --cif-list /path/to/cif_list.txt \
  --json-dir /path/to/json_w_msa \
  --output-dir /path/to/run_output \
  --log-dir /path/to/logs \
  --identity 0.4 \
  --partition PARTITION
```

## Output

```
run_output/
  msa_torchfold/common/seq_to_pdb_index.json
  msa_torchfold/mmcif_msa_template/{idx}/
  cluster_information/clusters-by-entity-<pct>.txt   # e.g. 40 for --identity 0.4
  output/<run_id>/step1_.../bioassembly/*.pkl.gz
  output/<run_id>/step1_.../indices.csv
```
