# Shared helpers. Sourced by run_*.sh / submit_*.sh.

abspath() {
  readlink -f "$1" 2>/dev/null || python3 -c "import os,sys; print(os.path.abspath(sys.argv[1]))" "$1"
}

ncpus() {
  if [[ -n "${SLURM_CPUS_PER_TASK:-}" && "${SLURM_CPUS_PER_TASK}" -gt 0 ]]; then
    echo "${SLURM_CPUS_PER_TASK}"
    return
  fi
  nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4
}

find_cdhit() {
  if [[ -n "${CDHIT_BIN:-}" && -x "${CDHIT_BIN}" ]]; then
    echo "${CDHIT_BIN}"
    return 0
  fi
  if command -v cd-hit >/dev/null 2>&1; then
    command -v cd-hit
    return 0
  fi
  if command -v cdhit >/dev/null 2>&1; then
    command -v cdhit
    return 0
  fi
  return 1
}

# 0.40 -> 40 ; 0.5 -> 50
identity_to_pct() {
  python3 -c "print(int(round(float('${1}') * 100)))"
}

# CD-HIT recommended word size for a given identity threshold.
cdhit_word_size() {
  python3 - <<PY
c = float("${1}")
if c >= 0.7:
    print(5)
elif c >= 0.6:
    print(4)
elif c >= 0.5:
    print(3)
else:
    print(2)
PY
}


print_msa_result_paths() {
  local msa_outdir
  msa_outdir="$(abspath "${MSA_OUTDIR}")"
  cat <<EOF

========== RESULT PATHS (MSA) ==========
MSA_OUTDIR=${msa_outdir}
SEQ_TO_PDB_INDEX=${msa_outdir}/common/seq_to_pdb_index.json
MMCIF_MSA_TEMPLATE=${msa_outdir}/mmcif_msa_template
========================================
EOF
}

print_cluster_result_paths() {
  local cluster_txt cluster_out
  cluster_out="$(abspath "${CLUSTER_OUT_DIR}")"
  cluster_txt="$(abspath "${CLUSTER_TXT}")"
  cat <<EOF

========== RESULT PATHS (CLUSTER) ==========
CLUSTER_OUT_DIR=${cluster_out}
CLUSTERS_BY_ENTITY_40=${cluster_txt}
===========================================
EOF
}

print_pipeline_result_paths() {
  local msa_outdir step1_bio step1_indices
  msa_outdir="$(abspath "${MSA_OUTDIR}")"
  step1_bio="$(abspath "${STEP1_BIO_DIR}")"
  step1_indices="$(abspath "${STEP1_INDICES_CSV}")"
  cat <<EOF

========== RESULT PATHS ==========
SEQ_TO_PDB_INDEX=${msa_outdir}/common/seq_to_pdb_index.json
MMCIF_MSA_TEMPLATE=${msa_outdir}/mmcif_msa_template
BIOASSEMBLY_DIR=${step1_bio}
INDICES_CSV=${step1_indices}
==================================
EOF
}
