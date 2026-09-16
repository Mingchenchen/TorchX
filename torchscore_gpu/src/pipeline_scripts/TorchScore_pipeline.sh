#!/bin/bash

# ============================== Configuration ==============================
# Usage: TorchScore_pipeline.sh <input_pdb_dir> <output_dir> <num_jobs>

PYTHON_EXEC="/path/to/torchx_env/bin/python"
# Path to the directory containing run_torchscore.py
TORCHFOLD_ROOT="$(dirname "$(realpath "$0")")/.."
# Model weights: set exactly one.
# MODEL_DIR: official AlphaFold 3 parameters directory
# To use a TorchFold trained checkpoint instead, replace --model_dir="$MODEL_DIR"
# with --checkpoint_path=/path/to/torchfold_checkpoint.pt in the inference call below.
MODEL_DIR=/path/to/AlphaFold3/parameters

# cuda resource configuration
TOTAL_CARDS=1
TASKS_PER_CARD=1
MAX_JOBS=$((TOTAL_CARDS * TASKS_PER_CARD))

pipeline_script_dir=$(dirname "$(realpath "$0")")

# ============================== Helper Functions ==============================
log_step() {
    echo "========================================================================================"
    echo "========== [Step $1] $2"
    echo "========================================================================================"
}

log_info() {
    echo "[INFO] $(date '+%Y-%m-%d %H:%M:%S')  $1"
}

# ============================== Argument Parsing ==============================
usage() {
    cat <<EOF
Usage: $0 <input_pdb_dir> <output_dir> <num_jobs>

Example:
  $0 /data/pdbs /output/torchscore 4
EOF
    exit 1
}

[[ $# -ge 3 ]] || usage

input_pdb_dir="$(realpath "$1")"
output_dir="$(realpath "$2")"
num_jobs="$3"

# ============================== Initialization ==============================
log_info "========== TorchScore Pipeline started =========="
log_info "Input PDB dir   : $input_pdb_dir"
log_info "Output dir      : $output_dir"
log_info "Num jobs        : $num_jobs"
log_info "cuda cards       : $TOTAL_CARDS (max parallel: $MAX_JOBS)"

start_time=$(date +%s)

# --- Prepare directories ---
output_base="$output_dir"
tf_input_batch="$output_base/tf_input_batch"
output_dir_cif="$output_base/single_chain_cif"
save_csv="$output_base/single_seq.csv"
output_dir_json="$output_base/json"
output_dir_jax="$tf_input_batch/jax"
output_dir_torchscore="$output_base/torchscore_outputs"
metric_csv="$output_base/torchscore_metrics.csv"
jax_log_dir="$output_base/logs/jax"
inference_log_dir="$output_base/logs/inference"

mkdir -p "$tf_input_batch" "$output_dir_cif" "$output_dir_jax" "$output_dir_json" \
         "$output_dir_torchscore" "$jax_log_dir" "$inference_log_dir"

# =================================================================
# Step 1: Prepare inputs (PDB -> CIF, JSON, batch split)
# =================================================================
log_step "01" "Preparing inputs (JSON generation and batch splitting)"

$PYTHON_EXEC "$pipeline_script_dir/01_prepare_get_json.py" \
    --input_dir "$input_pdb_dir" \
    --output_dir_cif "$output_dir_cif" \
    --save_csv "$save_csv" \
    --output_dir_json "$output_dir_json" \
    --batch_dir "$tf_input_batch" \
    --num_jobs "$num_jobs"

log_info "Step 1 completed."

# =================================================================
# Step 2: Convert PDB to H5 format (CPU parallel)
# =================================================================
log_step "02" "Converting PDBs to H5 format (CPU)"

export JAX_PLATFORMS=cpu
export OMP_NUM_THREADS=4

for subfolder in "$tf_input_batch/pdb"/*; do
    if [[ -d "$subfolder" ]]; then
        folder_name=$(basename "$subfolder")
        log_info "Processing batch: $folder_name"

        bash "$pipeline_script_dir/02_prepare_pdb2jax.sh" \
            "$subfolder" \
            "$output_dir_jax/$folder_name" \
            "$pipeline_script_dir" \
            "$PYTHON_EXEC" > "$jax_log_dir/${folder_name}.log" 2>&1 &

        while [ $(jobs -r | wc -l) -ge "$num_jobs" ]; do
            sleep 1
        done
    fi
done

wait
log_info "All H5 conversion tasks completed."

# --- Verify H5 generation ---
log_info "Verifying H5 file generation..."
total_missing=0
for subfolder in "$tf_input_batch/pdb"/*; do
    if [[ -d "$subfolder" ]]; then
        folder_name=$(basename "$subfolder")
        h5_dir="$output_dir_jax/$folder_name"
        num_pdb=$(find "$subfolder" -name "*.pdb" | wc -l)
        num_h5=$(find "$h5_dir" -name "*.h5" 2>/dev/null | wc -l)

        if [ "$num_pdb" -ne "$num_h5" ]; then
            log_info "FAIL $folder_name: Expected $num_pdb H5, found $num_h5"
            total_missing=$((total_missing + num_pdb - num_h5))
        else
            log_info "OK   $folder_name: All $num_h5 H5 files generated."
        fi
    fi
done

if [ "$total_missing" -gt 0 ]; then
    log_info "Warning: $total_missing H5 files missing. Check logs in $jax_log_dir"
fi

# =================================================================
# Step 3: cuda Inference
# =================================================================
log_step "03" "Running TorchScore inference on cuda"

export CPU_AFFINITY_CONF=1
export TASK_QUEUE_ENABLE=2

# Scan total tasks
TOTAL_TASKS=0
for json_batch_dir in "$tf_input_batch/json"/batch_*; do
    [[ -d "$json_batch_dir" ]] || continue
    count=$(find "$json_batch_dir" -name "*.json" | wc -l)
    ((TOTAL_TASKS += count))
done

if [ "$TOTAL_TASKS" -eq 0 ]; then
    log_info "Error: No JSON files found for inference."
    exit 1
fi

log_info "Total inference tasks: $TOTAL_TASKS | Max parallel: $MAX_JOBS"
find "$output_dir_torchscore" -name "completed.done" -type f -delete 2>/dev/null

INFER_START=$(date +%s)
submitted_count=0

for json_batch_dir in "$tf_input_batch/json"/batch_*; do
    [[ -d "$json_batch_dir" ]] || continue

    batch_name=$(basename "$json_batch_dir")
    h5_batch_dir="${json_batch_dir/json/jax}"

    for json_file in "$json_batch_dir"/*.json; do
        input_name=$(basename "$json_file" .json)
        h5_file="$h5_batch_dir/${input_name}.h5"
        [[ -f "$h5_file" ]] || continue

        card_id=$((submitted_count % TOTAL_CARDS))
        task_output_dir="$output_dir_torchscore"
        mkdir -p "$task_output_dir/$input_name/logs"

        (
            export CUDA_VISIBLE_DEVICES=$card_id

            $PYTHON_EXEC "$TORCHFOLD_ROOT/run_torchscore.py" \
                --run_data_pipeline=False \
                --json_path="$json_file" \
                --path="$h5_file" \
                --init_guess=true \
                --model_dir="$MODEL_DIR" \
                --output_dir="$task_output_dir" \
                > "$task_output_dir/$input_name/logs/inference.log" 2>&1

            touch "$task_output_dir/$input_name/logs/completed.done"
        ) &

        ((submitted_count++))

        # Progress reporting
        current_done=$(find "$output_dir_torchscore" -name "completed.done" 2>/dev/null | wc -l)
        current_time=$(date +%s)
        elapsed=$((current_time - INFER_START))

        if [ "$current_done" -gt 0 ]; then
            avg_time=$(echo "scale=2; $elapsed / $current_done" | bc)
            remaining=$((TOTAL_TASKS - current_done))
            ert_seconds=$(echo "$remaining * $avg_time" | bc | cut -d. -f1)
            printf "\rProgress: %d/%d (%.1f%%) | Elapsed: %ds | ETA: %ds" \
                "$current_done" "$TOTAL_TASKS" \
                "$(echo "scale=1; $current_done * 100 / $TOTAL_TASKS" | bc)" \
                "$elapsed" "$ert_seconds"
        fi

        # Concurrency control
        while [ $(jobs -r | wc -l) -ge "$MAX_JOBS" ]; do
            sleep 1
        done
    done
done

wait
echo ""
INFER_END=$(date +%s)
log_info "All inference tasks completed in $((INFER_END - INFER_START)) seconds."

# =================================================================
# Step 4: Extract Metrics
# =================================================================
log_step "04" "Extracting and Verifying Metrics"

log_info "Extracting metrics into CSV..."
$PYTHON_EXEC "$pipeline_script_dir/03_get_metrics.py" \
    --input_pdb_dir "$input_pdb_dir" \
    --tfscore_output_dir "$output_dir_torchscore" \
    --save_metric_csv "$metric_csv" \
    --num_workers 32

# Verify
if [ -f "$metric_csv" ]; then
    expected_count=$(ls -1q "$input_pdb_dir"/*.pdb 2>/dev/null | wc -l)
    actual_count=$(tail -n +2 "$metric_csv" | wc -l)

    if [ "$actual_count" -eq "$expected_count" ]; then
        log_info "Verification OK: $actual_count records match $expected_count PDBs."
    else
        log_info "Verification WARNING: Expected $expected_count, found $actual_count records."
    fi
    log_info "Metrics CSV: $metric_csv"
else
    log_info "Error: Metrics CSV was not generated."
fi

# =================================================================
# Final Summary
# =================================================================
end_time=$(date +%s)
duration=$((end_time - start_time))
log_info "========== Pipeline finished at: $(date) =========="
log_info "Total execution time: ${duration} seconds ($((duration / 3600))h $(((duration % 3600) / 60))m $((duration % 60))s)"
log_info "Results directory: $output_dir"
