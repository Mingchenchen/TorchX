#!/bin/bash

# Args: $1=pdb_folder $2=output_folder $3=script_base_dir

export JAX_PLATFORMS=cpu
unset XLA_PYTHON_CLIENT_MEM_FRACTION
unset XLA_PYTHON_CLIENT_ALLOCATOR

PYTHON_EXEC="python"

$PYTHON_EXEC "$3/02_prepare_pdb2jax.py" \
      --pdb_folder "$1" \
      --output_folder "$2" \
      --num_workers 4
