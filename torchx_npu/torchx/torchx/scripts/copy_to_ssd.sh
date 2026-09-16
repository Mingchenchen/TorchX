#!/bin/bash


set -euo pipefail

readonly SOURCE_DIR=${1:-$HOME/public_databases}
readonly TARGET_DIR=${2:-/mnt/disks/ssd/public_databases}

if [[ -d "${TARGET_DIR}" ]]; then
  echo "Target directory ${TARGET_DIR} already exists, skipping"
  exit 0
fi

# Create the parent directory if it doesn't exist
mkdir -p "$(dirname "${TARGET_DIR}")"

# Copy the databases to the SSD
echo "Copying databases from ${SOURCE_DIR} to ${TARGET_DIR}..."
cp -r "${SOURCE_DIR}" "${TARGET_DIR}"

echo "Databases copied successfully to ${TARGET_DIR}"
