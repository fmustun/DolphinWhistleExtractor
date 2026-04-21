#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

DEVICE="${1:-auto}"

run_one() {
  local config="$1"
  local output_dir="cnn_dataset/datasets/classification_f0_rebuilt_${config}"
  local artifacts_dir="cnn_dataset/artifacts_classification_f0_rebuilt_${config}"
  local cache_dir="/tmp/classification_f0_rebuild_${config}_cache"

  echo
  echo "============================================================"
  echo "Rebuilding config: ${config}"
  echo "Output dir       : ${output_dir}"
  echo "Artifacts dir    : ${artifacts_dir}"
  echo "Cache dir        : ${cache_dir}"
  echo "Device           : ${DEVICE}"
  echo "============================================================"

  python scripts/rebuild_classification_f0_dataset.py \
    --classification-config "${config}" \
    --classification-source auto \
    --device "${DEVICE}" \
    --cache-dir "${cache_dir}" \
    --output-dir "${output_dir}" \
    --artifacts-dir "${artifacts_dir}"
}

run_one default
run_one unbalanced
run_one all
