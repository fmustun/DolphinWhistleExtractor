#!/bin/bash
#SBATCH --job-name=dolphin-pt96k
#SBATCH -C cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=slurm_logs/pretraining96k_%j.out
#SBATCH --error=slurm_logs/pretraining96k_%j.err

set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/DolphinWhistleExtractor}"
STAGE_ROOT="${STAGE_ROOT:-$SCRATCH/dolphin_pretraining_96k_stage}"
PATH_CONFIG="${PATH_CONFIG:-${STAGE_ROOT}/generated/path_config.json}"
WORK_DIR="${WORK_DIR:-$SCRATCH/dolphin_pretraining_96k_work}"
CACHE_DIR="${CACHE_DIR:-$SCRATCH/dolphin_pretraining_96k_cache}"
TARGET_REPO="${TARGET_REPO:-dolphinteam/DolphinWhistle-Pretraining-96k}"
MAX_SHARD_SIZE="${MAX_SHARD_SIZE:-500MB}"
SPLITS="${SPLITS:-train validation}"
PYTHON_BIN="${PYTHON_BIN:-python}"
ENV_ACTIVATE="${ENV_ACTIVATE:-}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

mkdir -p "${REPO_DIR}/slurm_logs"
mkdir -p "${WORK_DIR}" "${CACHE_DIR}"

if [[ -n "${ENV_ACTIVATE}" ]]; then
  # shellcheck disable=SC1090
  source "${ENV_ACTIVATE}"
fi

if [[ ! -d "${REPO_DIR}" ]]; then
  echo "Repo dir not found: ${REPO_DIR}" >&2
  exit 1
fi

if [[ ! -f "${PATH_CONFIG}" ]]; then
  echo "Path config not found: ${PATH_CONFIG}" >&2
  exit 1
fi

export HF_HOME="${HF_HOME:-$SCRATCH/.hf}"

read -r -a split_args <<< "${SPLITS}"
read -r -a extra_args <<< "${EXTRA_ARGS}"

cd "${REPO_DIR}"

echo "Repo dir    : ${REPO_DIR}"
echo "Stage root  : ${STAGE_ROOT}"
echo "Path config : ${PATH_CONFIG}"
echo "Work dir    : ${WORK_DIR}"
echo "Cache dir   : ${CACHE_DIR}"
echo "Target repo : ${TARGET_REPO}"
echo "Splits      : ${SPLITS}"
echo "HF_HOME     : ${HF_HOME}"

"${PYTHON_BIN}" "${REPO_DIR}/jeanzay_pretraining_96k/rebuild_pretraining_96k_jeanzay.py" \
  --path-config "${PATH_CONFIG}" \
  --target-repo "${TARGET_REPO}" \
  --work-dir "${WORK_DIR}" \
  --cache-dir "${CACHE_DIR}" \
  --max-shard-size "${MAX_SHARD_SIZE}" \
  --reset-run-state \
  --push \
  --splits "${split_args[@]}" \
  "${extra_args[@]}"
