#!/bin/bash
#SBATCH --job-name=dolphin-inf
#SBATCH --account=vzf@v100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=slurm_logs/dolphin_inf_%A_%a.out
#SBATCH --error=slurm_logs/dolphin_inf_%A_%a.err

set -euo pipefail

REPO_DIR="${REPO_DIR:-/lustre/fswork/projects/rech/ioc/commun/DolphinWhistleExtractor}"
RECORDINGS_DIR="${RECORDINGS_DIR:-/store/rech/ioc/commun/dolphin_audio/2022}"
CHUNK_DIR="${CHUNK_DIR:-${REPO_DIR}/slurm/chunks_2022}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_DIR}/test/inference_2022_run3_jz}"
MODEL_PATH="${MODEL_PATH:-${REPO_DIR}/models/run3/model_vgg_best.pt}"
BATCH_SIZE="${BATCH_SIZE:-16}"
MAX_WORKERS="${MAX_WORKERS:-6}"
THRESHOLD="${THRESHOLD:-0.5}"
TARGET_FS="${TARGET_FS:-96000}"
SAVE_POSITIVES="${SAVE_POSITIVES:-0}"
PYTORCH_MODULE="${PYTORCH_MODULE:-pytorch-gpu/py3/2.1.1}"

mkdir -p "${REPO_DIR}/slurm_logs"
mkdir -p "${OUTPUT_DIR}"

cd "${REPO_DIR}"
module load "${PYTORCH_MODULE}"

if [[ ! -d "${CHUNK_DIR}" ]]; then
  echo "Chunk directory not found: ${CHUNK_DIR}" >&2
  exit 1
fi

shopt -s nullglob
chunk_files=( "${CHUNK_DIR}"/chunk_*.txt )
shopt -u nullglob

if [[ ${#chunk_files[@]} -eq 0 ]]; then
  echo "No chunk files found in ${CHUNK_DIR}" >&2
  exit 1
fi

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  echo "SLURM_ARRAY_TASK_ID is not set. Submit with sbatch --array=0-$(( ${#chunk_files[@]} - 1 )) ..." >&2
  exit 1
fi

task_id="${SLURM_ARRAY_TASK_ID}"
if (( task_id < 0 || task_id >= ${#chunk_files[@]} )); then
  echo "SLURM_ARRAY_TASK_ID=${task_id} is out of range for ${#chunk_files[@]} chunk files." >&2
  exit 1
fi

chunk_file="${chunk_files[${task_id}]}"
echo "Job ${SLURM_JOB_ID:-NA} task ${task_id}"
echo "Repo dir      : ${REPO_DIR}"
echo "Recordings dir: ${RECORDINGS_DIR}"
echo "Chunk file    : ${chunk_file}"
echo "Output dir    : ${OUTPUT_DIR}"
echo "Model path    : ${MODEL_PATH}"
echo "Batch size    : ${BATCH_SIZE}"
echo "Max workers   : ${MAX_WORKERS}"
echo "Threshold     : ${THRESHOLD}"
echo "Target fs     : ${TARGET_FS}"
echo "Save positives: ${SAVE_POSITIVES}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

if [[ ! -d "${RECORDINGS_DIR}" ]]; then
  echo "Recordings directory not found: ${RECORDINGS_DIR}" >&2
  exit 1
fi

if [[ ! -f "${MODEL_PATH}" ]]; then
  echo "Model checkpoint not found: ${MODEL_PATH}" >&2
  exit 1
fi

save_flag=()
if [[ "${SAVE_POSITIVES}" == "1" ]]; then
  save_flag+=( --save_p )
fi

python Predict_and_extract/main.py \
  --model_path "${MODEL_PATH}" \
  --recordings "${RECORDINGS_DIR}" \
  --saving_folder "${OUTPUT_DIR}" \
  --specific_files "${chunk_file}" \
  --batch_size "${BATCH_SIZE}" \
  --max_workers "${MAX_WORKERS}" \
  --threshold "${THRESHOLD}" \
  --target_fs "${TARGET_FS}" \
  "${save_flag[@]}"
