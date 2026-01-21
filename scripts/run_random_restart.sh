#!/usr/bin/env bash
set -euo pipefail

# Run random_restart on adv_behaviors + strong_reject datasets.
#
# Usage:
#   bash scripts/run_random_restart.sh
#   MODEL=Qwen/Qwen3-8B bash scripts/run_random_restart.sh
#   DATASETS="adv_behaviors strong_reject" bash scripts/run_random_restart.sh
#   IDX_EXPR='"list(range(50))"' bash scripts/run_random_restart.sh
#
# Notes:
# - Uses absolute `save_dir` paths because Hydra runs with `hydra.job.chdir=true`.
# - `log_embeddings=true` is required for decode-related suffix metrics.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

PYTHON="${PYTHON:-}"
if [[ -z "${PYTHON}" ]]; then
  if [[ -x "${ROOT_DIR}/.venv/bin/python" ]]; then
    PYTHON="${ROOT_DIR}/.venv/bin/python"
  else
    PYTHON="python"
  fi
fi

MODEL="${MODEL:-Qwen/Qwen3-8B}"
# Space-separated list to match Hydra dataset names in `conf/datasets/datasets.yaml`.
DATASETS_STR="${DATASETS:-adv_behaviors strong_reject}"
BATCH_SIZE="${BATCH_SIZE:-1}"
IDX_EXPR="${IDX_EXPR:-}"

run_attack() {
  local dataset_name="$1"
  local out_dir="${ROOT_DIR}/outputs/random_restart/${dataset_name}"
  mkdir -p "${out_dir}"

  local args=(
    "${ROOT_DIR}/run_attacks.py"
    "model=${MODEL}"
    "dataset=${dataset_name}"
    "attack=random_restart"
    "batch_size=${BATCH_SIZE}"
    "attacks.random_restart.log_embeddings=true"
    "save_dir=${out_dir}"
  )

  if [[ -n "${IDX_EXPR}" ]]; then
    args+=("datasets.${dataset_name}.idx=${IDX_EXPR}")
  fi

  "${PYTHON}" "${args[@]}"
}

for dataset_name in ${DATASETS_STR}; do
  run_attack "${dataset_name}"
done

echo "Done. Outputs:"
echo "  ${ROOT_DIR}/outputs/random_restart/<dataset>/YYYY-MM-DD/HH-MM-SS/"
