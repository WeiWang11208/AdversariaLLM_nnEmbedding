#!/usr/bin/env bash
set -euo pipefail

# Re-run PGD + natural_suffix_embedding on adv_behaviors + strong_reject datasets.
#
# Usage:
#   bash scripts/run_pgd_and_natural_suffix_embedding_strong_reject.sh
#   MODEL=Qwen/Qwen3-8B bash scripts/run_pgd_and_natural_suffix_embedding_strong_reject.sh
#   DATASETS="adv_behaviors strong_reject" bash scripts/run_pgd_and_natural_suffix_embedding_strong_reject.sh
#   IDX_EXPR='"list(range(50))"' bash scripts/run_pgd_and_natural_suffix_embedding_strong_reject.sh
#
# Notes:
# - Uses absolute `save_dir` paths because Hydra runs with `hydra.job.chdir=true`.
# - To compute decode-related suffix metrics later, both attacks enable `log_embeddings=true`.

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
  local attack_name="$1"
  local dataset_name="$2"
  local out_dir="${ROOT_DIR}/outputs/${attack_name}/${dataset_name}"
  mkdir -p "${out_dir}"

  local args=(
    "${ROOT_DIR}/run_attacks.py"
    "model=${MODEL}"
    "dataset=${dataset_name}"
    "attack=${attack_name}"
    "batch_size=${BATCH_SIZE}"
    "save_dir=${out_dir}"
  )

  if [[ -n "${IDX_EXPR}" ]]; then
    args+=("datasets.${dataset_name}.idx=${IDX_EXPR}")
  fi

  case "${attack_name}" in
    pgd)
      args+=("attacks.pgd.log_embeddings=true")
      ;;
    natural_suffix_embedding)
      args+=("attacks.natural_suffix_embedding.log_embeddings=true")
      ;;
  esac

  "${PYTHON}" "${args[@]}"
}

for dataset_name in ${DATASETS_STR}; do
  run_attack "pgd" "${dataset_name}"
  run_attack "natural_suffix_embedding" "${dataset_name}"
done

echo "Done. Outputs:"
echo "  ${ROOT_DIR}/outputs/pgd/<dataset>/YYYY-MM-DD/HH-MM-SS/"
echo "  ${ROOT_DIR}/outputs/natural_suffix_embedding/<dataset>/YYYY-MM-DD/HH-MM-SS/"
