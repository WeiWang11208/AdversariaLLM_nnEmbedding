#!/usr/bin/env bash
set -euo pipefail

# Run baseline attacks (gcg, pgd, beast) with the same model/dataset split and
# store results under outputs/{gcg,pgd,beast}.
#
# Usage:
#   bash scripts/run_baselines.sh
#   MODEL=Qwen/Qwen3-8B IDX_EXPR='"list(range(50))"' bash scripts/run_baselines.sh
#
# Notes:
# - We pass absolute save_dir paths because Hydra runs with `hydra.job.chdir=true`,
#   so relative paths would be created under `multirun/.../`.
# - Other parameters remain at their config defaults.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

MODEL="${MODEL:-Qwen/Qwen3-8B}"
DATASET="${DATASET:-adv_behaviors}"
IDX_EXPR="${IDX_EXPR:-\"list(range(50))\"}"

run_attack() {
  local attack_name="$1"
  local out_dir="${ROOT_DIR}/outputs/${attack_name}"
  mkdir -p "${out_dir}"

  python run_attacks.py \
    "model=${MODEL}" \
    "batch_size=1" \
    "dataset=${DATASET}" \
    "datasets.${DATASET}.idx=${IDX_EXPR}" \
    "attack=${attack_name}" \
    "save_dir=${out_dir}"
}

run_attack "pgd"
run_attack "gcg"
run_attack "beast"

echo "Done. Outputs:"
echo "  ${ROOT_DIR}/outputs/gcg"
echo "  ${ROOT_DIR}/outputs/pgd"
echo "  ${ROOT_DIR}/outputs/beast"

