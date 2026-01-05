#!/bin/bash
# Quick launcher for PGD embedding-space attack with batch processing
# This is a convenience wrapper around batch_run_attacks.sh
#
# Usage:
#   bash scripts/batch_run_pgd.sh [batch_size] [start_idx]
#
# Examples:
#   bash scripts/batch_run_pgd.sh           # Use default batch size 10, start from 0
#   bash scripts/batch_run_pgd.sh 5         # Use batch size 5
#   bash scripts/batch_run_pgd.sh 10 100    # Use batch size 10, start from sample 100

set -e

BATCH_SIZE=${1:-10}
START_IDX=${2:-0}

MODEL="Qwen/Qwen3-8B"
DATASET="adv_behaviors"
ATTACK="pgd"
TOTAL_SAMPLES=300  # Actual size after category filtering

echo "Starting PGD Embedding Attack with Batch Processing"
echo "Batch Size: $BATCH_SIZE"
echo "Start Index: $START_IDX"
echo ""

bash scripts/batch_run_attacks.sh \
    "$MODEL" \
    "$DATASET" \
    "$ATTACK" \
    "$BATCH_SIZE" \
    "$TOTAL_SAMPLES" \
    "$START_IDX" \
    attacks.pgd.attack_space=embedding \
    attacks.pgd.num_steps=100
