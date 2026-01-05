#!/bin/bash
# Batch runner for PGD embedding-space attacks to avoid OOM
# Usage: bash scripts/batch_run_pgd.sh [batch_size] [total_samples] [start_idx]

set -e

# Default parameters
BATCH_SIZE=${1:-10}        # Number of samples per batch (default: 10)
TOTAL_SAMPLES=${2:-520}    # Total number of samples in adv_behaviors dataset
START_IDX=${3:-0}          # Starting index (default: 0, useful for resuming)

MODEL="Qwen/Qwen3-8B"
DATASET="adv_behaviors"
ATTACK="pgd"

echo "========================================="
echo "Batch PGD Attack Runner"
echo "========================================="
echo "Model: $MODEL"
echo "Dataset: $DATASET"
echo "Attack: $ATTACK"
echo "Batch Size: $BATCH_SIZE"
echo "Total Samples: $TOTAL_SAMPLES"
echo "Starting Index: $START_IDX"
echo "========================================="

# Calculate number of batches
NUM_BATCHES=$(( ($TOTAL_SAMPLES - $START_IDX + $BATCH_SIZE - 1) / $BATCH_SIZE ))

echo "Will run $NUM_BATCHES batches"
echo ""

# Loop through batches
for ((i=0; i<$NUM_BATCHES; i++)); do
    CURRENT_START=$(( $START_IDX + $i * $BATCH_SIZE ))
    CURRENT_END=$(( $CURRENT_START + $BATCH_SIZE ))

    # Don't exceed total samples
    if [ $CURRENT_END -gt $TOTAL_SAMPLES ]; then
        CURRENT_END=$TOTAL_SAMPLES
    fi

    echo "========================================="
    echo "Batch $((i+1))/$NUM_BATCHES"
    echo "Processing samples [$CURRENT_START, $CURRENT_END)"
    echo "========================================="

    # Run the attack for this batch
    python run_attacks.py \
        model=$MODEL \
        dataset=$DATASET \
        "datasets.${DATASET}.idx='list(range(${CURRENT_START},${CURRENT_END}))'" \
        attack=$ATTACK \
        attacks.pgd.attack_space=embedding \
        attacks.pgd.num_steps=100

    # Show progress
    COMPLETED=$(( $CURRENT_END - $START_IDX ))
    REMAINING=$(( $TOTAL_SAMPLES - $CURRENT_END ))
    echo ""
    echo "Progress: $COMPLETED/$TOTAL_SAMPLES samples completed ($REMAINING remaining)"
    echo ""

    # Optional: Add a small delay to ensure VRAM is freed
    sleep 2
done

echo "========================================="
echo "All batches completed!"
echo "========================================="
