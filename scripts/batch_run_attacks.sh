#!/bin/bash
# Generic batch runner for attacks to avoid OOM
# Usage: bash scripts/batch_run_attacks.sh <model> <dataset> <attack> <batch_size> [total_samples] [start_idx] [extra_args...]

set -e

# Check minimum required arguments
if [ $# -lt 4 ]; then
    echo "Usage: bash scripts/batch_run_attacks.sh <model> <dataset> <attack> <batch_size> [total_samples] [start_idx] [extra_args...]"
    echo ""
    echo "Examples:"
    echo "  # PGD embedding attack with batch size 10"
    echo "  bash scripts/batch_run_attacks.sh Qwen/Qwen3-8B adv_behaviors pgd 10 300 0 attacks.pgd.attack_space=embedding attacks.pgd.num_steps=100"
    echo ""
    echo "  # GCG attack with batch size 20, starting from index 100"
    echo "  bash scripts/batch_run_attacks.sh Qwen/Qwen3-8B adv_behaviors gcg 20 300 100"
    echo ""
    exit 1
fi

MODEL=$1
DATASET=$2
ATTACK=$3
BATCH_SIZE=$4
TOTAL_SAMPLES=${5:-300}  # Default to 300 for adv_behaviors
START_IDX=${6:-0}
shift 6

# Remaining arguments are extra config overrides
EXTRA_ARGS="$@"

echo "========================================="
echo "Batch Attack Runner"
echo "========================================="
echo "Model: $MODEL"
echo "Dataset: $DATASET"
echo "Attack: $ATTACK"
echo "Batch Size: $BATCH_SIZE"
echo "Total Samples: $TOTAL_SAMPLES"
echo "Starting Index: $START_IDX"
if [ -n "$EXTRA_ARGS" ]; then
    echo "Extra Args: $EXTRA_ARGS"
fi
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
    echo "Time: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "========================================="

    # Build the command
    CMD="python run_attacks.py \
        model=$MODEL \
        dataset=$DATASET \
        datasets.${DATASET}.idx='list(range(${CURRENT_START},${CURRENT_END}))' \
        attack=$ATTACK"

    # Add extra arguments if provided
    if [ -n "$EXTRA_ARGS" ]; then
        CMD="$CMD $EXTRA_ARGS"
    fi

    # Run the attack for this batch
    eval $CMD

    # Show progress
    COMPLETED=$(( $CURRENT_END - $START_IDX ))
    REMAINING=$(( $TOTAL_SAMPLES - $CURRENT_END ))
    PERCENT=$(( $COMPLETED * 100 / ($TOTAL_SAMPLES - $START_IDX) ))

    echo ""
    echo "Progress: $COMPLETED/$TOTAL_SAMPLES samples completed ($REMAINING remaining) - $PERCENT%"
    echo ""

    # Optional: Add a small delay to ensure VRAM is freed
    sleep 2
done

echo "========================================="
echo "All batches completed!"
echo "Finished at: $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================="
