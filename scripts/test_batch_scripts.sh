#!/bin/bash
# Test script for batch processing functionality
# This runs a small test with 2 samples to verify everything works

set -e

echo "========================================="
echo "Testing Batch Processing Scripts"
echo "========================================="
echo "This will run a quick test with 2 samples"
echo ""

TEST_MODEL="Qwen/Qwen3-8B"
TEST_DATASET="adv_behaviors"
TEST_ATTACK="pgd"
TEST_BATCH_SIZE=1
TEST_TOTAL=2
TEST_START=0

echo "Test configuration:"
echo "  Model: $TEST_MODEL"
echo "  Dataset: $TEST_DATASET"
echo "  Attack: $TEST_ATTACK"
echo "  Batch Size: $TEST_BATCH_SIZE"
echo "  Total Samples: $TEST_TOTAL"
echo ""

read -p "Press Enter to start test..."

echo ""
echo "========================================="
echo "Test 1: Bash Script"
echo "========================================="

bash scripts/batch_run_attacks.sh \
    "$TEST_MODEL" \
    "$TEST_DATASET" \
    "$TEST_ATTACK" \
    "$TEST_BATCH_SIZE" \
    "$TEST_TOTAL" \
    "$TEST_START" \
    attacks.pgd.attack_space=embedding \
    attacks.pgd.num_steps=10

echo ""
echo "========================================="
echo "Test 2: Python Script"
echo "========================================="

python scripts/batch_run_attacks.py \
    --model "$TEST_MODEL" \
    --dataset "$TEST_DATASET" \
    --attack "$TEST_ATTACK" \
    --batch-size "$TEST_BATCH_SIZE" \
    --total-samples "$TEST_TOTAL" \
    --start-idx "$TEST_START" \
    --extra-args "attacks.pgd.attack_space=embedding attacks.pgd.num_steps=10"

echo ""
echo "========================================="
echo "Test 3: Quick Launch Script"
echo "========================================="
echo "This would normally run 520 samples."
echo "Skipping to avoid long test time."
echo "To test manually, run: bash scripts/quick_run_pgd.sh 1 0"

echo ""
echo "========================================="
echo "All tests completed successfully!"
echo "========================================="
echo ""
echo "You can now use the batch scripts for full runs:"
echo "  bash scripts/quick_run_pgd.sh 10"
echo "  OR"
echo "  python scripts/batch_run_attacks.py --model Qwen/Qwen3-8B --dataset adv_behaviors --attack pgd --batch-size 10 --extra-args \"attacks.pgd.attack_space=embedding attacks.pgd.num_steps=100\""
