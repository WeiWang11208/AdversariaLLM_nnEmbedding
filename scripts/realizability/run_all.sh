#!/bin/bash
# Run complete realizability detection experiments

set -e  # Exit on error

MODEL="Qwen/Qwen3-8B"
OUTPUT_DIR="outputs/realizability_data"

echo "=========================================="
echo "Realizability Detection Experiments"
echo "=========================================="

# Phase A: Data Collection
echo ""
echo "[Phase A] Collecting data..."
python scripts/realizability/collect_data.py \
    --model "$MODEL" \
    --output "$OUTPUT_DIR" \
    --max-samples 300

# Phase B: Feature Extraction
echo ""
echo "[Phase B] Extracting features..."
python scripts/realizability/extract_features.py \
    --data-dir "$OUTPUT_DIR" \
    --model "$MODEL" \
    --level 1,2

# Phase C: Train Detectors
echo ""
echo "[Phase C] Training detectors..."
for level in 1 2; do
    echo "  Level $level..."
    python scripts/realizability/train_detector.py \
        --data-dir "$OUTPUT_DIR" \
        --model "$MODEL" \
        --level $level \
        --fpr-target 0.01,0.05,0.10
done

# Phase D: Ablation Studies
echo ""
echo "[Phase D] Running ablation studies..."

echo "  D1: Epsilon vs realizability..."
python scripts/realizability/ablation_epsilon.py \
    --data-dir "$OUTPUT_DIR" \
    --model "$MODEL" \
    --epsilons 0.5,1.0,2.0,5.0 \
    --max-samples 50

echo "  D2: Projection defense..."
python scripts/realizability/ablation_projection.py \
    --data-dir "$OUTPUT_DIR" \
    --model "$MODEL" \
    --projection none,nearest,weighted \
    --max-samples 50

# Analysis
echo ""
echo "[Analysis] Analyzing results..."
python scripts/realizability/analyze_results.py \
    --data-dir "$OUTPUT_DIR" \
    --level 1

echo ""
echo "=========================================="
echo "Experiments complete!"
echo "Results saved to: $OUTPUT_DIR"
echo "=========================================="
