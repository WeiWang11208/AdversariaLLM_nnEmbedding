# Realizability Detection Scripts

This directory contains scripts for implementing Idea 3: Embedding Realizability Detection.

## Quick Start

### 1. Collect Data (Phase A)

First, run attacks to generate data:

```bash
# See ideas/03-implementation-plan.md for full attack commands
# After attacks are complete, collect data:

python scripts/realizability/collect_data.py \
    --model Qwen/Qwen3-8B \
    --output outputs/realizability_data \
    --max-samples 300
```

### 2. Extract Features (Phase B)

```bash
python scripts/realizability/extract_features.py \
    --data-dir outputs/realizability_data \
    --model Qwen/Qwen3-8B \
    --level 1,2,3
```

### 3. Train Detector (Phase C)

```bash
# Level 1 (fastest)
python scripts/realizability/train_detector.py \
    --data-dir outputs/realizability_data \
    --model Qwen/Qwen3-8B \
    --level 1 \
    --fpr-target 0.01,0.05,0.10

# Level 2 (more accurate)
python scripts/realizability/train_detector.py \
    --data-dir outputs/realizability_data \
    --model Qwen/Qwen3-8B \
    --level 2 \
    --fpr-target 0.01,0.05,0.10
```

### 4. Ablation Experiments (Phase D)

```bash
# D1: Attack strength vs realizability
python scripts/realizability/ablation_epsilon.py \
    --data-dir outputs/realizability_data \
    --model Qwen/Qwen3-8B \
    --epsilons 0.5,1.0,2.0,5.0

# D2: Projection defense
python scripts/realizability/ablation_projection.py \
    --data-dir outputs/realizability_data \
    --model Qwen/Qwen3-8B \
    --projection none,nearest,weighted \
    --max-samples 50
```

### 5. Analyze Results

```bash
python scripts/realizability/analyze_results.py \
    --data-dir outputs/realizability_data \
    --level 1
```

## Scripts Overview

- `collect_data.py`: Collect embeddings from attack results
- `extract_features.py`: Extract realizability features (3 levels)
- `train_detector.py`: Fit thresholds and evaluate detectors
- `ablation_epsilon.py`: Analyze epsilon vs realizability trade-off
- `ablation_projection.py`: Evaluate projection defense
- `analyze_results.py`: Visualize and analyze results

## Output Files

All outputs are saved to `outputs/realizability_data/`:

- `benign.pt`, `refusal.pt`, `token_space.pt`, `embedding_space.pt`: Collected data
- `features_level{1,2,3}.csv`: Extracted features
- `detector_results_level{1,2,3}.csv`: Detection performance
- `ablation_epsilon.csv`, `ablation_epsilon.png`: Epsilon ablation results
- `ablation_projection.csv`: Projection defense results
- `feature_distributions.png`: Feature distribution plots
