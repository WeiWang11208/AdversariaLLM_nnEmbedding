#!/usr/bin/env python3
"""Phase C: Train/fit threshold detectors"""

import argparse
import torch
import pandas as pd
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.detectors.realizability import RealizabilityDetector
from transformers import AutoModel
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc


def evaluate_detector(
    detector: RealizabilityDetector,
    benign_data: list[dict],
    attack_data: list[dict],
    fpr_target: float
):
    """Evaluate detector with a specific FPR target"""
    # Fit on benign data
    benign_embeddings = [item['embeddings'].to(detector.device) for item in benign_data]
    detector.fit(benign_embeddings, fpr_target=fpr_target)

    # Evaluate on attack data
    tp = 0
    total = len(attack_data)

    for item in attack_data:
        try:
            result = detector.detect(
                embeddings=item['embeddings'].to(detector.device),
                tokens=item.get('tokens')
            )
            if result.is_attack:
                tp += 1
        except Exception as e:
            print(f"Error detecting: {e}")
            continue

    tpr = tp / total if total > 0 else 0

    return {
        'fpr_target': fpr_target,
        'tpr': tpr,
        'threshold': detector.threshold
    }


def compute_roc_auc(detector: RealizabilityDetector, benign_data, attack_data):
    """Compute ROC-AUC"""
    y_true = []
    y_score = []

    # Benign samples (label=0)
    for item in benign_data:
        try:
            features = detector.extract_features(
                item['embeddings'].to(detector.device),
                tokens=item.get('tokens')
            )
            # Get main score
            if detector.level == 1:
                score = features['p90_nn_l2']
            elif detector.level == 2:
                score = features['seq_realizability_cost_normalized']
            else:
                score = features.get('reconstruction_consistency', features['seq_realizability_cost_normalized'])
                if 'reconstruction_consistency' in features:
                    score = 1.0 - score  # Invert for consistency

            y_true.append(0)
            y_score.append(score)
        except:
            continue

    # Attack samples (label=1)
    for item in attack_data:
        try:
            features = detector.extract_features(
                item['embeddings'].to(detector.device),
                tokens=item.get('tokens')
            )
            if detector.level == 1:
                score = features['p90_nn_l2']
            elif detector.level == 2:
                score = features['seq_realizability_cost_normalized']
            else:
                score = features.get('reconstruction_consistency', features['seq_realizability_cost_normalized'])
                if 'reconstruction_consistency' in features:
                    score = 1.0 - score

            y_true.append(1)
            y_score.append(score)
        except:
            continue

    if len(y_true) == 0:
        return 0.0, 0.0

    roc_auc = roc_auc_score(y_true, y_score)

    # Compute PR-AUC
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    pr_auc = auc(recall, precision)

    return roc_auc, pr_auc


def main():
    parser = argparse.ArgumentParser(description="Train realizability detectors")
    parser.add_argument("--data-dir", type=str, required=True,
                       help="Directory with collected data (.pt files)")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-8B",
                       help="Model ID for embedding matrix")
    parser.add_argument("--level", type=int, default=1,
                       help="Detection level (1, 2, or 3)")
    parser.add_argument("--fpr-target", type=str, default="0.01,0.05,0.10",
                       help="Target FPR values (comma-separated)")
    parser.add_argument("--topk", type=int, default=20,
                       help="Top-K for level 2")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to run on")

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    fpr_targets = [float(f) for f in args.fpr_target.split(',')]

    # Load model embedding matrix
    print(f"Loading model: {args.model}")
    model = AutoModel.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map="cpu"
    )
    embedding_matrix = model.get_input_embeddings().weight.data.float()
    embed_scale = getattr(model.config, 'embed_scale', 1.0)

    # Load datasets
    print("Loading data...")
    benign = torch.load(data_dir / "benign.pt")
    embedding_space = torch.load(data_dir / "embedding_space.pt")
    token_space = torch.load(data_dir / "token_space.pt")

    print(f"Benign: {len(benign)} samples")
    print(f"Embedding-space attacks: {len(embedding_space)} samples")
    print(f"Token-space attacks: {len(token_space)} samples")

    # Create detector
    detector = RealizabilityDetector(
        embedding_matrix=embedding_matrix,
        level=args.level,
        topk=args.topk,
        embed_scale=embed_scale,
        device=args.device
    )

    # Compute ROC-AUC first
    print(f"\n{'='*60}")
    print(f"Computing ROC-AUC for Level {args.level}")
    print(f"{'='*60}")

    roc_auc, pr_auc = compute_roc_auc(detector, benign, embedding_space)
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"PR-AUC: {pr_auc:.4f}")

    # Evaluate at different FPR targets
    print(f"\n{'='*60}")
    print(f"Evaluating against Embedding-space attacks")
    print(f"{'='*60}")

    results = []
    for fpr_target in fpr_targets:
        result = evaluate_detector(detector, benign, embedding_space, fpr_target)
        results.append(result)
        print(f"FPR={fpr_target:.2%}: TPR={result['tpr']:.2%}, threshold={result['threshold']:.6f}")

    # Test false positives on token-space attacks
    print(f"\n{'='*60}")
    print(f"False Positive Test: Token-space attacks")
    print(f"{'='*60}")

    for fpr_target in fpr_targets:
        result = evaluate_detector(detector, benign, token_space, fpr_target)
        print(f"FPR={fpr_target:.2%}: False alarm rate={result['tpr']:.2%}")

    # Save results
    results_df = pd.DataFrame(results)
    output_path = data_dir / f"detector_results_level{args.level}.csv"
    results_df.to_csv(output_path, index=False)
    print(f"\n✓ Results saved to {output_path}")


if __name__ == "__main__":
    main()
