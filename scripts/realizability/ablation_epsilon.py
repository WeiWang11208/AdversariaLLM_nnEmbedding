#!/usr/bin/env python3
"""Phase D: Ablation - Attack strength (epsilon) vs realizability"""

import argparse
import json
import torch
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from safetensors.torch import load_file
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.detectors.realizability import RealizabilityDetector
from src.io_utils.database import get_filtered_and_grouped_paths
from transformers import AutoModel


def analyze_epsilon_vs_realizability(
    epsilons: list[float],
    detector: RealizabilityDetector,
    db_path: str,
    max_samples: int = 50
):
    """Analyze relationship between PGD epsilon and realizability scores"""
    results = []

    for eps in epsilons:
        print(f"\nAnalyzing epsilon={eps}")

        # Get paths for this epsilon
        paths_dict = get_filtered_and_grouped_paths(
            db_path=db_path,
            filter_by={
                'attack': 'pgd',
                'attack_params': {
                    'attack_space': 'embedding',
                    'epsilon': eps
                }
            }
        )

        # Flatten paths
        all_paths = []
        for group_paths in paths_dict.values():
            all_paths.extend(group_paths)

        realizability_scores = []
        attack_losses = []
        success_rates = []

        for path in tqdm(all_paths[:max_samples]):
            try:
                with open(path) as f:
                    data = json.load(f)

                for run in data['runs']:
                    for step in run['steps']:
                        # Load embedding
                        embed_info = step.get('model_input_embeddings')
                        if embed_info is None:
                            continue

                        if isinstance(embed_info, str):
                            embed_path = Path(path).parent / embed_info
                            if not embed_path.exists():
                                continue
                            embeddings = load_file(str(embed_path))['embeddings']
                        else:
                            embeddings = torch.tensor(embed_info)

                        # Extract features
                        features = detector.extract_features(embeddings.to(detector.device))

                        if detector.level == 1:
                            score = features['p90_nn_l2']
                        else:
                            score = features['seq_realizability_cost_normalized']

                        realizability_scores.append(score)

                        # Record attack metrics
                        attack_losses.append(step.get('loss', float('nan')))

                        # Success rate from judge scores
                        scores = step.get('scores', {})
                        for judge_name, judge_scores in scores.items():
                            if 'p_harmful' in judge_scores:
                                p_harmful = judge_scores['p_harmful'][0]
                                success_rates.append(p_harmful > 0.5)
                                break

            except Exception as e:
                print(f"Error processing {path}: {e}")
                continue

        if realizability_scores:
            results.append({
                'epsilon': eps,
                'mean_realizability': float(torch.tensor(realizability_scores).mean()),
                'std_realizability': float(torch.tensor(realizability_scores).std()),
                'mean_loss': float(torch.tensor(attack_losses).nanmean()),
                'success_rate': float(torch.tensor(success_rates).float().mean()) if success_rates else 0.0,
                'n_samples': len(realizability_scores)
            })

            print(f"  Mean realizability: {results[-1]['mean_realizability']:.4f}")
            print(f"  Success rate: {results[-1]['success_rate']:.2%}")

    return pd.DataFrame(results)


def plot_results(df: pd.DataFrame, output_dir: Path):
    """Plot epsilon vs realizability and success rate"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Plot 1: Epsilon vs Realizability
    axes[0].errorbar(
        df['epsilon'],
        df['mean_realizability'],
        yerr=df['std_realizability'],
        marker='o',
        capsize=5
    )
    axes[0].set_xlabel('PGD Epsilon', fontsize=12)
    axes[0].set_ylabel('Realizability Score', fontsize=12)
    axes[0].set_title('Attack Strength vs Realizability', fontsize=14)
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Epsilon vs Success Rate
    axes[1].plot(df['epsilon'], df['success_rate'], marker='o', color='red', linewidth=2)
    axes[1].set_xlabel('PGD Epsilon', fontsize=12)
    axes[1].set_ylabel('Attack Success Rate', fontsize=12)
    axes[1].set_title('Attack Strength vs Success Rate', fontsize=14)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim([-0.05, 1.05])

    plt.tight_layout()
    output_path = output_dir / 'ablation_epsilon.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Plot saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Ablation: epsilon vs realizability")
    parser.add_argument("--data-dir", type=str, required=True,
                       help="Output directory")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-8B",
                       help="Model ID")
    parser.add_argument("--db-path", type=str, default="outputs/runs.db",
                       help="Database path")
    parser.add_argument("--epsilons", type=str, default="0.5,1.0,2.0,5.0",
                       help="Epsilon values to analyze")
    parser.add_argument("--level", type=int, default=2,
                       help="Detection level")
    parser.add_argument("--max-samples", type=int, default=50,
                       help="Max samples per epsilon")
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()

    epsilons = [float(e) for e in args.epsilons.split(',')]
    data_dir = Path(args.data_dir)
    data_dir.mkdir(exist_ok=True, parents=True)

    # Load model
    print(f"Loading model: {args.model}")
    model = AutoModel.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map="cpu"
    )
    embedding_matrix = model.get_input_embeddings().weight.data.float()
    embed_scale = getattr(model.config, 'embed_scale', 1.0)

    # Create detector
    detector = RealizabilityDetector(
        embedding_matrix=embedding_matrix,
        level=args.level,
        topk=20,
        embed_scale=embed_scale,
        device=args.device
    )

    # Analyze
    print(f"\n{'='*60}")
    print(f"Analyzing epsilon vs realizability")
    print(f"{'='*60}")

    df = analyze_epsilon_vs_realizability(
        epsilons=epsilons,
        detector=detector,
        db_path=args.db_path,
        max_samples=args.max_samples
    )

    # Save results
    output_path = data_dir / 'ablation_epsilon.csv'
    df.to_csv(output_path, index=False)
    print(f"\n✓ Results saved to {output_path}")

    # Print summary
    print("\nSummary:")
    print(df[['epsilon', 'mean_realizability', 'success_rate', 'n_samples']])

    # Plot
    plot_results(df, data_dir)

    print("\n" + "="*60)
    print("Ablation complete!")
    print("="*60)


if __name__ == "__main__":
    main()
