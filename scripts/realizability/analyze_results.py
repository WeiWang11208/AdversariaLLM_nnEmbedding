#!/usr/bin/env python3
"""Analyze and visualize realizability features"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path


def plot_feature_distributions(df: pd.DataFrame, feature_cols: list[str], output_dir: Path):
    """Plot feature distributions by label"""
    n_features = len(feature_cols)
    fig, axes = plt.subplots(1, n_features, figsize=(5*n_features, 4))

    if n_features == 1:
        axes = [axes]

    for idx, feature in enumerate(feature_cols):
        ax = axes[idx]

        for label in ['benign', 'refusal', 'token_space', 'embedding_space']:
            if label in df['label'].values:
                data = df[df['label'] == label][feature]
                ax.hist(data, alpha=0.5, label=label, bins=30, density=True)

        ax.set_xlabel(feature, fontsize=10)
        ax.set_ylabel('Density', fontsize=10)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = output_dir / 'feature_distributions.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved plot to {output_path}")


def compute_separability(df: pd.DataFrame, feature: str):
    """Compute Cohen's d between benign and embedding_space"""
    benign = df[df['label'] == 'benign'][feature].dropna()
    embed_attack = df[df['label'] == 'embedding_space'][feature].dropna()

    if len(benign) == 0 or len(embed_attack) == 0:
        return 0.0

    mean_diff = embed_attack.mean() - benign.mean()
    pooled_std = np.sqrt((embed_attack.std()**2 + benign.std()**2) / 2)

    cohen_d = mean_diff / pooled_std if pooled_std > 0 else 0.0
    return cohen_d


def main():
    parser = argparse.ArgumentParser(description="Analyze realizability features")
    parser.add_argument("--data-dir", type=str, required=True,
                       help="Directory with feature CSVs")
    parser.add_argument("--level", type=int, default=1,
                       help="Level to analyze")

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    csv_path = data_dir / f"features_level{args.level}.csv"

    if not csv_path.exists():
        print(f"Error: {csv_path} not found")
        return

    # Load features
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} samples from {csv_path}")

    # Select key features
    if args.level == 1:
        key_features = ['mean_nn_l2', 'p90_nn_l2', 'max_nn_l2']
    elif args.level == 2:
        key_features = ['mean_nn_l2', 'seq_realizability_cost_normalized']
    else:
        key_features = ['seq_realizability_cost_normalized']

    # Print statistics
    print("\n" + "="*60)
    print("Feature Statistics by Label:")
    print("="*60)
    summary = df.groupby('label')[key_features].agg(['mean', 'std'])
    print(summary)

    # Compute separability
    print("\n" + "="*60)
    print("Separability (Cohen's d) - Benign vs Embedding-space:")
    print("="*60)
    for feature in key_features:
        cohen_d = compute_separability(df, feature)
        print(f"{feature:40s}: {cohen_d:6.3f}")

    # Plot distributions
    plot_feature_distributions(df, key_features, data_dir)

    # Correlation analysis
    print("\n" + "="*60)
    print("Feature Correlations:")
    print("="*60)
    corr = df[key_features].corr()
    print(corr)


if __name__ == "__main__":
    main()
