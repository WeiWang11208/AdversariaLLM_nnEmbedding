#!/usr/bin/env python3
"""Phase B: Extract realizability features from embeddings"""

import argparse
import torch
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.detectors.realizability import RealizabilityDetector
from transformers import AutoModel


def extract_features_for_dataset(
    data: list[dict],
    detector: RealizabilityDetector,
    label: str
):
    """Extract features for all samples in a dataset"""
    features_list = []

    for item in tqdm(data, desc=f"Extracting {label}"):
        try:
            embeddings = item['embeddings']
            tokens = item.get('tokens')

            features = detector.extract_features(
                embeddings=embeddings,
                tokens=tokens
            )

            features['label'] = label
            features['run_idx'] = item['run_idx']
            features['step'] = item['step']
            features['source'] = item.get('source', 'unknown')

            features_list.append(features)

        except Exception as e:
            print(f"Error extracting features: {e}")
            continue

    return pd.DataFrame(features_list)


def main():
    parser = argparse.ArgumentParser(description="Extract realizability features")
    parser.add_argument("--data-dir", type=str, required=True,
                       help="Directory with collected data (.pt files)")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-8B",
                       help="Model ID for embedding matrix")
    parser.add_argument("--level", type=str, default="1,2,3",
                       help="Detection levels to extract (comma-separated)")
    parser.add_argument("--topk", type=int, default=20,
                       help="Top-K for level 2")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to run on")

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    levels = [int(l) for l in args.level.split(',')]

    # Load model embedding matrix
    print(f"Loading model: {args.model}")
    model = AutoModel.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map="cpu"  # We only need embedding matrix
    )
    embedding_matrix = model.get_input_embeddings().weight.data.float()

    # Get embed_scale if exists (for Gemma, etc.)
    embed_scale = getattr(model.config, 'embed_scale', 1.0)
    print(f"Embedding scale: {embed_scale}")

    # Load datasets
    datasets = {}
    for category in ['benign', 'refusal', 'token_space', 'embedding_space']:
        data_file = data_dir / f"{category}.pt"
        if data_file.exists():
            datasets[category] = torch.load(data_file)
            print(f"Loaded {len(datasets[category])} samples from {category}")
        else:
            print(f"Warning: {data_file} not found, skipping {category}")

    # Extract features for each level
    for level in levels:
        print(f"\n{'='*60}")
        print(f"Extracting Level {level} features")
        print(f"{'='*60}")

        detector = RealizabilityDetector(
            embedding_matrix=embedding_matrix,
            level=level,
            topk=args.topk,
            embed_scale=embed_scale,
            device=args.device
        )

        all_features = []
        for label, data in datasets.items():
            df = extract_features_for_dataset(data, detector, label)
            all_features.append(df)

        # Combine all features
        features_df = pd.concat(all_features, ignore_index=True)

        # Save
        output_path = data_dir / f"features_level{level}.csv"
        features_df.to_csv(output_path, index=False)
        print(f"\n✓ Saved features to {output_path}")

        # Print summary statistics
        print("\nSummary by label:")
        summary = features_df.groupby('label').mean(numeric_only=True)
        print(summary)

    print("\n" + "="*60)
    print("Feature extraction complete!")
    print("="*60)


if __name__ == "__main__":
    main()
