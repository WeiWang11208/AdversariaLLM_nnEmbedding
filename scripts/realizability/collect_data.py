#!/usr/bin/env python3
"""Phase A: Collect embedding data from attack results

This script collects four categories of data:
1. Benign: Normal queries (or_bench, xs_test)
2. Refusal: Direct harmful queries (model refuses)
3. Token-space: GCG, BEAST, RandomSearch attacks
4. Embedding-space: PGD attacks with attack_space='embedding'
"""

import argparse
import json
import torch
from pathlib import Path
from tqdm import tqdm
from safetensors.torch import load_file
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.io_utils.database import get_filtered_and_grouped_paths
from transformers import AutoModel


def load_embedding_from_run(run_json_path: Path, model=None):
    """Load embeddings from a run.json file"""
    with open(run_json_path) as f:
        data = json.load(f)

    results = []
    for run_idx, run in enumerate(data['runs']):
        for step in run['steps']:
            embed_info = step.get('model_input_embeddings')
            tokens = step.get('model_input_tokens')

            if embed_info is None and tokens is not None:
                # Reconstruct from tokens (token-space attacks)
                if model is None:
                    print(f"Warning: tokens found but no model provided to reconstruct embeddings")
                    continue

                tokens_tensor = torch.tensor(tokens).to(model.device)
                with torch.no_grad():
                    embeddings = model.get_input_embeddings()(tokens_tensor).cpu()

                results.append({
                    'run_idx': run_idx,
                    'step': step['step'],
                    'tokens': tokens,
                    'embeddings': embeddings,
                    'source': 'tokens'
                })

            elif isinstance(embed_info, str):
                # Load from .safetensors file
                embed_path = Path(run_json_path).parent / embed_info
                if not embed_path.exists():
                    print(f"Warning: embedding file not found: {embed_path}")
                    continue

                embeddings = load_file(str(embed_path))['embeddings']
                results.append({
                    'run_idx': run_idx,
                    'step': step['step'],
                    'tokens': tokens,
                    'embeddings': embeddings,
                    'source': 'file'
                })

            elif embed_info is not None:
                # Direct embedding in JSON
                embeddings = torch.tensor(embed_info)
                results.append({
                    'run_idx': run_idx,
                    'step': step['step'],
                    'tokens': tokens,
                    'embeddings': embeddings,
                    'source': 'json'
                })

    return results


def collect_dataset(filter_config: dict, model, db_path: str, max_samples: int = None):
    """Collect data for a specific category"""
    paths_dict = get_filtered_and_grouped_paths(
        db_path=db_path,
        filter_by=filter_config
    )

    # Flatten all groups
    all_paths = []
    for group_paths in paths_dict.values():
        all_paths.extend(group_paths)

    all_data = []
    for path in tqdm(all_paths[:max_samples] if max_samples else all_paths):
        try:
            run_data = load_embedding_from_run(Path(path), model)
            all_data.extend(run_data)
        except Exception as e:
            print(f"Error loading {path}: {e}")
            continue

        if max_samples and len(all_data) >= max_samples:
            break

    return all_data[:max_samples] if max_samples else all_data


def main():
    parser = argparse.ArgumentParser(description="Collect embedding data for realizability experiments")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-8B",
                       help="Model ID to load embeddings")
    parser.add_argument("--db-path", type=str, default="outputs/runs.db",
                       help="Path to database")
    parser.add_argument("--output", type=str, default="outputs/realizability_data",
                       help="Output directory")
    parser.add_argument("--max-samples", type=int, default=None,
                       help="Maximum samples per category")

    args = parser.parse_args()

    # Load model for token-to-embedding reconstruction
    print(f"Loading model: {args.model}")
    model = AutoModel.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Define data collection configs
    datasets = {
        'benign': [
            {'dataset': 'or_bench', 'attack': 'direct'},
            {'dataset': 'xs_test', 'attack': 'direct'},
        ],
        'refusal': [
            {'dataset': 'adv_behaviors', 'attack': 'direct'},
        ],
        'token_space': [
            {'dataset': 'adv_behaviors', 'attack': 'gcg'},
            {'dataset': 'adv_behaviors', 'attack': 'beast'},
            {'dataset': 'adv_behaviors', 'attack': 'random_search'},
        ],
        'embedding_space': [
            {'dataset': 'adv_behaviors', 'attack': 'pgd',
             'attack_params': {'attack_space': 'embedding'}},
        ]
    }

    # Collect each category
    for category, configs in datasets.items():
        print(f"\n{'='*60}")
        print(f"Collecting {category} data...")
        print(f"{'='*60}")

        all_data = []
        for config in configs:
            print(f"\nFilter: {config}")
            data = collect_dataset(
                filter_config=config,
                model=model,
                db_path=args.db_path,
                max_samples=args.max_samples
            )
            all_data.extend(data)
            print(f"Collected {len(data)} samples")

        # Save
        output_path = output_dir / f"{category}.pt"
        torch.save(all_data, output_path)
        print(f"\n✓ Saved {len(all_data)} samples to {output_path}")

    print("\n" + "="*60)
    print("Data collection complete!")
    print("="*60)


if __name__ == "__main__":
    main()
