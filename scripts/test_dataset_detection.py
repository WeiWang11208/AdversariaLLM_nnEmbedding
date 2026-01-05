#!/usr/bin/env python3
"""
Test script to verify dataset auto-detection functionality
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.dataset import PromptDataset
from omegaconf import OmegaConf

def test_dataset_detection(dataset_name: str):
    """Test dataset size detection"""
    print(f"\n{'='*60}")
    print(f"Testing dataset: {dataset_name}")
    print('='*60)

    try:
        # Load default config
        from hydra import initialize, compose
        from hydra.core.global_hydra import GlobalHydra

        # Clear any existing Hydra instance
        if GlobalHydra.instance().is_initialized():
            GlobalHydra.instance().clear()

        with initialize(config_path="../conf/datasets", version_base="1.3"):
            cfg = compose(config_name="datasets")
            dataset_cfg = cfg[dataset_name]

            # Create dataset
            dataset = PromptDataset.from_name(dataset_name)(dataset_cfg)
            size = len(dataset)

            print(f"✓ Dataset loaded successfully")
            print(f"✓ Auto-detected size: {size}")

            # Show sample
            if size > 0:
                sample = dataset[0]
                print(f"\n✓ Sample data (first item):")
                print(f"  User message: {sample[0]['content'][:100]}...")

            return size

    except Exception as e:
        print(f"✗ Failed to detect dataset size: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    datasets_to_test = [
        "adv_behaviors",
        "alpaca",
        # "or_bench",  # Requires download
    ]

    print("\n" + "="*60)
    print("Dataset Auto-Detection Test")
    print("="*60)

    results = {}
    for dataset_name in datasets_to_test:
        size = test_dataset_detection(dataset_name)
        results[dataset_name] = size

    # Summary
    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    for dataset_name, size in results.items():
        status = f"✓ {size} samples" if size is not None else "✗ Failed"
        print(f"{dataset_name:20s}: {status}")

    print("\n" + "="*60)
    print("Test completed!")
    print("="*60)
