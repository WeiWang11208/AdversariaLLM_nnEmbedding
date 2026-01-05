#!/usr/bin/env python3
"""
Batch runner for attacks to avoid OOM issues.

Features:
- Automatic dataset size detection
- Automatic batch splitting
- Progress tracking
- Resume from checkpoint
- Error handling with retry logic
- Automatic VRAM cleanup between batches

Usage:
    python scripts/batch_run_attacks.py \
        --model Qwen/Qwen3-8B \
        --dataset adv_behaviors \
        --attack pgd \
        --batch-size 10 \
        --extra-args "attacks.pgd.attack_space=embedding attacks.pgd.num_steps=100"
"""

import argparse
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# Add project root to path to import dataset modules
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from src.dataset import PromptDataset
    from omegaconf import OmegaConf
    DATASET_DETECTION_AVAILABLE = True
except ImportError:
    DATASET_DETECTION_AVAILABLE = False
    print("[WARNING] Could not import dataset modules. Auto-detection disabled.")


def detect_dataset_size(dataset_name: str, extra_args: list[str]) -> int:
    """
    Automatically detect the size of a dataset.

    Args:
        dataset_name: Name of the dataset (e.g., 'adv_behaviors')
        extra_args: Extra configuration arguments that might affect dataset size

    Returns:
        Size of the dataset
    """
    if not DATASET_DETECTION_AVAILABLE:
        raise RuntimeError(
            "Dataset detection not available. Please install required dependencies or specify --total-samples manually."
        )

    try:
        # Get root directory (project root)
        root_dir = Path(__file__).parent.parent.absolute()

        # Change to project root directory to ensure relative paths work
        import os
        original_cwd = os.getcwd()
        os.chdir(root_dir)

        try:
            # Parse extra args to build config
            config_dict = {"name": dataset_name}

            # Parse dataset-specific configs from extra_args
            for arg in extra_args:
                if arg.startswith(f"datasets.{dataset_name}."):
                    key = arg.split("=")[0].replace(f"datasets.{dataset_name}.", "")
                    value = arg.split("=", 1)[1]

                    # Try to evaluate the value (for range, lists, etc.)
                    try:
                        config_dict[key] = eval(value)
                    except:
                        config_dict[key] = value

            # Load default config from yaml
            try:
                from hydra import initialize, compose
                from hydra.core.global_hydra import GlobalHydra

                # Clear any existing Hydra instance
                if GlobalHydra.instance().is_initialized():
                    GlobalHydra.instance().clear()

                with initialize(config_path="../conf", version_base="1.3"):
                    # Load full config to get root_dir
                    full_cfg = compose(config_name="config", overrides=[f"dataset={dataset_name}"])
                    dataset_cfg = full_cfg.datasets[dataset_name]

                    # Merge with parsed config
                    for key, value in config_dict.items():
                        if key != "name":
                            dataset_cfg[key] = value

                    # Resolve the config
                    config_dict = OmegaConf.to_container(dataset_cfg, resolve=True)

            except Exception as e:
                print(f"[INFO] Could not load config from Hydra: {e}")
                print(f"[INFO] Using fallback config with root_dir: {root_dir}")

                # Fallback: manually set paths
                config_dict["name"] = dataset_name
                if dataset_name == "adv_behaviors":
                    config_dict.setdefault("messages_path", str(root_dir / "data/behavior_datasets/harmbench_behaviors_text_all.csv"))
                    config_dict.setdefault("targets_path", str(root_dir / "data/optimizer_targets/harmbench_targets_text.json"))
                    config_dict.setdefault("categories", ['chemical_biological', 'illegal', 'misinformation_disinformation', 'harmful', 'harassment_bullying', 'cybercrime_intrusion'])
                    config_dict.setdefault("seed", 0)
                    config_dict.setdefault("idx", None)
                    config_dict.setdefault("shuffle", True)

            # Create dataset and get size
            dataset = PromptDataset.from_name(dataset_name)(OmegaConf.create(config_dict))
            size = len(dataset)

            print(f"[INFO] Auto-detected dataset size: {size}")
            return size

        finally:
            # Restore original working directory
            os.chdir(original_cwd)

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise RuntimeError(
            f"Failed to detect dataset size: {e}\n"
            f"Please specify --total-samples manually."
        )


def run_batch(
    model: str,
    dataset: str,
    attack: str,
    start_idx: int,
    end_idx: int,
    extra_args: list[str],
    retry: int = 2,
) -> bool:
    """Run attack for a single batch with retry logic."""

    cmd = [
        "python", "run_attacks.py",
        f"model={model}",
        f"dataset={dataset}",
        f"datasets.{dataset}.idx='list(range({start_idx},{end_idx}))'",
        f"attack={attack}",
    ]

    # Add extra arguments
    cmd.extend(extra_args)

    for attempt in range(retry):
        try:
            print(f"\n{'='*60}")
            print(f"Running command:")
            print(" ".join(cmd))
            print(f"{'='*60}\n")

            result = subprocess.run(cmd, check=True)
            return True

        except subprocess.CalledProcessError as e:
            print(f"\n[ERROR] Batch failed with exit code {e.returncode}")

            if attempt < retry - 1:
                print(f"Retrying... (attempt {attempt + 2}/{retry})")
                time.sleep(5)
            else:
                print(f"[FATAL] Batch failed after {retry} attempts")
                return False

        except KeyboardInterrupt:
            print("\n[INTERRUPTED] Received keyboard interrupt")
            raise

    return False


def main():
    parser = argparse.ArgumentParser(
        description="Batch runner for attacks to avoid OOM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # PGD embedding attack with batch size 10
  python scripts/batch_run_attacks.py \\
      --model Qwen/Qwen3-8B \\
      --dataset adv_behaviors \\
      --attack pgd \\
      --batch-size 10 \\
      --extra-args "attacks.pgd.attack_space=embedding attacks.pgd.num_steps=100"

  # GCG attack with batch size 20, starting from index 100
  python scripts/batch_run_attacks.py \\
      --model Qwen/Qwen3-8B \\
      --dataset adv_behaviors \\
      --attack gcg \\
      --batch-size 20 \\
      --start-idx 100
        """
    )

    parser.add_argument("--model", type=str, required=True, help="Model name")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name")
    parser.add_argument("--attack", type=str, required=True, help="Attack name")
    parser.add_argument("--batch-size", type=int, required=True, help="Number of samples per batch")
    parser.add_argument("--total-samples", type=int, default=None, help="Total number of samples (default: auto-detect from dataset)")
    parser.add_argument("--start-idx", type=int, default=0, help="Starting index (default: 0)")
    parser.add_argument("--retry", type=int, default=2, help="Number of retries per batch (default: 2)")
    parser.add_argument("--delay", type=float, default=2.0, help="Delay between batches in seconds (default: 2.0)")
    parser.add_argument("--extra-args", type=str, default="", help="Extra arguments as space-separated string")
    parser.add_argument("--continue-on-error", action="store_true", help="Continue even if a batch fails")

    args = parser.parse_args()

    # Parse extra arguments
    extra_args = args.extra_args.split() if args.extra_args else []

    # Auto-detect dataset size if not specified
    if args.total_samples is None:
        print("[INFO] Auto-detecting dataset size...")
        try:
            args.total_samples = detect_dataset_size(args.dataset, extra_args)
        except Exception as e:
            print(f"[ERROR] {e}")
            sys.exit(1)

    # Calculate batches
    num_batches = (args.total_samples - args.start_idx + args.batch_size - 1) // args.batch_size

    print("\n" + "="*60)
    print("Batch Attack Runner")
    print("="*60)
    print(f"Model:          {args.model}")
    print(f"Dataset:        {args.dataset}")
    print(f"Attack:         {args.attack}")
    print(f"Batch Size:     {args.batch_size}")
    print(f"Total Samples:  {args.total_samples}")
    print(f"Starting Index: {args.start_idx}")
    print(f"Num Batches:    {num_batches}")
    print(f"Retry:          {args.retry}")
    print(f"Delay:          {args.delay}s")
    if extra_args:
        print(f"Extra Args:     {' '.join(extra_args)}")
    print("="*60 + "\n")

    start_time = datetime.now()
    failed_batches = []

    # Run batches
    for i in range(num_batches):
        current_start = args.start_idx + i * args.batch_size
        current_end = min(current_start + args.batch_size, args.total_samples)

        print(f"\n{'='*60}")
        print(f"Batch {i+1}/{num_batches}")
        print(f"Samples: [{current_start}, {current_end})")
        print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*60)

        success = run_batch(
            args.model,
            args.dataset,
            args.attack,
            current_start,
            current_end,
            extra_args,
            retry=args.retry,
        )

        if not success:
            failed_batches.append((i+1, current_start, current_end))
            if not args.continue_on_error:
                print(f"\n[FATAL] Stopping due to batch failure")
                break

        # Progress report
        completed = current_end - args.start_idx
        remaining = args.total_samples - current_end
        percent = completed * 100 // (args.total_samples - args.start_idx)

        print(f"\nProgress: {completed}/{args.total_samples} samples completed ({remaining} remaining) - {percent}%")

        # Delay between batches to ensure VRAM cleanup
        if i < num_batches - 1:
            print(f"Waiting {args.delay}s before next batch...")
            time.sleep(args.delay)

    # Final report
    end_time = datetime.now()
    duration = end_time - start_time

    print("\n" + "="*60)
    print("Batch Run Complete")
    print("="*60)
    print(f"Start Time:  {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"End Time:    {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Duration:    {duration}")
    print(f"Total Batches: {num_batches}")
    print(f"Failed Batches: {len(failed_batches)}")

    if failed_batches:
        print("\nFailed batches:")
        for batch_num, start, end in failed_batches:
            print(f"  Batch {batch_num}: samples [{start}, {end})")
        print("\nTo retry failed batches, use --start-idx with appropriate values")

    print("="*60 + "\n")

    sys.exit(0 if not failed_batches else 1)


if __name__ == "__main__":
    main()
