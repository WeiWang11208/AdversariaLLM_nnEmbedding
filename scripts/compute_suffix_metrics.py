#!/usr/bin/env python3
"""
Compute suffix-related metrics for attack results.

This script computes:
1. suffix_text: The attack suffix text
2. suffix_tokens: Number of tokens in the suffix
3. suffix_ppl: Perplexity of the suffix (using target model from run.json)
4. decode_text: Text decoded from embeddings via nearest neighbor (if embeddings available)
5. decode_match: Whether decoded text matches original suffix (for embedding attacks)

Usage:
    # Auto-detect model from run.json config
    python scripts/compute_suffix_metrics.py --results_dir outputs/ --recursive

    # Process single file
    python scripts/compute_suffix_metrics.py --results_file outputs/xxx/run.json

    # Override model (optional)
    python scripts/compute_suffix_metrics.py --results_dir outputs/ --model Qwen/Qwen3-8B
"""

import argparse
import glob
import json
import logging
import os
import sys
from pathlib import Path
from typing import Optional

import torch
import safetensors.torch
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class SuffixMetricsComputer:
    """Compute suffix metrics for attack results."""

    def __init__(
        self,
        device: Optional[str] = None,
        model_override: Optional[str] = None,
    ):
        if device is not None:
            self.device = device
        else:
            # torch.cuda.is_available() can itself warn/fail in some environments; fall back to CPU.
            try:
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            except Exception:
                self.device = "cpu"
        self.model = None
        self.tokenizer = None
        self.vocab_embeddings = None
        self.vocab_embeddings_norm = None
        self.current_model_id = None
        self.model_override = model_override  # Optional override for model

    def _load_model(self, model_id: str):
        """Lazy load model for both PPL computation and NN decode."""
        if self.model is not None and self.current_model_id == model_id:
            return

        from transformers import AutoModelForCausalLM, AutoTokenizer

        logger.info(f"Loading model: {model_id}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="auto",
        )
        self.model.eval()
        self.current_model_id = model_id

        # Cache vocab embeddings for NN decode
        embed_layer = self.model.get_input_embeddings()
        # Keep NN decode artifacts on CPU to avoid GPU/CPU device mismatches and large VRAM usage.
        self.vocab_embeddings = embed_layer.weight.detach().float().cpu()
        if hasattr(embed_layer, "embed_scale"):
            self.vocab_embeddings = self.vocab_embeddings * embed_layer.embed_scale.detach().float().cpu().to(self.vocab_embeddings)
        self.vocab_embeddings_norm = self.vocab_embeddings / (self.vocab_embeddings.norm(dim=-1, keepdim=True) + 1e-8)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def extract_suffix(
        self,
        model_input: list[dict],
        original_prompt: list[dict],
    ) -> str:
        """
        Extract suffix from model_input by comparing with original_prompt.

        Args:
            model_input: The full input with suffix, e.g., [{"role": "user", "content": "How to... SUFFIX"}, ...]
            original_prompt: The original prompt without suffix, e.g., [{"role": "user", "content": "How to..."}, ...]

        Returns:
            The suffix string
        """
        if not model_input or not original_prompt:
            return ""

        # Find the user message in both
        orig_user_content = ""
        attack_user_content = ""

        for msg in original_prompt:
            if msg.get("role") == "user":
                orig_user_content = msg.get("content", "")
                break

        for msg in model_input:
            if msg.get("role") == "user":
                attack_user_content = msg.get("content", "")
                break

        # Extract suffix as the difference
        if attack_user_content.startswith(orig_user_content):
            return attack_user_content[len(orig_user_content):]

        # Fallback: try to find common prefix
        min_len = min(len(orig_user_content), len(attack_user_content))
        common_len = 0
        for i in range(min_len):
            if orig_user_content[i] == attack_user_content[i]:
                common_len = i + 1
            else:
                break

        return attack_user_content[common_len:]

    def compute_ppl(self, text: str, model_id: str) -> float:
        """
        Compute perplexity of text using the target model.

        Args:
            text: The text to compute perplexity for
            model_id: The model to use for PPL computation

        Returns:
            Perplexity value (lower = more natural)
        """
        if not text or not text.strip():
            return float("inf")

        # Use override if provided, otherwise use the model_id from run.json
        effective_model_id = self.model_override or model_id
        if not effective_model_id:
            logger.warning("No model specified for PPL computation")
            return float("inf")

        self._load_model(effective_model_id)

        try:
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True,
            )
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

            with torch.no_grad():
                labels = inputs["input_ids"].clone()
                # Do not score padding tokens
                if "attention_mask" in inputs:
                    labels[inputs["attention_mask"] == 0] = -100
                outputs = self.model(**inputs, labels=labels)
                loss = outputs.loss
                ppl = torch.exp(loss.float()).item()

            return ppl if ppl < 1e6 else float("inf")
        except Exception as e:
            logger.warning(f"Failed to compute PPL for '{text[:50]}...': {e}")
            return float("inf")

    def rebuild_attack_mask(
        self,
        model_input: list[dict],
        original_prompt: list[dict],
    ) -> tuple[int, int, list[int]]:
        """
        Rebuild attack mask by comparing tokenized model_input and original_prompt.

        Returns:
            (suffix_start, suffix_len, suffix_token_ids)
        """
        # Get user content
        orig_user = ""
        attack_user = ""
        for msg in original_prompt:
            if msg.get("role") == "user":
                orig_user = msg.get("content", "")
                break
        for msg in model_input:
            if msg.get("role") == "user":
                attack_user = msg.get("content", "")
                break

        # Tokenize both
        orig_tokens = self.tokenizer.encode(orig_user, add_special_tokens=False)
        attack_tokens = self.tokenizer.encode(attack_user, add_special_tokens=False)

        # Find where they diverge
        common_len = 0
        for i in range(min(len(orig_tokens), len(attack_tokens))):
            if orig_tokens[i] == attack_tokens[i]:
                common_len = i + 1
            else:
                break

        # The suffix tokens are from common_len to end of attack_tokens
        suffix_start = common_len
        suffix_len = len(attack_tokens) - common_len

        return suffix_start, suffix_len, attack_tokens[suffix_start:]

    def locate_suffix_in_tokens(
        self,
        *,
        full_token_ids: list[int],
        suffix_text: str,
    ) -> tuple[int, int, list[int]] | None:
        """
        Best-effort: locate suffix token ids inside the already-logged full token sequence.

        This is more reliable than re-tokenizing only the user string because model_input_tokens
        typically include chat template / system tokens.
        """
        if not suffix_text.strip():
            return None
        suffix_ids = self.tokenizer.encode(suffix_text, add_special_tokens=False)
        if not suffix_ids:
            return None

        # Find the last occurrence (suffix is usually near the end).
        last = -1
        for i in range(0, len(full_token_ids) - len(suffix_ids) + 1):
            if full_token_ids[i : i + len(suffix_ids)] == suffix_ids:
                last = i
        if last == -1:
            return None
        return last, len(suffix_ids), suffix_ids

    def nn_decode_embeddings(
        self,
        embeddings: torch.Tensor,
        suffix_start: int,
        suffix_len: int,
    ) -> tuple[str, list[int]]:
        """
        Decode embeddings to tokens via nearest neighbor search.

        Args:
            embeddings: Full sequence embeddings (seq_len, dim)
            suffix_start: Start index of suffix in the sequence
            suffix_len: Length of suffix

        Returns:
            (decoded_text, decoded_token_ids)
        """
        if self.vocab_embeddings is None or self.vocab_embeddings_norm is None:
            raise RuntimeError("Model not loaded. Call _load_model first.")

        if embeddings.ndim != 2:
            raise ValueError(f"Expected 2D embeddings (seq_len, dim), got shape {tuple(embeddings.shape)}")

        # Decode on CPU to match cached vocab embeddings and avoid device mismatch errors.
        embeddings = embeddings.detach().cpu()

        # Extract suffix embeddings
        suffix_embeds = embeddings[suffix_start:suffix_start + suffix_len].float()  # (actual_len, dim)

        # Use actual length of extracted embeddings (may differ from suffix_len if embeddings are truncated)
        actual_len = suffix_embeds.size(0)
        if actual_len == 0:
            return "", []

        if actual_len != suffix_len:
            logger.warning(f"Suffix length mismatch: expected {suffix_len}, got {actual_len}")

        # Normalize for cosine similarity
        suffix_norm = suffix_embeds / (suffix_embeds.norm(dim=-1, keepdim=True) + 1e-8)
        vocab_norm = self.vocab_embeddings_norm

        # Compute similarities and find nearest neighbors (chunked over vocab to avoid OOM).
        # Vectorize over all suffix tokens for speed.
        chunk_size = 4096
        best_sim = torch.full((actual_len,), -float("inf"), dtype=torch.float32)
        best_id = torch.zeros((actual_len,), dtype=torch.long)

        for start in range(0, vocab_norm.size(0), chunk_size):
            end = min(start + chunk_size, vocab_norm.size(0))
            chunk = vocab_norm[start:end]  # (chunk, dim) on CPU
            sims = suffix_norm @ chunk.t()  # (suffix_len, chunk)
            chunk_best_sim, chunk_best_idx = sims.max(dim=-1)  # (suffix_len,)
            improved = chunk_best_sim > best_sim
            best_sim[improved] = chunk_best_sim[improved]
            best_id[improved] = chunk_best_idx[improved] + start

        # Decode to text
        decoded_ids = best_id.tolist()
        decoded_text = self.tokenizer.decode(decoded_ids, skip_special_tokens=True)

        return decoded_text, decoded_ids

    def compute_decode_match(
        self,
        original_suffix: str,
        decoded_text: str,
        original_token_ids: list[int],
        decoded_token_ids: list[int],
    ) -> dict:
        """
        Compute various decode match metrics.

        Returns:
            {
                "exact_text_match": bool,
                "exact_token_match": bool,
                "token_match_rate": float,  # percentage of matching tokens
                "edit_distance": int,
            }
        """
        # Exact text match
        exact_text_match = original_suffix.strip() == decoded_text.strip()

        # Exact token match
        exact_token_match = original_token_ids == decoded_token_ids

        # Token match rate
        if len(original_token_ids) > 0:
            matches = sum(1 for a, b in zip(original_token_ids, decoded_token_ids) if a == b)
            token_match_rate = matches / max(len(original_token_ids), len(decoded_token_ids))
        else:
            token_match_rate = 1.0 if len(decoded_token_ids) == 0 else 0.0

        # Edit distance (Levenshtein)
        edit_distance = self._levenshtein_distance(original_suffix, decoded_text)

        return {
            "exact_text_match": exact_text_match,
            "exact_token_match": exact_token_match,
            "token_match_rate": token_match_rate,
            "edit_distance": edit_distance,
        }

    @staticmethod
    def _levenshtein_distance(s1: str, s2: str) -> int:
        """Compute Levenshtein edit distance between two strings."""
        if len(s1) < len(s2):
            return SuffixMetricsComputer._levenshtein_distance(s2, s1)

        if len(s2) == 0:
            return len(s1)

        prev_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            curr_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = prev_row[j + 1] + 1
                deletions = curr_row[j] + 1
                substitutions = prev_row[j] + (c1 != c2)
                curr_row.append(min(insertions, deletions, substitutions))
            prev_row = curr_row

        return prev_row[-1]

    def process_run_file(self, run_path: str, force: bool = False) -> bool:
        """
        Process a single run.json file and add suffix metrics.

        Args:
            run_path: Path to run.json file
            force: If True, recompute even if metrics already exist

        Returns:
            True if file was modified, False otherwise
        """
        try:
            with open(run_path, "r") as f:
                run_data = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load {run_path}: {e}")
            return False

        config = run_data.get("config", {})
        runs = run_data.get("runs", [])
        attack_name = config.get("attack", "unknown")

        # Extract model_id from config
        # Priority: model_params.id (local path) > model (HuggingFace ID)
        model_id = ""
        model_params = config.get("model_params", {})
        if isinstance(model_params, dict) and model_params.get("id"):
            model_id = model_params.get("id")
        else:
            model_config = config.get("model", {})
            if isinstance(model_config, dict):
                model_id = model_config.get("id", "")
            else:
                model_id = str(model_config) if model_config else ""

        if not model_id and not self.model_override:
            logger.warning(f"No model found in config for {run_path}")
            return False

        # Use override if provided
        effective_model_id = self.model_override or model_id

        modified = False

        for run in runs:
            original_prompt = run.get("original_prompt", [])
            steps = run.get("steps", [])

            for step in steps:
                # Skip if already computed (unless force)
                if not force and "suffix_metrics" in step.get("scores", {}):
                    continue

                model_input = step.get("model_input", [])
                model_input_tokens = step.get("model_input_tokens")
                embeddings_path = step.get("model_input_embeddings")

                # Extract suffix
                suffix_text = self.extract_suffix(model_input, original_prompt)
                if not suffix_text:
                    continue

                # Load model for this run
                self._load_model(effective_model_id)

                # Compute PPL using target model
                suffix_ppl = self.compute_ppl(suffix_text, effective_model_id)

                # Count tokens using target model's tokenizer
                suffix_tokens = len(self.tokenizer.encode(suffix_text, add_special_tokens=False))

                # Initialize metrics
                metrics = {
                    "suffix_text": [suffix_text],
                    "suffix_tokens": [suffix_tokens],
                    "suffix_ppl": [suffix_ppl],
                }

                # If embeddings available, do NN decode
                if embeddings_path and os.path.exists(embeddings_path):
                    try:
                        # Load embeddings
                        embed_data = safetensors.torch.load_file(embeddings_path)
                        embeddings = embed_data.get("embeddings")
                        if embeddings is None:
                            # Try other common keys
                            for key in embed_data.keys():
                                embeddings = embed_data[key]
                                break

                        if embeddings is not None:
                            # Determine suffix span in *embedding space*.
                            # Prefer using logged `model_input_tokens` (full chat-template tokens).
                            suffix_loc = None
                            if isinstance(model_input_tokens, list) and all(isinstance(t, int) for t in model_input_tokens):
                                suffix_loc = self.locate_suffix_in_tokens(
                                    full_token_ids=model_input_tokens,
                                    suffix_text=suffix_text,
                                )
                            if suffix_loc is not None:
                                suffix_start, suffix_len, orig_suffix_ids = suffix_loc
                            else:
                                # Fallback: token-level diff on user message only (can be misaligned with chat templates).
                                suffix_start, suffix_len, orig_suffix_ids = self.rebuild_attack_mask(
                                    model_input, original_prompt
                                )

                            # If embeddings don't look like full-sequence embeddings, skip NN decode.
                            if embeddings.ndim != 2 or embeddings.size(0) < (suffix_start + suffix_len):
                                metrics["decode_skipped"] = [1.0]
                                metrics["decode_skip_reason"] = [f"embeddings_shape={tuple(embeddings.shape)} suffix_span=({suffix_start},{suffix_len})"]
                                # Do not treat this as an error: some attacks log non-sequence tensors (e.g. deltas).
                                embeddings = None

                            if embeddings is not None:
                                # NN decode
                                decoded_text, decoded_ids = self.nn_decode_embeddings(
                                    embeddings, suffix_start, suffix_len
                                )

                                # Compute decode match
                                match_metrics = self.compute_decode_match(
                                    suffix_text, decoded_text, orig_suffix_ids, decoded_ids
                                )

                                # Compute PPL of decoded text (for comparison)
                                decoded_ppl = self.compute_ppl(decoded_text, effective_model_id)

                                metrics["decode_text"] = [decoded_text]
                                metrics["decode_ppl"] = [decoded_ppl]
                                metrics["decode_exact_text_match"] = [1.0 if match_metrics["exact_text_match"] else 0.0]
                                metrics["decode_exact_token_match"] = [1.0 if match_metrics["exact_token_match"] else 0.0]
                                metrics["decode_token_match_rate"] = [match_metrics["token_match_rate"]]
                                metrics["decode_edit_distance"] = [match_metrics["edit_distance"]]

                    except Exception as e:
                        logger.warning(f"Failed to process embeddings for {run_path}: {e}")

                # Add metrics to scores
                if "scores" not in step:
                    step["scores"] = {}
                step["scores"]["suffix_metrics"] = metrics
                modified = True

        # Save back
        if modified:
            try:
                with open(run_path, "w") as f:
                    json.dump(run_data, f, indent=2, ensure_ascii=False)
                logger.info(f"Updated {run_path}")
            except Exception as e:
                logger.error(f"Failed to save {run_path}: {e}")
                return False

        return modified


def collect_run_files(args) -> list[str]:
    """Collect all run.json files based on arguments."""
    paths = []

    if args.results_file:
        if os.path.exists(args.results_file):
            paths.append(args.results_file)
        else:
            logger.error(f"File not found: {args.results_file}")

    elif args.results_dir:
        if args.recursive:
            pattern = os.path.join(args.results_dir, "**", "run.json")
            paths = glob.glob(pattern, recursive=True)
        else:
            pattern = os.path.join(args.results_dir, "*", "run.json")
            paths = glob.glob(pattern)
            if not paths:
                pattern = os.path.join(args.results_dir, "*", "*", "run.json")
                paths = glob.glob(pattern)

    return sorted(paths)


def main():
    parser = argparse.ArgumentParser(description="Compute suffix metrics for attack results")
    parser.add_argument("--results_dir", type=str, help="Directory containing run.json files")
    parser.add_argument("--results_file", type=str, help="Single run.json file to process")
    parser.add_argument("--recursive", action="store_true", help="Search recursively for run.json files")
    parser.add_argument("--model", type=str, default=None,
                        help="Override model for PPL/decode (default: auto-detect from run.json config)")
    parser.add_argument("--device", type=str, default=None, help="Device to use (default: auto)")
    parser.add_argument("--force", action="store_true", help="Recompute even if suffix_metrics already exist")

    args = parser.parse_args()

    if not args.results_dir and not args.results_file:
        parser.error("Either --results_dir or --results_file must be specified")

    # Collect run files
    run_files = collect_run_files(args)
    if not run_files:
        logger.error("No run.json files found")
        sys.exit(1)

    logger.info(f"Found {len(run_files)} run files to process")

    # Initialize computer
    computer = SuffixMetricsComputer(
        device=args.device,
        model_override=args.model,
    )

    # Process each file
    modified_count = 0
    for path in tqdm(run_files, desc="Computing suffix metrics"):
        if computer.process_run_file(path, force=args.force):
            modified_count += 1

    logger.info(f"Modified {modified_count}/{len(run_files)} files")


if __name__ == "__main__":
    main()
