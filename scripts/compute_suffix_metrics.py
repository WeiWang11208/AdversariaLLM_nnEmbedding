#!/usr/bin/env python3
"""
Compute suffix-related metrics for attack results.

This script computes:
1. suffix_text: The attack suffix text
2. suffix_tokens: Number of tokens in the suffix
3. suffix_ppl: Perplexity of the suffix (using target model from run.json)
4. decode_text: Text decoded from embeddings via nearest neighbor (if embeddings available)
5. decode_match: Whether decoded text matches original suffix (for embedding attacks)

Notes:
- For `natural_suffix_embedding`, `model_input_embeddings` logs per-token deltas (N_attack, D). This script reconstructs
  the perturbed embeddings by adding the deltas to the base token embeddings before NN decoding.

Usage:
    # Auto-detect model from run.json config
    python scripts/compute_suffix_metrics.py --results_dir outputs/ --recursive

    # Process single file
    python scripts/compute_suffix_metrics.py --results_file outputs/xxx/run.json

    # Override model (optional)
    python scripts/compute_suffix_metrics.py --results_dir outputs/ --model Qwen/Qwen3-8B
"""

import argparse
import copy
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

# Ensure repo root is on sys.path so `import src.*` works even when run from elsewhere.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


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

        # Find the last user message in both (most attacks modify the final user turn).
        orig_user_content = next((m.get("content", "") for m in reversed(original_prompt) if m.get("role") == "user"), "")
        attack_user_content = next((m.get("content", "") for m in reversed(model_input) if m.get("role") == "user"), "")

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

    def _attack_conversation_with_target(
        self,
        *,
        clean_conversation: list[dict],
        attack_conversation: list[dict],
    ) -> list[dict]:
        """
        Build an "attack conversation" that keeps the clean assistant target.

        Many attacks log `model_input` with the assistant content blank (generation prompt), but tokenization
        and logged embeddings may have been computed with the original target present. Keeping the clean
        target makes `prepare_conversation()` stable.
        """
        if not clean_conversation:
            return copy.deepcopy(attack_conversation)
        out = copy.deepcopy(attack_conversation)
        # If both have an assistant as the final message, copy the clean target content.
        if (
            out
            and clean_conversation
            and out[-1].get("role") == "assistant"
            and clean_conversation[-1].get("role") == "assistant"
        ):
            out[-1]["content"] = clean_conversation[-1].get("content", "")
        return out

    def _suffix_span_via_prepare_conversation(
        self,
        *,
        clean_conversation: list[dict],
        attack_conversation: list[dict],
    ) -> tuple[int, int, list[int], int] | None:
        """
        Return (suffix_start_in_full, suffix_len, suffix_token_ids, attack_prefix_len) if possible.

        Uses the repo's `prepare_conversation()` splitter, which matches the attacks' tokenization behavior
        (chat templates, tokenizer quirks, etc.) far better than re-tokenizing only the user string.
        """
        try:
            from src.lm_utils.tokenization import TokenMergeError, prepare_conversation
        except Exception:
            return None

        if not clean_conversation or not attack_conversation:
            return None
        if clean_conversation[-1].get("role") != "assistant":
            return None

        # Ensure we include the clean target content for stable segmentation.
        attack_with_target = self._attack_conversation_with_target(
            clean_conversation=clean_conversation,
            attack_conversation=attack_conversation,
        )

        try:
            parts_per_turn = prepare_conversation(self.tokenizer, clean_conversation, attack_with_target)
        except TokenMergeError:
            return None
        if not parts_per_turn:
            return None

        # We assume the suffix attack is on the last user turn.
        # Important: for multi-turn conversations, the last tuple's `pre_toks` is only the separator
        # before that user message, so we need to add the total length of previous turns.
        prefix_total = 0
        if len(parts_per_turn) > 1:
            for turn_parts in parts_per_turn[:-1]:
                prefix_total += int(torch.cat(turn_parts).size(0))

        pre_toks, attack_prefix_toks, prompt_toks, attack_suffix_toks, _, _ = parts_per_turn[-1]
        suffix_start_turn = int(pre_toks.size(0) + attack_prefix_toks.size(0) + prompt_toks.size(0))
        suffix_start = prefix_total + suffix_start_turn
        suffix_len = int(attack_suffix_toks.size(0))
        suffix_ids = attack_suffix_toks.tolist()
        attack_prefix_len = int(attack_prefix_toks.size(0))
        return suffix_start, suffix_len, suffix_ids, attack_prefix_len

    @staticmethod
    def _lcp_len(a: list[int], b: list[int]) -> int:
        n = min(len(a), len(b))
        i = 0
        while i < n and a[i] == b[i]:
            i += 1
        return i

    @staticmethod
    def _lcsuf_len(a: list[int], b: list[int], *, max_len: int | None = None) -> int:
        n = min(len(a), len(b))
        if max_len is not None:
            n = min(n, max_len)
        i = 0
        while i < n and a[-1 - i] == b[-1 - i]:
            i += 1
        return i

    def _suffix_span_via_token_diff(
        self,
        *,
        clean_conversation: list[dict],
        attack_conversation: list[dict],
        suffix_tokens: int,
    ) -> tuple[int, int, list[int]] | None:
        """
        Robust fallback: tokenize full chats (clean vs attack), then take the middle diff segment.

        This handles cases where tokenizing `suffix_text` alone does not match the in-context tokenization
        (BPE merges across the prompt/suffix boundary), and where `prepare_conversation()` fails with merges.
        """
        try:
            from src.lm_utils.tokenization import tokenize_chats
        except Exception:
            return None

        clean_with_target = self._attack_conversation_with_target(
            clean_conversation=clean_conversation,
            attack_conversation=clean_conversation,
        )
        attack_with_target = self._attack_conversation_with_target(
            clean_conversation=clean_conversation,
            attack_conversation=attack_conversation,
        )

        clean_ids = tokenize_chats([clean_with_target], self.tokenizer)[0].tolist()
        attack_ids = tokenize_chats([attack_with_target], self.tokenizer)[0].tolist()
        if not clean_ids or not attack_ids:
            return None

        lcp = self._lcp_len(clean_ids, attack_ids)
        # Ensure suffix comparison doesn't overlap the prefix.
        max_suf = min(len(clean_ids) - lcp, len(attack_ids) - lcp)
        if max_suf < 0:
            return None
        lcsuf = self._lcsuf_len(clean_ids, attack_ids, max_len=max_suf)
        mid_len = len(attack_ids) - lcp - lcsuf
        if mid_len <= 0:
            return None

        mid_start = lcp
        mid_ids = attack_ids[mid_start : mid_start + mid_len]

        # For suffix attacks, we usually want the *tail* of the diff segment.
        if suffix_tokens > 0 and len(mid_ids) > suffix_tokens:
            mid_start = mid_start + (len(mid_ids) - suffix_tokens)
            mid_ids = mid_ids[-suffix_tokens:]

        return mid_start, len(mid_ids), mid_ids

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
        # Try a couple of tokenizations: suffixes often start with a leading space in the
        # full chat template even if `suffix_text` itself doesn't.
        candidates = [suffix_text]
        if not suffix_text.startswith(" "):
            candidates.append(" " + suffix_text)

        for cand in candidates:
            suffix_ids = self.tokenizer.encode(cand, add_special_tokens=False)
            if not suffix_ids:
                continue

            # Find the last occurrence (suffix is usually near the end).
            last = -1
            for i in range(0, len(full_token_ids) - len(suffix_ids) + 1):
                if full_token_ids[i : i + len(suffix_ids)] == suffix_ids:
                    last = i
            if last != -1:
                return last, len(suffix_ids), suffix_ids

        return None

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

        if suffix_len <= 0:
            return "", []

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

    def _attach_decode_metrics(
        self,
        metrics: dict,
        decoded_text: str,
        decoded_ids: list[int],
        suffix_text: str,
        attack_params: dict,
        model_id: str,
        *,
        orig_suffix_ids: list[int] | None = None,
    ) -> None:
        decoded_text_raw = self.tokenizer.decode(decoded_ids, skip_special_tokens=False)
        decoded_ppl = self.compute_ppl(decoded_text, model_id)
        metrics["decode_text"] = [decoded_text]
        metrics["decode_text_raw"] = [decoded_text_raw]
        metrics["decode_token_ids"] = [decoded_ids]
        metrics["decode_ppl"] = [decoded_ppl]

        if orig_suffix_ids is None:
            orig_suffix_ids = self.tokenizer.encode(suffix_text, add_special_tokens=False)

        init_text = attack_params.get("optim_str_init")
        if isinstance(init_text, str) and init_text:
            init_ids = self.tokenizer.encode(init_text, add_special_tokens=False)
            init_metrics = self.compute_decode_match(
                init_text, decoded_text, init_ids, decoded_ids
            )
            metrics["decode_exact_text_match"] = [1.0 if init_metrics["exact_text_match"] else 0.0]
            metrics["decode_exact_token_match"] = [1.0 if init_metrics["exact_token_match"] else 0.0]
            metrics["decode_token_match_rate"] = [init_metrics["token_match_rate"]]
            metrics["decode_edit_distance"] = [init_metrics["edit_distance"]]
            # Retain explicit init comparison for downstream tooling.
            metrics["decode_init_exact_text_match"] = [1.0 if init_metrics["exact_text_match"] else 0.0]
            metrics["decode_init_exact_token_match"] = [1.0 if init_metrics["exact_token_match"] else 0.0]
            metrics["decode_init_token_match_rate"] = [init_metrics["token_match_rate"]]
            metrics["decode_init_edit_distance"] = [init_metrics["edit_distance"]]

            suffix_metrics = self.compute_decode_match(
                suffix_text, decoded_text, orig_suffix_ids, decoded_ids
            )
            metrics["decode_suffix_exact_text_match"] = [1.0 if suffix_metrics["exact_text_match"] else 0.0]
            metrics["decode_suffix_exact_token_match"] = [1.0 if suffix_metrics["exact_token_match"] else 0.0]
            metrics["decode_suffix_token_match_rate"] = [suffix_metrics["token_match_rate"]]
            metrics["decode_suffix_edit_distance"] = [suffix_metrics["edit_distance"]]
        else:
            match_metrics = self.compute_decode_match(
                suffix_text, decoded_text, orig_suffix_ids, decoded_ids
            )
            metrics["decode_exact_text_match"] = [1.0 if match_metrics["exact_text_match"] else 0.0]
            metrics["decode_exact_token_match"] = [1.0 if match_metrics["exact_token_match"] else 0.0]
            metrics["decode_token_match_rate"] = [match_metrics["token_match_rate"]]
            metrics["decode_edit_distance"] = [match_metrics["edit_distance"]]

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
        attack_params = config.get("attack_params", {}) if isinstance(config.get("attack_params", {}), dict) else {}

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
                suffix_from_optim_init = False
                if not suffix_text:
                    # Fallback for attacks that historically logged attacked prompts as `original_prompt`:
                    # try to recover the textual suffix from `optim_str_init` if present.
                    try:
                        attack_user_content = next((m.get("content", "") for m in reversed(model_input) if m.get("role") == "user"), "")
                    except Exception:
                        attack_user_content = ""
                    optim_str_init = attack_params.get("optim_str_init")
                    if isinstance(optim_str_init, str) and optim_str_init and attack_user_content:
                        idx = attack_user_content.rfind(optim_str_init)
                        if idx != -1:
                            suffix_text = attack_user_content[idx:]
                            suffix_from_optim_init = True
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

                # If the attack already saved decoded token ids, reuse them to avoid NN decode.
                decode_scores = step.get("scores", {}).get("decode", {})
                saved_token_ids = decode_scores.get("token_ids") if isinstance(decode_scores, dict) else None
                decoded_ids: list[int] | None = None
                if isinstance(saved_token_ids, list) and saved_token_ids:
                    if all(isinstance(t, (int, float)) for t in saved_token_ids):
                        decoded_ids = [int(t) for t in saved_token_ids]
                    elif isinstance(saved_token_ids[0], list):
                        decoded_ids = [int(t) for t in saved_token_ids[0]]
                if decoded_ids is not None:
                    decoded_text = self.tokenizer.decode(decoded_ids, skip_special_tokens=True)
                    self._attach_decode_metrics(
                        metrics,
                        decoded_text,
                        decoded_ids,
                        suffix_text,
                        attack_params,
                        effective_model_id,
                    )
                    embeddings_path = None

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
                            if embeddings.ndim != 2:
                                metrics["decode_skipped"] = [1.0]
                                metrics["decode_skip_reason"] = [f"embeddings_ndim={embeddings.ndim} shape={tuple(embeddings.shape)}"]
                                embeddings = None

                        # Random-restart logs suffix-only embeddings (N_suffix, D).
                        if embeddings is not None and attack_name == "random_restart":
                            emb_len = int(embeddings.size(0))
                            # Heuristic: treat as suffix-only if it matches suffix token count.
                            if emb_len > 0 and emb_len <= max(1, suffix_tokens + 2):
                                decoded_text, decoded_ids = self.nn_decode_embeddings(embeddings, 0, emb_len)
                                self._attach_decode_metrics(
                                    metrics,
                                    decoded_text,
                                    decoded_ids,
                                    suffix_text,
                                    attack_params,
                                    effective_model_id,
                                )
                                embeddings = None

                        if embeddings is not None:
                            # Prefer computing suffix span via `prepare_conversation` (matches attack tokenization).
                            clean_for_span = original_prompt
                            if suffix_from_optim_init:
                                # If original_prompt already contains the suffix, reconstruct a "clean" conversation
                                # by trimming the suffix from the final user message. This enables suffix span
                                # recovery even for legacy logs where clean/attack prompts are identical.
                                try:
                                    clean_guess = copy.deepcopy(original_prompt)
                                    user_idx = next((i for i in range(len(clean_guess) - 1, -1, -1) if clean_guess[i].get("role") == "user"), None)
                                    if user_idx is not None:
                                        uc = str(clean_guess[user_idx].get("content", ""))
                                        j = uc.rfind(suffix_text)
                                        if j != -1 and j + len(suffix_text) == len(uc):
                                            uc2 = uc[:j].rstrip()
                                            clean_guess[user_idx]["content"] = uc2
                                            clean_for_span = clean_guess
                                except Exception:
                                    clean_for_span = original_prompt

                            span = self._suffix_span_via_prepare_conversation(
                                clean_conversation=clean_for_span,
                                attack_conversation=model_input,
                            )
                            if span is not None and span[1] == 0 and suffix_text.strip():
                                # If `original_prompt` already contains the suffix (legacy logs), prepare_conversation
                                # will see no diff and return an empty attack span. Fall back to token search.
                                span = None

                            suffix_loc = None
                            if span is None and isinstance(model_input_tokens, list) and all(isinstance(t, int) for t in model_input_tokens):
                                suffix_loc = self.locate_suffix_in_tokens(
                                    full_token_ids=model_input_tokens,
                                    suffix_text=suffix_text,
                                )
                                if suffix_loc is not None:
                                    suffix_start, suffix_len, orig_suffix_ids = suffix_loc
                                    attack_prefix_len = 0
                            elif span is None:
                                # Fallback: if we have no logged token ids, reconstruct them via the repo tokenizer helper.
                                try:
                                    from src.lm_utils.tokenization import tokenize_chats

                                    attack_with_target = self._attack_conversation_with_target(
                                        clean_conversation=original_prompt,
                                        attack_conversation=model_input,
                                    )
                                    full_ids = tokenize_chats([attack_with_target], self.tokenizer)[0].tolist()
                                    suffix_loc = self.locate_suffix_in_tokens(
                                        full_token_ids=full_ids,
                                        suffix_text=suffix_text,
                                    )
                                    if suffix_loc is not None:
                                        suffix_start, suffix_len, orig_suffix_ids = suffix_loc
                                        attack_prefix_len = 0
                                except Exception:
                                    suffix_loc = None

                            if span is not None:
                                suffix_start, suffix_len, orig_suffix_ids, attack_prefix_len = span
                            elif suffix_loc is None:
                                # Last-resort: token-level diff on user message only (can be misaligned with chat templates).
                                suffix_start, suffix_len, orig_suffix_ids = self.rebuild_attack_mask(
                                    model_input, original_prompt
                                )
                                attack_prefix_len = 0

                            # If we still got an empty span (legacy logs: clean == attack, or token merges),
                            # try a full-chat token diff (clean vs attack).
                            if suffix_len <= 0:
                                diff = self._suffix_span_via_token_diff(
                                    clean_conversation=clean_for_span,
                                    attack_conversation=model_input,
                                    suffix_tokens=suffix_tokens,
                                )
                                if diff is not None:
                                    suffix_start, suffix_len, orig_suffix_ids = diff
                                    attack_prefix_len = 0
                            if suffix_len <= 0:
                                metrics["decode_skipped"] = [1.0]
                                metrics["decode_skip_reason"] = [f"suffix_len={suffix_len} (cannot decode)"]
                                embeddings = None

                            # Handle attacks that log deltas (e.g. natural_suffix_embedding): embeddings are (N_attack, D)
                            # and need to be added to the base token embeddings before NN decode.
                            is_delta = attack_name in {"natural_suffix_embedding", "natural_suffix_embedding_attack"}
                            if is_delta:
                                # Reconstruct attack token ids via prepare_conversation.
                                try:
                                    from src.lm_utils.tokenization import TokenMergeError, prepare_conversation
                                except Exception as e:
                                    metrics["decode_skipped"] = [1.0]
                                    metrics["decode_skip_reason"] = [f"delta_decode_import_error={e}"]
                                    embeddings = None
                                if embeddings is not None:
                                    attack_with_target = self._attack_conversation_with_target(
                                        clean_conversation=original_prompt,
                                        attack_conversation=model_input,
                                    )
                                    try:
                                        parts_per_turn = prepare_conversation(self.tokenizer, original_prompt, attack_with_target)
                                    except TokenMergeError as e:
                                        metrics["decode_skipped"] = [1.0]
                                        metrics["decode_skip_reason"] = [f"delta_prepare_conversation_failed={e}"]
                                        embeddings = None
                                    if embeddings is not None and parts_per_turn:
                                        pre_toks, attack_prefix_toks, _, attack_suffix_toks, _, _ = parts_per_turn[-1]
                                        attack_prefix_ids = attack_prefix_toks.tolist()
                                        attack_suffix_ids = attack_suffix_toks.tolist()
                                        attack_ids = attack_prefix_ids + attack_suffix_ids
                                        if embeddings.size(0) != len(attack_ids):
                                            # Sometimes we only log suffix deltas (no prefix).
                                            if embeddings.size(0) == len(attack_suffix_ids):
                                                attack_prefix_ids = []
                                                attack_ids = attack_suffix_ids
                                            else:
                                                metrics["decode_skipped"] = [1.0]
                                                metrics["decode_skip_reason"] = [f"delta_len_mismatch embeddings_n={embeddings.size(0)} attack_n={len(attack_ids)} suffix_n={len(attack_suffix_ids)}"]
                                                embeddings = None

                                        if embeddings is not None:
                                            embed_layer = self.model.get_input_embeddings()
                                            ids_t = torch.tensor(attack_ids, dtype=torch.long, device=self.model.device)
                                            with torch.no_grad():
                                                base = embed_layer(ids_t).detach().cpu().float()
                                            delta = embeddings.detach().cpu().float()
                                            perturbed = base + delta
                                            suffix_start_in_attack = len(attack_prefix_ids)
                                            suffix_len_in_attack = len(attack_suffix_ids) if attack_suffix_ids else len(orig_suffix_ids)
                                            # Override orig suffix ids for match metrics (more reliable).
                                            if attack_suffix_ids:
                                                orig_suffix_ids = attack_suffix_ids
                                            decoded_text, decoded_ids = self.nn_decode_embeddings(
                                                perturbed, suffix_start_in_attack, suffix_len_in_attack
                                            )
                                            self._attach_decode_metrics(
                                                metrics,
                                                decoded_text,
                                                decoded_ids,
                                                suffix_text,
                                                attack_params,
                                                effective_model_id,
                                                orig_suffix_ids=orig_suffix_ids,
                                            )
                                            embeddings = None  # Done

                            # Non-delta path: decode from the full-sequence embeddings.
                            if embeddings is not None:
                                if embeddings.size(0) < (suffix_start + suffix_len):
                                    metrics["decode_skipped"] = [1.0]
                                    metrics["decode_skip_reason"] = [f"embeddings_shape={tuple(embeddings.shape)} suffix_span=({suffix_start},{suffix_len})"]
                                    embeddings = None

                            if embeddings is not None:
                                decoded_text, decoded_ids = self.nn_decode_embeddings(embeddings, suffix_start, suffix_len)
                                self._attach_decode_metrics(
                                    metrics,
                                    decoded_text,
                                    decoded_ids,
                                    suffix_text,
                                    attack_params,
                                    effective_model_id,
                                    orig_suffix_ids=orig_suffix_ids,
                                )

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
