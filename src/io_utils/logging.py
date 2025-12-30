"""
Attack result logging and management utilities.

This module provides functions for logging attack results to JSON files
and managing tensor offloading for large model embeddings.
"""

import copy
import hashlib
import logging
import os
from dataclasses import asdict
import safetensors.torch
import orjson
from omegaconf import DictConfig, OmegaConf

from ..attacks import AttackResult
from .database import log_config_to_db
from .json_utils import CompactJSONEncoder


def offload_tensors(run_config, result: AttackResult, embed_dir: str):
    """Offload tensors from the AttackResult to separate .safetensors"""
    assert embed_dir is not None, "embed_dir must be set to offload tensors"
    if not any(step.model_input_embeddings is not None for run in result.runs for step in run.steps):
        return

    if not os.path.exists(embed_dir):
        os.makedirs(embed_dir, exist_ok=True)

    for i, run in enumerate(result.runs):
        for j, step in enumerate(run.steps):
            if step.model_input_embeddings is not None:
                run_hash = hashlib.sha256(f"{str(run_config)} {run.original_prompt} {step.model_completions}, {step.step}".encode()).hexdigest()
                filename = f"{run_hash}.safetensors"
                embed_path = os.path.join(embed_dir, filename)
                safetensors.torch.save_file({"embeddings": step.model_input_embeddings}, embed_path)
                logging.info(f"Saved embeddings for run {i}, step {j} to {embed_path}")
                step.model_input_embeddings = embed_path


def log_attack(run_config, result: AttackResult, cfg: DictConfig, date_time_string: str):
    """Logs the attack results to a JSON file and MongoDB."""
    save_dir = cfg.save_dir
    embed_dir = cfg.embed_dir
    for idx, run in enumerate(result.runs):
        subrun_config = copy.deepcopy(run_config)
        subrun_config.dataset_params["idx"] = [run_config.dataset_params["idx"][idx]]
        subrun_result = AttackResult(runs=[run])

        # Create a structured log message as a JSON object
        OmegaConf.resolve(subrun_config.attack_params)
        OmegaConf.resolve(subrun_config.dataset_params)
        OmegaConf.resolve(subrun_config.model_params)
        log_message = {
            "config": OmegaConf.to_container(OmegaConf.structured(subrun_config), resolve=True)
        }
        offload_tensors(subrun_config, subrun_result, embed_dir)

        log_message.update(asdict(subrun_result))
        # Find the first available run_i.json file
        i = 0
        log_dir = os.path.join(save_dir, date_time_string)
        while os.path.exists(os.path.join(log_dir, str(i), f"run.json")):
            i += 1
        log_file = os.path.join(log_dir, str(i), f"run.json")

        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        tmp_path = log_file + ".tmp"

        # Scheme A support: if the attack streamed per-step results to a JSONL file during
        # execution, assemble the final run.json by reading that file incrementally.
        stream_steps_path = getattr(run, "_stream_steps_path", None)
        if stream_steps_path:
            with open(tmp_path, "wb") as f:
                f.write(b'{"config":')
                f.write(orjson.dumps(log_message["config"], option=orjson.OPT_INDENT_2))
                f.write(b',"runs":[')

                # SingleAttackRunResult fields
                f.write(b'{"original_prompt":')
                f.write(orjson.dumps(run.original_prompt, option=orjson.OPT_INDENT_2))
                f.write(b',"steps":[')
                # The stream file contains:
                # - step_meta records (for live monitoring) -> ignored here
                # - full step dicts (asdict(AttackStepResult)) -> used verbatim here
                step_lines_by_idx: dict[int, bytes] = {}

                def _extract_step_idx(line: bytes) -> int | None:
                    key = b'"step":'
                    pos = line.find(key)
                    if pos == -1:
                        return None
                    j = pos + len(key)
                    # parse optional spaces then digits
                    while j < len(line) and line[j] in b" \t":
                        j += 1
                    k = j
                    while k < len(line) and line[k] >= 48 and line[k] <= 57:
                        k += 1
                    if k == j:
                        return None
                    return int(line[j:k])

                with open(stream_steps_path, "rb") as sf:
                    for raw in sf:
                        line = raw.strip()
                        if not line:
                            continue
                        # Ignore monitoring records
                        if b'"record_type"' in line:
                            continue
                        idx = _extract_step_idx(line)
                        if idx is None:
                            continue
                        step_lines_by_idx[idx] = line

                first = True
                for idx in sorted(step_lines_by_idx.keys()):
                    if not first:
                        f.write(b",")
                    f.write(step_lines_by_idx[idx])
                    first = False
                f.write(b'],"total_time":')
                f.write(orjson.dumps(run.total_time))
                f.write(b"}]}")
        else:
            # Default: dump the full AttackResult dict (fast, but can be slow for huge results)
            # Writing large run.json can be a bottleneck. Use fast binary JSON via orjson.
            data = orjson.dumps(log_message, option=orjson.OPT_INDENT_2)
            with open(tmp_path, "wb") as f:
                f.write(data)

        os.replace(tmp_path, log_file)
        logging.info(f"Attack logged to {log_file}")
        log_config_to_db(subrun_config, subrun_result, log_file)
