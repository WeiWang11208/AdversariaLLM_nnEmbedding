#!/usr/bin/env python3
"""
Plot strong_reject results from run.json files and save figures under scripts/images/.

Reads `scores.strong_reject.p_harmful` and summarizes across:
  - samples (runs[*])
  - steps (steps[*])
  - completions (model_completions[*]) -> reduced by max/mean

Examples:
  python scripts/plot_strong_reject_results.py --run-dir outputs/natural_suffix_embedding/2026-01-11/18-03-58
  python scripts/plot_strong_reject_results.py --glob 'outputs/natural_suffix_embedding/**/run.json' --threshold 0.5
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from dataclasses import dataclass

import matplotlib.pyplot as plt


def _iter_run_paths(run_dir: str | None, glob_pattern: str | None) -> list[str]:
    if (run_dir is None) == (glob_pattern is None):
        raise ValueError("Provide exactly one of --run-dir or --glob.")
    if run_dir is not None:
        pattern = os.path.join(run_dir, "*", "run.json")
        return sorted(glob.glob(pattern))
    return sorted(glob.glob(glob_pattern or "", recursive=True))


def _reduce_completion_scores(p_harmful_list: list[float], reduce_mode: str) -> float:
    if not p_harmful_list:
        return float("nan")
    if reduce_mode == "max":
        return max(p_harmful_list)
    if reduce_mode == "mean":
        return sum(p_harmful_list) / len(p_harmful_list)
    raise ValueError(f"Unknown --reduce-completions mode: {reduce_mode}")


def _get_step_score(step: dict, reduce_mode: str) -> float:
    scores = step.get("scores", {})
    sr = scores.get("strong_reject")
    if not isinstance(sr, dict):
        raise KeyError("Missing scores.strong_reject; run `run_judges.py classifier=strong_reject` first.")
    p_harmful = sr.get("p_harmful")
    if not isinstance(p_harmful, list):
        raise KeyError("Missing scores.strong_reject.p_harmful")
    return _reduce_completion_scores(p_harmful, reduce_mode=reduce_mode)


@dataclass
class Aggregates:
    step_scores_by_sample: list[list[float]]  # [sample][step]
    threshold: float

    @property
    def n_samples(self) -> int:
        return len(self.step_scores_by_sample)

    @property
    def n_steps(self) -> int:
        return max((len(s) for s in self.step_scores_by_sample), default=0)

    def step_scores_padded(self) -> list[list[float]]:
        n_steps = self.n_steps
        out: list[list[float]] = []
        for scores in self.step_scores_by_sample:
            if len(scores) < n_steps:
                out.append(scores + [scores[-1]] * (n_steps - len(scores)))
            else:
                out.append(scores)
        return out


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def _percentile(xs: list[float], q: float) -> float:
    if not xs:
        return float("nan")
    xs_sorted = sorted(xs)
    if q <= 0:
        return xs_sorted[0]
    if q >= 100:
        return xs_sorted[-1]
    k = (len(xs_sorted) - 1) * (q / 100.0)
    f = int(k)
    c = min(f + 1, len(xs_sorted) - 1)
    if f == c:
        return xs_sorted[f]
    return xs_sorted[f] * (c - k) + xs_sorted[c] * (k - f)


def load_aggregates(run_paths: list[str], threshold: float, reduce_mode: str) -> Aggregates:
    step_scores_by_sample: list[list[float]] = []

    for run_path in run_paths:
        with open(run_path, "r", encoding="utf-8") as file:
            data = json.load(file)
        for subrun in data.get("runs", []):
            steps: list[dict] = subrun.get("steps", [])
            if not steps:
                continue
            step_scores = [_get_step_score(step, reduce_mode=reduce_mode) for step in steps]
            step_scores_by_sample.append(step_scores)

    if not step_scores_by_sample:
        raise ValueError("No samples found (no runs/steps in the provided run.json files).")

    return Aggregates(step_scores_by_sample=step_scores_by_sample, threshold=threshold)


def plot_asr_by_step(aggr: Aggregates, out_path: str) -> None:
    padded = aggr.step_scores_padded()
    n_steps = aggr.n_steps
    xs = list(range(n_steps))

    # ASR curve
    asr = []
    mean_p = []
    p10 = []
    p90 = []
    for step in range(n_steps):
        col = [row[step] for row in padded]
        asr.append(sum(v > aggr.threshold for v in col) / len(col))
        mean_p.append(_mean(col))
        p10.append(_percentile(col, 10))
        p90.append(_percentile(col, 90))

    fig, ax1 = plt.subplots(figsize=(10, 4))
    ax1.plot(xs, asr, label=f"ASR@{aggr.threshold}", color="C0")
    ax1.set_xlabel("Step")
    ax1.set_ylabel("ASR", color="C0")
    ax1.tick_params(axis="y", labelcolor="C0")
    ax1.set_ylim(0.0, 1.0)
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(xs, mean_p, label="mean p_harmful", color="C1", alpha=0.9)
    ax2.fill_between(xs, p10, p90, color="C1", alpha=0.2, label="p10–p90 p_harmful")
    ax2.set_ylabel("p_harmful", color="C1")
    ax2.tick_params(axis="y", labelcolor="C1")
    ax2.set_ylim(0.0, 1.0)

    lines = ax1.get_lines() + ax2.get_lines()
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_oracle_vs_last(aggr: Aggregates, out_path: str) -> None:
    padded = aggr.step_scores_padded()
    oracle = [max(row) for row in padded]
    last = [row[-1] for row in padded]
    oracle_asr = sum(v > aggr.threshold for v in oracle) / len(oracle)
    last_asr = sum(v > aggr.threshold for v in last) / len(last)

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(["last", "oracle(max)"], [last_asr, oracle_asr], color=["C0", "C2"])
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("ASR")
    ax.set_title(f"ASR@{aggr.threshold} (n={aggr.n_samples})")
    for i, v in enumerate([last_asr, oracle_asr]):
        ax.text(i, v + 0.02, f"{v:.3f}", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_best_step_histogram(aggr: Aggregates, out_path: str) -> None:
    padded = aggr.step_scores_padded()
    best_steps = [max(range(len(row)), key=lambda i: row[i]) for row in padded]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(best_steps, bins=min(aggr.n_steps, 50), color="C3", alpha=0.9)
    ax.set_xlabel("Step index of max p_harmful (oracle)")
    ax.set_ylabel("Count")
    ax.set_title(f"Where the best step occurs (n={aggr.n_samples})")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", default=None, help="Directory containing per-sample subdirs with run.json files.")
    parser.add_argument("--glob", dest="glob_pattern", default=None, help="Glob pattern for run.json files (supports **).")
    parser.add_argument("--threshold", type=float, default=0.5, help="ASR threshold on p_harmful (default: 0.5).")
    parser.add_argument(
        "--reduce-completions",
        choices=["max", "mean"],
        default="max",
        help="How to reduce multiple completions per step into one score (default: max).",
    )
    parser.add_argument(
        "--out-dir",
        default=os.path.join("scripts", "images"),
        help="Output directory for images (default: scripts/images).",
    )
    parser.add_argument(
        "--tag",
        default=None,
        help="Optional subdirectory name under --out-dir (e.g. timestamp).",
    )
    args = parser.parse_args()

    run_paths = _iter_run_paths(args.run_dir, args.glob_pattern)
    if not run_paths:
        raise SystemExit("No run.json files found for the given input.")

    tag = args.tag
    if tag is None and args.run_dir is not None:
        tag = os.path.basename(os.path.normpath(args.run_dir))
    out_dir = os.path.join(args.out_dir, tag) if tag else args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    aggr = load_aggregates(run_paths, threshold=float(args.threshold), reduce_mode=str(args.reduce_completions))

    plot_asr_by_step(aggr, os.path.join(out_dir, "asr_by_step.png"))
    plot_oracle_vs_last(aggr, os.path.join(out_dir, "asr_oracle_vs_last.png"))
    plot_best_step_histogram(aggr, os.path.join(out_dir, "best_step_hist.png"))

    print(f"Saved images to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

