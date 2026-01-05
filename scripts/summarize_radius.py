"""
Idea2 辅助脚本：汇总 scripts/estimate_radius.py 生成的 jsonl，输出简单统计与 CSV。

用法：
  uv run python scripts/summarize_radius.py --in outputs/idea2_radius_*.jsonl --out outputs/idea2_radius_summary.csv
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
from dataclasses import dataclass
from typing import Any


@dataclass
class Row:
    model: str
    dataset: str
    idx: str
    judge: str
    metric: str
    threshold: float
    i: int
    status: str
    r_star: float | None


def percentile(xs: list[float], p: float) -> float:
    if not xs:
        return float("nan")
    xs = sorted(xs)
    k = (len(xs) - 1) * p
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return xs[int(k)]
    return xs[f] * (c - k) + xs[c] * (k - f)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="输入 jsonl（支持 glob）")
    ap.add_argument("--out", required=True, help="输出 csv 路径")
    args = ap.parse_args()

    paths = sorted(glob.glob(args.inp))
    if not paths:
        raise SystemExit(f"No files match: {args.inp}")

    rows: list[Row] = []
    for path in paths:
        meta: dict[str, Any] | None = None
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                if rec.get("record_type") == "meta":
                    meta = rec
                    continue
                if rec.get("record_type") != "sample":
                    continue
                assert meta is not None, "meta record must come first"
                rows.append(
                    Row(
                        model=str(meta.get("model")),
                        dataset=str(meta.get("dataset")),
                        idx=str(meta.get("idx")),
                        judge=str(meta.get("judge")),
                        metric=str(meta.get("metric")),
                        threshold=float(meta.get("threshold")),
                        i=int(rec.get("i")),
                        status=str(rec.get("status")),
                        r_star=float(rec["r_star"]) if rec.get("status") == "ok" else None,
                    )
                )

    ok = [r for r in rows if r.status == "ok" and r.r_star is not None]
    failed = [r for r in rows if r.status != "ok"]
    rs = [r.r_star for r in ok if r.r_star is not None]

    print(f"files: {len(paths)}")
    print(f"samples: {len(rows)} | ok: {len(ok)} | failed: {len(failed)}")
    if rs:
        print(f"r*: mean={sum(rs)/len(rs):.4f}  p10={percentile(rs,0.1):.4f}  p50={percentile(rs,0.5):.4f}  p90={percentile(rs,0.9):.4f}")

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model", "dataset", "idx", "judge", "metric", "threshold", "i", "status", "r_star"])
        for r in rows:
            w.writerow([r.model, r.dataset, r.idx, r.judge, r.metric, r.threshold, r.i, r.status, r.r_star])
    print(f"[done] wrote: {args.out}")


if __name__ == "__main__":
    main()





