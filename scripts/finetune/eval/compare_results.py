#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved
"""Render a side-by-side table from two evaluate_interactive.py result files.

Usage:
  python scripts/finetune/eval/compare_results.py \
      --pretrained runs/eval/pretrained.json \
      --finetuned  runs/eval/finetuned.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def fmt(v):
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pretrained", type=Path, required=True)
    parser.add_argument("--finetuned", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, default=None)
    args = parser.parse_args()

    pre = json.loads(Path(args.pretrained).read_text())["summary"]
    ft = json.loads(Path(args.finetuned).read_text())["summary"]

    rows = []
    rows.append(("n_instances", pre.get("n_instances"), ft.get("n_instances")))
    for k in ("0", "1", "3"):
        rows.append(
            (
                f"mIoU @ {k} clicks",
                pre.get("mIoU_at_clicks", {}).get(k),
                ft.get("mIoU_at_clicks", {}).get(k),
            )
        )
        rows.append(
            (
                f"BoundaryIoU @ {k} clicks",
                pre.get("BoundaryIoU_at_clicks", {}).get(k),
                ft.get("BoundaryIoU_at_clicks", {}).get(k),
            )
        )
    target = next((k for k in pre if k.startswith("NoC@")), "NoC@95")
    rows.append((target, pre.get(target), ft.get(target)))
    rows.append(
        (
            target.replace("NoC", "ReachRate"),
            pre.get(target.replace("NoC@", "reach_rate_at_")),
            ft.get(target.replace("NoC@", "reach_rate_at_")),
        )
    )

    header = f"| Metric | Pretrained | Fine-tuned | Δ |"
    sep = "|---|---|---|---|"
    body = []
    for name, p, f in rows:
        delta = ""
        if isinstance(p, (int, float)) and isinstance(f, (int, float)):
            delta = f"{f - p:+.4f}" if isinstance(f, float) else f"{f - p:+d}"
        body.append(f"| {name} | {fmt(p)} | {fmt(f)} | {delta} |")
    md = "\n".join([header, sep, *body])
    print(md)
    if args.out_md:
        args.out_md.parent.mkdir(parents=True, exist_ok=True)
        args.out_md.write_text(md + "\n")
        print(f"\nwrote {args.out_md}")


if __name__ == "__main__":
    main()
