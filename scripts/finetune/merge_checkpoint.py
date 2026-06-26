#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved
"""Merge a fine-tuned trainer checkpoint with the pretrained SAM3 weights
into a single self-contained file.

The SAM3 trainer only saves parameters of modules it built. With
enable_inst_interactivity=False during training (the default), the
saved checkpoint omits the SAM-1 click predictor, the SAM-2 neck convs,
and the tracker — leaving them at random init when later loaded by
inference code that builds those modules.

This script produces a single state-dict file with:
  - Fine-tuned values for every key the trainer saved.
  - Pretrained values for every key the fine-tune missed
    (sam2_convs, inst_interactive_predictor.*, tracker.*).

The output file is OSS-format (no "detector." prefix), so it loads
cleanly via build_sam3_image_model(checkpoint_path=...) without any
--pretrained-fallback flag.

Usage:
  python scripts/finetune/merge_checkpoint.py \
      --finetuned  runs/aws_sam_finetune/checkpoints/checkpoint.pt \
      --pretrained /home/ec2-user/SageMaker/efs/Models/sam3/sam3.pt \
      --output     runs/aws_sam_finetune/checkpoints/checkpoint_merged.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def _unprefix_detector(state: dict) -> dict:
    """Strip the OSS 'detector.' prefix; remap 'tracker.' to the OSS layout."""
    out = {}
    for k, v in state.items():
        if k.startswith("detector."):
            out[k[len("detector.") :]] = v
        elif k.startswith("tracker."):
            out["inst_interactive_predictor.model." + k[len("tracker.") :]] = v
        else:
            out[k] = v
    return out


def load_state(path: Path) -> dict:
    obj = torch.load(str(path), map_location="cpu", weights_only=False)
    if isinstance(obj, dict) and "model" in obj and isinstance(obj["model"], dict):
        obj = obj["model"]
    return _unprefix_detector(obj)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--finetuned", type=Path, required=True)
    parser.add_argument("--pretrained", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)

    print(f"[merge] loading pretrained: {args.pretrained}")
    pre = load_state(args.pretrained)
    print(f"[merge]   pretrained keys: {len(pre)}")

    print(f"[merge] loading fine-tuned: {args.finetuned}")
    ft = load_state(args.finetuned)
    print(f"[merge]   fine-tuned keys: {len(ft)}")

    merged = dict(pre)
    overwritten = 0
    only_in_ft = 0
    for k, v in ft.items():
        if k in merged:
            overwritten += 1
        else:
            only_in_ft += 1
        merged[k] = v
    only_in_pre = len(pre) - overwritten

    print(
        f"[merge] result: total={len(merged)}, "
        f"updated_from_ft={overwritten}, "
        f"kept_from_pretrained={only_in_pre}, "
        f"only_in_ft={only_in_ft}"
    )

    torch.save(merged, str(args.output))
    print(f"[merge] wrote {args.output} ({args.output.stat().st_size / 1e9:.2f} GB)")
    print(
        "[merge] use it directly with build_sam3_image_model(checkpoint_path=...)"
        " — no --pretrained-fallback needed."
    )


if __name__ == "__main__":
    main()
