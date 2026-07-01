#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved
"""SageMaker training entry point for SAM3 fine-tuning.

The notebook in this directory packages the SAM3 source + this script and
launches a SageMaker TrainingJob. SageMaker drops your training data into
/opt/ml/input/data/<channel>/ and expects checkpoints + model artifacts
under /opt/ml/model/ when the job finishes. This script:

  1. Installs missing runtime deps that aren't in the base PyTorch image.
  2. Resolves $PWD-relative defaults in aws_sam_finetune.yaml to the
     SageMaker channel paths via Hydra dotted-key overrides… but
     train.py doesn't accept CLI overrides, so we write a derived YAML.
  3. Invokes sam3/train/train.py with the resulting config.
  4. After training finishes, runs merge_checkpoint.py and copies the
     merged checkpoint into /opt/ml/model/ so SageMaker uploads it to S3.

Hyperparameters passed to the SageMaker Estimator are forwarded as
argparse args (max-epochs, train-batch-size, etc.).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


# SageMaker conventions
SM_MODEL_DIR = Path(os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
SM_OUTPUT_DIR = Path(os.environ.get("SM_OUTPUT_DATA_DIR", "/opt/ml/output/data"))
SM_CHANNEL_TRAIN = Path(os.environ.get("SM_CHANNEL_TRAIN", "/opt/ml/input/data/train"))
SM_CHANNEL_PRETRAINED = Path(
    os.environ.get("SM_CHANNEL_PRETRAINED", "/opt/ml/input/data/pretrained")
)


def install_runtime_deps() -> None:
    """Install SAM3 runtime deps that the base image may be missing."""
    # The base PyTorch container ships torch/torchvision; everything else
    # we list under [train] needs to be installed at job start.
    subprocess.run(
        [
            sys.executable, "-m", "pip", "install",
            "-e", ".[dev,train]",
            "--quiet",
        ],
        check=True,
    )
    # Some transitive deps that pyproject lists as base deps but were
    # discovered later — make sure they're present even on older base images.
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "einops", "psutil", "--quiet"],
        check=False,
    )


def write_runtime_config(args: argparse.Namespace) -> Path:
    """Write a config file with SageMaker-channel paths and any hyperparam
    overrides patched in. Returns the path of the derived file."""
    src = Path(args.config_template)
    if not src.exists():
        raise SystemExit(f"config template not found: {src}")

    text = src.read_text()

    # Replace the ${oc.env:PWD}-relative paths with SageMaker channel paths.
    replacements = {
        "${oc.env:PWD}/data/AWS_SAM": str(SM_CHANNEL_TRAIN / "AWS_SAM"),
        "${oc.env:PWD}/data/AWS_SAM_split": str(SM_CHANNEL_TRAIN / "AWS_SAM_split"),
        "${oc.env:PWD}/runs/aws_sam_finetune": str(SM_OUTPUT_DIR / "aws_sam_finetune"),
        "${oc.env:PWD}/sam3/assets/bpe_simple_vocab_16e6.txt.gz":
            "sam3/assets/bpe_simple_vocab_16e6.txt.gz",
        "/home/ec2-user/SageMaker/efs/Models/sam3/sam3.pt":
            str(SM_CHANNEL_PRETRAINED / "sam3.pt"),
    }
    for old, new in replacements.items():
        text = text.replace(old, new)

    # Per-job hyperparameter overrides (simple line patches; the trainer
    # has no CLI override support, so we mutate the YAML directly).
    if args.max_epochs is not None:
        text = _replace_yaml_scalar(text, "max_epochs:", args.max_epochs)
    if args.train_batch_size is not None:
        text = _replace_yaml_scalar(text, "train_batch_size:", args.train_batch_size)
    if args.lr_scale is not None:
        text = _replace_yaml_scalar(text, "lr_scale:", args.lr_scale)
    if args.num_train_workers is not None:
        text = _replace_yaml_scalar(text, "num_train_workers:", args.num_train_workers)

    out = Path("sam3/train/configs/aws_sam/aws_sam_finetune_sm.yaml")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    print(f"[train_entry] wrote derived config -> {out}")
    return out


def _replace_yaml_scalar(text: str, key: str, value) -> str:
    """Replace the first '  key: <old>' line with the supplied value. Naive
    line-based patcher — fine for top-level scalars (max_epochs, lr_scale)."""
    out_lines = []
    replaced = False
    for line in text.splitlines():
        if not replaced and key in line:
            indent = line.split(key)[0]
            out_lines.append(f"{indent}{key} {value}")
            replaced = True
        else:
            out_lines.append(line)
    if not replaced:
        print(f"[train_entry] WARNING: key {key!r} not found in template")
    return "\n".join(out_lines) + "\n"


def run_training(config_path: Path, args: argparse.Namespace) -> None:
    """Invoke sam3/train/train.py with the derived config."""
    # train.py resolves -c relative to sam3/train/configs/, so we pass
    # the relative subpath under that root.
    relative = config_path.relative_to(Path("sam3/train/configs"))
    cmd = [
        sys.executable, "sam3/train/train.py",
        "-c", str(relative),
        "--use-cluster", "0",
        "--num-gpus", str(args.num_gpus),
    ]
    print(f"[train_entry] launching: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def merge_and_export(args: argparse.Namespace) -> None:
    """After training, merge the trainer's checkpoint with the pretrained
    weights and stage the result for SageMaker to upload."""
    ckpt_dir = SM_OUTPUT_DIR / "aws_sam_finetune" / "checkpoints"
    raw_ckpt = ckpt_dir / "checkpoint.pt"
    if not raw_ckpt.exists():
        print(f"[train_entry] WARNING: no checkpoint at {raw_ckpt} — skipping merge")
        return
    pretrained = SM_CHANNEL_PRETRAINED / "sam3.pt"
    merged = SM_MODEL_DIR / "checkpoint_merged.pt"
    SM_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "scripts/finetune/merge_checkpoint.py",
        "--finetuned", str(raw_ckpt),
        "--pretrained", str(pretrained),
        "--output", str(merged),
    ]
    print(f"[train_entry] merging: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    print(f"[train_entry] merged checkpoint -> {merged}")

    # Also copy the BPE vocab so the model artifact is self-contained.
    bpe = Path("sam3/assets/bpe_simple_vocab_16e6.txt.gz")
    if bpe.exists():
        shutil.copy(bpe, SM_MODEL_DIR / bpe.name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-template",
        default="sam3/train/configs/aws_sam/aws_sam_finetune.yaml",
        help="Path to the template Hydra config to derive from.",
    )
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--train-batch-size", type=int, default=None)
    parser.add_argument("--lr-scale", type=float, default=None)
    parser.add_argument("--num-train-workers", type=int, default=None)
    parser.add_argument(
        "--skip-install", action="store_true",
        help="Skip pip install (useful when the base image already has SAM3 baked in).",
    )
    parser.add_argument(
        "--skip-merge", action="store_true",
        help="Skip the merge step (export raw checkpoint.pt instead).",
    )
    args, _ = parser.parse_known_args()

    print("[train_entry] SageMaker training job started")
    print(f"[train_entry] SM_MODEL_DIR={SM_MODEL_DIR}")
    print(f"[train_entry] SM_CHANNEL_TRAIN={SM_CHANNEL_TRAIN}")
    print(f"[train_entry] SM_CHANNEL_PRETRAINED={SM_CHANNEL_PRETRAINED}")

    if not args.skip_install:
        install_runtime_deps()

    cfg_path = write_runtime_config(args)
    run_training(cfg_path, args)

    if args.skip_merge:
        # Just copy the raw checkpoint to the model dir.
        raw = SM_OUTPUT_DIR / "aws_sam_finetune" / "checkpoints" / "checkpoint.pt"
        if raw.exists():
            SM_MODEL_DIR.mkdir(parents=True, exist_ok=True)
            shutil.copy(raw, SM_MODEL_DIR / raw.name)
    else:
        merge_and_export(args)

    print("[train_entry] done")


if __name__ == "__main__":
    main()
