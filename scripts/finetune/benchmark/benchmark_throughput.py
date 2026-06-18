#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved
"""Throughput / latency benchmark for SAM3 image inference.

Measures three things, reported as median + p50/p90/p95/p99 latency and
images-or-prompts-per-second:

  1. set_image     — image preprocessing + ViT embedding only.
  2. text          — set_image + set_text_prompt with a fixed noun phrase.
  3. click_n       — set_image + predict_inst with N positive points
                     (varied via --click-points).

Modes 2 and 3 also report the "amortized" prompt-only cost
(prompt latency with set_image excluded), which is what matters when
the same image is queried many times.

Inputs come from a directory of images (default: data/AWS_SAM, sampled
deterministically). Warm-up runs are excluded from stats.

Example:
  python scripts/finetune/benchmark/benchmark_throughput.py \
      --image-dir data/AWS_SAM \
      --num-iters 50 --warmup 5 \
      --click-points 1 3 5 \
      --output runs/bench/sam3_pretrained.json

  # Fine-tuned weights:
  python scripts/finetune/benchmark/benchmark_throughput.py \
      --checkpoint runs/aws_sam_finetune/checkpoints/checkpoint.pt \
      --output runs/bench/sam3_finetuned.json
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image


IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


def gather_images(root: Path, n: int, seed: int) -> list[Path]:
    files = [p for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXTS]
    if not files:
        raise SystemExit(f"no images found under {root}")
    rng = random.Random(seed)
    rng.shuffle(files)
    return files[:n]


def percentiles(arr: list[float]) -> dict:
    a = np.asarray(arr, dtype=np.float64)
    return {
        "n": int(a.size),
        "median_ms": float(np.median(a) * 1000),
        "p50_ms": float(np.percentile(a, 50) * 1000),
        "p90_ms": float(np.percentile(a, 90) * 1000),
        "p95_ms": float(np.percentile(a, 95) * 1000),
        "p99_ms": float(np.percentile(a, 99) * 1000),
        "mean_ms": float(np.mean(a) * 1000),
        "throughput_per_s": float(1.0 / max(np.mean(a), 1e-9)),
    }


def cuda_sync(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def random_clicks(image: Image.Image, n: int, rng: random.Random) -> tuple[np.ndarray, np.ndarray]:
    """Return n positive clicks distributed roughly uniformly over the image."""
    w, h = image.size
    pts = np.array([[rng.uniform(w * 0.1, w * 0.9), rng.uniform(h * 0.1, h * 0.9)]
                    for _ in range(n)], dtype=np.float32)
    labels = np.ones(n, dtype=np.int64)
    return pts, labels


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-dir", type=Path, default=Path("data/AWS_SAM"))
    parser.add_argument("--num-iters", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument(
        "--click-points",
        type=int,
        nargs="+",
        default=[1, 3, 5],
        help="Number of clicks to use in the click-prompt benchmark.",
    )
    parser.add_argument(
        "--text-prompt",
        default="grass",
        help="Noun phrase used in the text-prompt benchmark.",
    )
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--bpe-path", default="sam3/assets/bpe_simple_vocab_16e6.txt.gz"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output", type=Path, default=Path("runs/bench/sam3_bench.json")
    )
    parser.add_argument(
        "--skip-set-image",
        action="store_true",
        help="Skip the image-embedding-only benchmark.",
    )
    parser.add_argument("--skip-text", action="store_true")
    parser.add_argument("--skip-click", action="store_true")
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    print(f"[bench] checkpoint={args.checkpoint or 'facebook/sam3 (HF)'}")
    print(f"[bench] device={args.device}")

    from sam3 import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    model = build_sam3_image_model(
        bpe_path=args.bpe_path,
        device=args.device,
        eval_mode=True,
        enable_segmentation=True,
        enable_inst_interactivity=not args.skip_click,
        checkpoint_path=str(args.checkpoint) if args.checkpoint else None,
        load_from_HF=args.checkpoint is None,
    )
    processor = Sam3Processor(model, device=args.device)

    n_images = args.num_iters + args.warmup
    paths = gather_images(args.image_dir, n_images, args.seed)
    images = [Image.open(p).convert("RGB") for p in paths]
    print(f"[bench] loaded {len(images)} images from {args.image_dir}")

    rng = random.Random(args.seed)
    results: dict = {
        "checkpoint": str(args.checkpoint) if args.checkpoint else "facebook/sam3",
        "device": args.device,
        "num_iters": args.num_iters,
        "warmup": args.warmup,
        "image_dir": str(args.image_dir),
    }

    # ------------------------------------------------------------------
    # 1. set_image only (image preprocessing + ViT embedding)
    # ------------------------------------------------------------------
    if not args.skip_set_image:
        print("[bench] set_image-only ...")
        latencies: list[float] = []
        with torch.inference_mode():
            for i, img in enumerate(images):
                cuda_sync(args.device)
                t0 = time.perf_counter()
                processor.set_image(img)
                cuda_sync(args.device)
                if i >= args.warmup:
                    latencies.append(time.perf_counter() - t0)
        results["set_image"] = percentiles(latencies)
        print(f"  median={results['set_image']['median_ms']:.1f} ms "
              f"throughput={results['set_image']['throughput_per_s']:.2f} img/s")

    # ------------------------------------------------------------------
    # 2. text prompt — full pipeline + amortized prompt-only
    # ------------------------------------------------------------------
    if not args.skip_text:
        print(f"[bench] text-prompt ('{args.text_prompt}') ...")
        full: list[float] = []
        prompt_only: list[float] = []
        with torch.inference_mode():
            for i, img in enumerate(images):
                cuda_sync(args.device)
                t0 = time.perf_counter()
                state = processor.set_image(img)
                cuda_sync(args.device)
                t_setimg = time.perf_counter()
                state = processor.set_text_prompt(args.text_prompt, state=state)
                cuda_sync(args.device)
                t1 = time.perf_counter()
                if i >= args.warmup:
                    full.append(t1 - t0)
                    prompt_only.append(t1 - t_setimg)
        results["text"] = {
            "full_pipeline": percentiles(full),
            "prompt_only_amortized": percentiles(prompt_only),
        }
        print(f"  full median={results['text']['full_pipeline']['median_ms']:.1f} ms")
        print(f"  prompt-only median="
              f"{results['text']['prompt_only_amortized']['median_ms']:.1f} ms")

    # ------------------------------------------------------------------
    # 3. click prompt — full + amortized, sweep click count
    # ------------------------------------------------------------------
    if not args.skip_click:
        results["click"] = {}
        for n_clicks in args.click_points:
            print(f"[bench] click-prompt n={n_clicks} ...")
            full = []
            prompt_only = []
            with torch.inference_mode():
                for i, img in enumerate(images):
                    pts, labs = random_clicks(img, n_clicks, rng)
                    cuda_sync(args.device)
                    t0 = time.perf_counter()
                    state = processor.set_image(img)
                    cuda_sync(args.device)
                    t_setimg = time.perf_counter()
                    model.predict_inst(
                        state,
                        point_coords=pts,
                        point_labels=labs,
                        multimask_output=True,
                    )
                    cuda_sync(args.device)
                    t1 = time.perf_counter()
                    if i >= args.warmup:
                        full.append(t1 - t0)
                        prompt_only.append(t1 - t_setimg)
            results["click"][f"n_{n_clicks}"] = {
                "full_pipeline": percentiles(full),
                "prompt_only_amortized": percentiles(prompt_only),
            }
            print(
                f"  full median="
                f"{results['click'][f'n_{n_clicks}']['full_pipeline']['median_ms']:.1f} ms"
                f"  prompt-only median="
                f"{results['click'][f'n_{n_clicks}']['prompt_only_amortized']['median_ms']:.1f} ms"
            )

    if args.device.startswith("cuda"):
        results["gpu"] = torch.cuda.get_device_name(0)
        results["torch_version"] = torch.__version__
        try:
            free, total = torch.cuda.mem_get_info()
            results["gpu_mem_total_gb"] = total / (1 << 30)
        except Exception:
            pass

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[bench] wrote {args.output}")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
