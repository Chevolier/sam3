#!/usr/bin/env python3
"""Latency + throughput benchmark for a deployed SAM3 SageMaker endpoint.

Fires `--num-iters` invocations against `--endpoint` and reports p50/p90/
p95/p99 latency + throughput for each request mode. Optionally spins up
`--concurrency` parallel client threads to measure sustained QPS under
load.

Two modes are exercised (subset via `--modes`):

  text     — {"mode":"text",  image_b64, text, confidence}
  click    — {"mode":"click", image_b64, points=[[x,y]], labels=[1],
              multimask_output=true}

Warm-up invocations are excluded from the reported stats. The client
uses the SageMaker runtime API, so latency numbers include TLS setup,
network round-trip, TorchServe queueing, and container-side inference —
i.e., end-to-end user-perceived time, not GPU-only kernel time. Compare
against the local-GPU numbers from
`scripts/finetune/benchmark/benchmark_throughput.py` to see the
network-and-serving overhead your endpoint adds.

Usage:
  python scripts/finetune/benchmark/benchmark_endpoint.py \\
      --endpoint sam3-aws-sam-20260701-092709 \\
      --image-dir data/AWS_SAM \\
      --text grass \\
      --num-iters 50 --warmup 5 \\
      --concurrency 1 \\
      --output runs/bench/endpoint_g5xl.json

  # With 4 clients hitting the endpoint in parallel:
  python scripts/finetune/benchmark/benchmark_endpoint.py \\
      --endpoint sam3-aws-sam-20260701-092709 --concurrency 4 \\
      --modes text --num-iters 100
"""

from __future__ import annotations

import argparse
import base64
import json
import random
import statistics
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


def gather_images(root: Path, n: int, seed: int) -> list[Path]:
    files = [p for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXTS]
    if not files:
        raise SystemExit(f"no images found under {root}")
    rng = random.Random(seed)
    rng.shuffle(files)
    # Cycle through the pool if the caller wants more iters than images.
    if n <= len(files):
        return files[:n]
    return [files[i % len(files)] for i in range(n)]


def percentiles(arr: list[float]) -> dict:
    a = sorted(arr)
    n = len(a)
    if n == 0:
        return {"n": 0}

    def pct(p):
        # Linear interpolation between the two nearest ranks — matches
        # numpy default. Avoids the numpy dep in this pure-boto3 script.
        k = (n - 1) * (p / 100.0)
        f = int(k)
        c = min(f + 1, n - 1)
        return a[f] + (a[c] - a[f]) * (k - f)

    return {
        "n": n,
        "min_ms":    float(a[0] * 1000),
        "median_ms": float(pct(50) * 1000),
        "p50_ms":    float(pct(50) * 1000),
        "p90_ms":    float(pct(90) * 1000),
        "p95_ms":    float(pct(95) * 1000),
        "p99_ms":    float(pct(99) * 1000),
        "max_ms":    float(a[-1] * 1000),
        "mean_ms":   float(statistics.fmean(a) * 1000),
        "stddev_ms": float(statistics.pstdev(a) * 1000) if n > 1 else 0.0,
    }


def build_body(mode: str, image_bytes: bytes, args: argparse.Namespace, rng: random.Random) -> dict:
    """Assemble a JSON body matching what sagemaker/deploy/inference.py expects."""
    b64 = base64.b64encode(image_bytes).decode("ascii")
    if mode == "text":
        return {
            "mode": "text",
            "image_b64": b64,
            "text": args.text,
            "confidence": args.confidence,
        }
    if mode == "click":
        # Random positive click in the middle 80% of the image, so the
        # measurement isn't polluted by edge-case rejections. Coords are
        # image-space pixels; the handler doesn't check them against the
        # decoded image size at request time.
        x = rng.randint(int(args.image_w * 0.1), int(args.image_w * 0.9))
        y = rng.randint(int(args.image_h * 0.1), int(args.image_h * 0.9))
        return {
            "mode": "click",
            "image_b64": b64,
            "points": [[x, y]],
            "labels": [1],
            "multimask_output": True,
        }
    raise ValueError(f"unknown mode: {mode}")


def invoke_once(
    smr,
    endpoint: str,
    body: dict,
) -> tuple[float, int, str | None]:
    """Send one request, return (elapsed_seconds, n_masks, error_or_None)."""
    payload = json.dumps(body)
    t0 = time.perf_counter()
    try:
        resp = smr.invoke_endpoint(
            EndpointName=endpoint,
            ContentType="application/json",
            Body=payload,
        )
        out = json.loads(resp["Body"].read())
        dt = time.perf_counter() - t0
        return dt, len(out.get("masks_rle", [])), None
    except Exception as e:
        dt = time.perf_counter() - t0
        return dt, 0, f"{type(e).__name__}: {e}"


def bench_mode(
    smr,
    endpoint: str,
    mode: str,
    images: list[Path],
    args: argparse.Namespace,
) -> dict:
    """Sequential or parallel benchmark of one request mode."""
    rng = random.Random(args.seed + hash(mode) % (2**31))

    # Pre-load the image bytes once — reading from disk on each iter would
    # skew microbenchmarks on slow disks.
    image_bytes = [p.read_bytes() for p in images]

    print(f"\n=== mode={mode}  concurrency={args.concurrency}  "
          f"iters={args.num_iters}  warmup={args.warmup} ===")

    # Warm-up: fire args.warmup requests serially to let the container
    # finish JIT / cuDNN plan search. NOT counted toward stats.
    for i in range(args.warmup):
        body = build_body(mode, image_bytes[i % len(image_bytes)], args, rng)
        dt, n, err = invoke_once(smr, endpoint, body)
        if err:
            print(f"  warmup #{i}: FAIL ({err})")
        else:
            print(f"  warmup #{i}: {dt*1000:6.1f}ms  n_masks={n}")

    # Real measurement.
    lat: list[float] = []
    errors: list[str] = []
    n_masks: list[int] = []
    lock = threading.Lock()

    def worker(idx: int) -> None:
        body = build_body(mode, image_bytes[idx % len(image_bytes)], args, rng)
        dt, n, err = invoke_once(smr, endpoint, body)
        with lock:
            if err:
                errors.append(err)
            else:
                lat.append(dt)
                n_masks.append(n)

    t0 = time.perf_counter()
    if args.concurrency <= 1:
        for i in range(args.num_iters):
            worker(i)
    else:
        with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
            futures = [ex.submit(worker, i) for i in range(args.num_iters)]
            for f in as_completed(futures):
                f.result()  # re-raise any thread-side crash
    wall = time.perf_counter() - t0

    stats = percentiles(lat)
    stats["errors"] = len(errors)
    stats["error_examples"] = errors[:3]
    stats["wall_s"] = wall
    # Throughput measured two ways:
    # - per-request: 1 / mean_ms   → what one client sees
    # - wall-clock:  n / wall_s    → what the endpoint sustains under load
    stats["throughput_per_request"] = (1000.0 / stats["mean_ms"]) if stats.get("mean_ms") else 0.0
    stats["throughput_sustained"]   = len(lat) / wall if wall > 0 else 0.0
    stats["avg_masks"] = statistics.fmean(n_masks) if n_masks else 0.0

    _print_stats(stats)
    return stats


def _print_stats(s: dict) -> None:
    print(
        f"  n={s['n']}  errors={s['errors']}  wall={s['wall_s']:.2f}s\n"
        f"  latency  p50={s.get('p50_ms', 0):6.1f}  "
        f"p90={s.get('p90_ms', 0):6.1f}  "
        f"p95={s.get('p95_ms', 0):6.1f}  "
        f"p99={s.get('p99_ms', 0):6.1f}  ms\n"
        f"  min={s.get('min_ms', 0):6.1f}  "
        f"mean={s.get('mean_ms', 0):6.1f}  "
        f"max={s.get('max_ms', 0):6.1f}  "
        f"stddev={s.get('stddev_ms', 0):5.1f}  ms\n"
        f"  throughput per-request={s['throughput_per_request']:.2f} req/s  "
        f"sustained={s['throughput_sustained']:.2f} req/s\n"
        f"  avg_masks_returned={s['avg_masks']:.2f}"
    )
    if s["errors"]:
        for e in s["error_examples"]:
            print(f"    error: {e}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", required=True, help="Deployed endpoint name.")
    parser.add_argument("--region", default=None,
                        help="AWS region (default: from AWS_REGION env or boto default).")
    parser.add_argument("--image-dir", type=Path, default=Path("data/AWS_SAM"))
    parser.add_argument("--num-iters", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--concurrency", type=int, default=1,
                        help="Parallel client threads (uses ThreadPoolExecutor).")
    parser.add_argument("--modes", nargs="+", default=["text", "click"],
                        choices=["text", "click"])
    parser.add_argument("--text", default="grass",
                        help="Noun-phrase prompt for text-mode requests.")
    parser.add_argument("--confidence", type=float, default=0.5)
    parser.add_argument("--image-w", type=int, default=1280,
                        help="Approx image width used to pick random click coords.")
    parser.add_argument("--image-h", type=int, default=960,
                        help="Approx image height used to pick random click coords.")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--read-timeout", type=int, default=300,
                        help="Socket read-timeout in seconds (default 300).")
    parser.add_argument("--output", type=Path, default=None,
                        help="Write results as JSON.")
    args = parser.parse_args()

    # Deferred import so --help works without boto3 installed.
    import boto3
    from botocore.config import Config

    cfg = Config(
        read_timeout=args.read_timeout,
        connect_timeout=10,
        retries={"max_attempts": 3, "mode": "standard"},
        # Bump the connection pool so --concurrency>1 doesn't serialize
        # on the default max_pool_connections=10.
        max_pool_connections=max(10, args.concurrency * 4),
    )
    session = boto3.Session(region_name=args.region)
    smr = session.client("sagemaker-runtime", config=cfg)

    # Sample images up front so every mode benchmarks against the same set.
    images = gather_images(args.image_dir, args.num_iters + args.warmup, args.seed)
    print(f"[bench] endpoint={args.endpoint}  images={len(set(images))}  "
          f"iters={args.num_iters}  warmup={args.warmup}  concurrency={args.concurrency}")

    results: dict[str, Any] = {
        "endpoint":    args.endpoint,
        "region":      session.region_name,
        "image_dir":   str(args.image_dir),
        "num_iters":   args.num_iters,
        "warmup":      args.warmup,
        "concurrency": args.concurrency,
        "text":        args.text,
        "confidence":  args.confidence,
        "modes":       {},
    }

    for mode in args.modes:
        results["modes"][mode] = bench_mode(smr, args.endpoint, mode, images, args)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2))
        print(f"\nresults → {args.output}")


if __name__ == "__main__":
    main()
