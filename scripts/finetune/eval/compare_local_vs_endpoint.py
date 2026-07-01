#!/usr/bin/env python3
"""Side-by-side comparison of local checkpoint vs. deployed SageMaker endpoint.

Runs the SAME image + text/click prompt through:
  1. `runs/aws_sam_finetune/checkpoints/checkpoint_merged.pt` loaded locally
     via `build_sam3_image_model` + `Sam3Processor` (same code path as
     `sagemaker/deploy/inference.py::model_fn`).
  2. The deployed endpoint via `sagemaker-runtime.invoke_endpoint`.

Emits per-mask IoU between the two, side-by-side PNG overlays, and the
scores. Use this to confirm the endpoint is doing what the local checkpoint
would do (or to catch skew from library-version drift on the container).

Usage:
  python scripts/finetune/eval/compare_local_vs_endpoint.py \\
      --checkpoint runs/aws_sam_finetune/checkpoints/checkpoint_merged.pt \\
      --endpoint   sam3-aws-sam-20260701-XXXXXX \\
      --image      data/AWS_SAM/companypremises2025101600217.png \\
      --text       grass \\
      --confidence 0.5 \\
      --output     runs/eval/compare
"""

from __future__ import annotations

import argparse
import base64
import json
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from pycocotools import mask as mask_utils


# ---- geometry helpers ----------------------------------------------------


def iou(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(bool)
    b = b.astype(bool)
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter) / float(union) if union else 1.0


def rle_to_mask(rle) -> np.ndarray:
    if isinstance(rle["counts"], str):
        rle = {**rle, "counts": rle["counts"].encode("ascii")}
    return mask_utils.decode(rle).astype(np.uint8)


# ---- local inference (mirrors sagemaker/deploy/inference.py::_predict_text) ----


def local_text(
    image: Image.Image,
    text: str,
    confidence: float,
    checkpoint: Path,
    bpe_path: Path,
    device: str,
) -> dict:
    """Run local model with the same recipe the container uses."""
    from sam3 import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    print(f"[local]  building model from {checkpoint}")
    model = build_sam3_image_model(
        bpe_path=str(bpe_path),
        device=device,
        eval_mode=True,
        enable_segmentation=True,
        enable_inst_interactivity=True,
        checkpoint_path=str(checkpoint),
        load_from_HF=False,
    )
    # Belt-and-suspenders: builder does model.to(device) internally, but some
    # sub-modules stashed by activation-checkpoint wrappers can end up left on
    # CPU. Force everything onto the target device before Sam3Processor caches
    # any features. Without this, patch_embed.proj triggers:
    #   Input type (CUDABFloat16Type) and weight type (torch.FloatTensor)
    #   should be the same
    model = model.to(device)
    processor = Sam3Processor(model, device=device)
    processor.confidence_threshold = confidence

    # bf16 autocast matches the container (also needed to satisfy the
    # perflib.fused.addmm_act kernel).
    dev_type = device.split(":")[0]
    amp_ctx = (
        torch.autocast(dev_type, dtype=torch.bfloat16)
        if dev_type == "cuda"
        else torch.autocast("cpu", dtype=torch.bfloat16, enabled=False)
    )
    with torch.inference_mode(), amp_ctx:
        state = processor.set_image(image)
        state = processor.set_text_prompt(text, state=state)

    masks = state.get("masks")
    scores = state.get("scores")
    boxes = state.get("boxes")
    if masks is None or len(masks) == 0:
        return {"masks": [], "scores": [], "boxes": []}
    if isinstance(masks, torch.Tensor):
        masks = masks.detach().to(torch.float32).cpu().numpy()
    if isinstance(scores, torch.Tensor):
        scores = scores.detach().to(torch.float32).cpu().numpy()
    if isinstance(boxes, torch.Tensor):
        boxes = boxes.detach().to(torch.float32).cpu().numpy()
    masks = np.asarray(masks).astype(np.uint8)
    if masks.ndim == 4 and masks.shape[1] == 1:
        masks = masks[:, 0]
    if masks.ndim == 2:
        masks = masks[None]
    return {
        "masks": [m for m in masks],
        "scores": [float(s) for s in np.asarray(scores).tolist()],
        "boxes": np.asarray(boxes).tolist() if boxes is not None else [],
    }


# ---- endpoint inference --------------------------------------------------


def endpoint_text(
    image_bytes: bytes,
    text: str,
    confidence: float,
    endpoint: str,
    region: str,
) -> dict:
    """Invoke the deployed endpoint with the same JSON payload as inference.py expects."""
    import boto3
    from botocore.config import Config

    smr = boto3.client(
        "sagemaker-runtime",
        region_name=region,
        config=Config(read_timeout=300, retries={"max_attempts": 3}),
    )
    body = {
        "mode": "text",
        "image_b64": base64.b64encode(image_bytes).decode("ascii"),
        "text": text,
        "confidence": confidence,
    }
    print(f"[remote] invoking {endpoint}")
    resp = smr.invoke_endpoint(
        EndpointName=endpoint,
        ContentType="application/json",
        Body=json.dumps(body),
    )
    out = json.loads(resp["Body"].read())
    return {
        "masks": [rle_to_mask(rle) for rle in out["masks_rle"]],
        "scores": out.get("scores", []),
        "boxes": out.get("boxes", []),
    }


# ---- pair up local/remote masks -------------------------------------------


def hungarian_pair(a_masks: list[np.ndarray], b_masks: list[np.ndarray]) -> list[tuple[int, int, float]]:
    """Greedy IoU-max matching. Returns [(a_idx, b_idx, iou), ...].
    Unmatched a's paired with -1, unmatched b's with -1."""
    if not a_masks and not b_masks:
        return []
    ious = np.zeros((len(a_masks), len(b_masks)), dtype=np.float32)
    for i, ma in enumerate(a_masks):
        for j, mb in enumerate(b_masks):
            if ma.shape != mb.shape:
                # Reshape mismatch — very unusual but handle gracefully.
                ious[i, j] = 0.0
                continue
            ious[i, j] = iou(ma, mb)

    pairs: list[tuple[int, int, float]] = []
    used_a, used_b = set(), set()
    # Greedy: pick highest-IoU pair, repeat.
    while True:
        remaining = [
            (i, j, ious[i, j])
            for i in range(len(a_masks))
            for j in range(len(b_masks))
            if i not in used_a and j not in used_b
        ]
        if not remaining:
            break
        i, j, v = max(remaining, key=lambda t: t[2])
        if v < 0.05:
            break
        pairs.append((i, j, v))
        used_a.add(i)
        used_b.add(j)
    # Add leftover unmatched
    for i in range(len(a_masks)):
        if i not in used_a:
            pairs.append((i, -1, 0.0))
    for j in range(len(b_masks)):
        if j not in used_b:
            pairs.append((-1, j, 0.0))
    return pairs


# ---- visualization --------------------------------------------------------


def render(
    image: np.ndarray,
    local: dict,
    remote: dict,
    pairs: list[tuple[int, int, float]],
    out_dir: Path,
) -> None:
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)

    def overlay(img, mask, color):
        vis = img.copy()
        m = mask.astype(bool)
        vis[m] = (0.5 * vis[m] + 0.5 * np.array(color)).astype(np.uint8)
        return vis

    # Single figure per pair: local vs remote side-by-side + IoU header.
    for idx, (li, ri, val) in enumerate(pairs):
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        if li >= 0:
            m = local["masks"][li]
            axes[0].imshow(overlay(image, m, [255, 60, 60]))
            axes[0].set_title(f"local  score={local['scores'][li]:.3f}")
        else:
            axes[0].imshow(image)
            axes[0].set_title("local (no match)")
        if ri >= 0:
            m = remote["masks"][ri]
            axes[1].imshow(overlay(image, m, [60, 60, 255]))
            axes[1].set_title(f"remote score={remote['scores'][ri]:.3f}")
        else:
            axes[1].imshow(image)
            axes[1].set_title("remote (no match)")
        for ax in axes:
            ax.axis("off")
        fig.suptitle(f"pair #{idx}   IoU(local, remote) = {val:.3f}", fontsize=13)
        fig.tight_layout()
        fig.savefig(out_dir / f"pair_{idx:02d}.png", dpi=120)
        plt.close(fig)


# ---- main -----------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True,
                   help="Local merged checkpoint (same one baked into the endpoint tarball).")
    p.add_argument("--endpoint", required=True, help="Deployed SageMaker endpoint name.")
    p.add_argument("--image", type=Path, required=True)
    p.add_argument("--text", required=True, help="Noun-phrase prompt.")
    p.add_argument("--confidence", type=float, default=0.5)
    p.add_argument("--bpe-path", type=Path,
                   default=Path("sam3/assets/bpe_simple_vocab_16e6.txt.gz"))
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--region", default=os.environ.get("AWS_REGION", "us-west-2"))
    p.add_argument("--output", type=Path, default=Path("runs/eval/compare"))
    args = p.parse_args()

    image = Image.open(args.image).convert("RGB")
    image_np = np.array(image)
    image_bytes = args.image.read_bytes()

    print(f"[compare] image {args.image} → {image.size}  text={args.text!r}  conf={args.confidence}")

    local = local_text(image, args.text, args.confidence, args.checkpoint, args.bpe_path, args.device)
    remote = endpoint_text(image_bytes, args.text, args.confidence, args.endpoint, args.region)

    print(f"\n[local]  {len(local['masks'])} masks   scores={[round(s,3) for s in local['scores']]}")
    print(f"[remote] {len(remote['masks'])} masks   scores={[round(s,3) for s in remote['scores']]}")

    pairs = hungarian_pair(local["masks"], remote["masks"])
    print("\nPairing (local_idx, remote_idx, IoU):")
    for li, ri, v in pairs:
        print(f"  {li:>3}  {ri:>3}  {v:.4f}")

    render(image_np, local, remote, pairs, args.output)
    print(f"\nrendered side-by-side PNGs → {args.output}/")

    # Also dump a machine-readable summary. Cast pair IoU to float —
    # hungarian_pair returns them as numpy.float32 which stdlib json
    # can't encode. Scores are already plain floats.
    summary = {
        "image": str(args.image),
        "text": args.text,
        "confidence": args.confidence,
        "local":  {"n_masks": len(local["masks"]),  "scores": [float(s) for s in local["scores"]]},
        "remote": {"n_masks": len(remote["masks"]), "scores": [float(s) for s in remote["scores"]]},
        "pairs":  [{"local": int(li), "remote": int(ri), "iou": float(v)} for (li, ri, v) in pairs],
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"summary → {args.output}/summary.json")


if __name__ == "__main__":
    main()
