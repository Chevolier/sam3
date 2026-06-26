#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved
"""Evaluate SAM3 (pretrained or fine-tuned) on AWS_SAM with interactive metrics.

Metrics:
  - mIoU @ 0 / 1 / 3 clicks
  - Boundary IoU @ 0 / 1 / 3 clicks  (Cheng et al. 2021, dilation in pixels =
    max(2, round(d_ratio * diag(image)))
  - NoC@95: mean number of clicks needed to reach IoU >= 0.95, capped at 20
            (instances that fail to reach 0.95 are counted as the cap value).

Click protocol (standard for interactive segmentation):
  - 0-click: pure text prompt on the category name. The mask among returned
    instances with highest IoU vs. the GT polygon is taken. (Skipped if
    --skip-zero-click.)
  - 1-click: a positive click at the geometric center of the largest connected
    component of the GT mask (computed via distance transform peak).
  - >1 clicks: at each step, locate the largest erroneous region (FN if
    pred misses GT, FP if pred over-segments) and place a click of the
    corresponding label at its distance-transform peak.
  - Each step's prediction conditions on the previous low-res mask logits
    (mask_input) plus all accumulated points, with multimask_output=False
    after the first click (per the SAM-1 task example notebook).

Usage:
  # Pretrained model (default — pulls facebook/sam3 from HF)
  python scripts/finetune/eval/evaluate_interactive.py \
      --coco data/AWS_SAM_split/test.json \
      --image-root data/AWS_SAM \
      --output runs/eval/pretrained.json

  # Fine-tuned model from a local checkpoint
  python scripts/finetune/eval/evaluate_interactive.py \
      --coco data/AWS_SAM_split/test.json \
      --image-root data/AWS_SAM \
      --checkpoint runs/aws_sam_finetune/checkpoints/checkpoint.pt \
      --output runs/eval/finetuned.json
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from PIL import Image
from pycocotools import mask as mask_utils

# --- model imports happen inside main() so --help works without [train] deps ---


# ---------------------------------------------------------------------------
# Mask + click utilities (no model dependency)
# ---------------------------------------------------------------------------


def polygons_to_mask(polys: list[list[float]], h: int, w: int) -> np.ndarray:
    """COCO polygon list (one [x1,y1,x2,y2,...] per ring) -> uint8 HxW mask."""
    rles = mask_utils.frPyObjects(polys, h, w)
    rle = mask_utils.merge(rles)
    return mask_utils.decode(rle).astype(np.uint8)


def iou(pred: np.ndarray, gt: np.ndarray) -> float:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    if union == 0:
        return 1.0 if inter == 0 else 0.0
    return float(inter) / float(union)


def boundary_iou(pred: np.ndarray, gt: np.ndarray, d_ratio: float = 0.02) -> float:
    """Boundary IoU (Cheng et al. 2021): IoU restricted to a band along the
    boundary, dilation radius = max(2, round(d_ratio * diag(image)))."""
    import cv2

    pred = pred.astype(np.uint8)
    gt = gt.astype(np.uint8)
    if pred.shape != gt.shape:
        raise ValueError(f"shape mismatch {pred.shape} vs {gt.shape}")
    h, w = pred.shape
    d = max(2, int(round(d_ratio * float(np.hypot(h, w)))))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    pred_eroded = cv2.erode(pred, kernel, iterations=d)
    gt_eroded = cv2.erode(gt, kernel, iterations=d)
    pred_band = pred & ~pred_eroded
    gt_band = gt & ~gt_eroded
    inter = np.logical_and(pred_band, gt_band).sum()
    union = np.logical_or(pred_band, gt_band).sum()
    if union == 0:
        return 1.0 if inter == 0 else 0.0
    return float(inter) / float(union)


def distance_transform_peak(binary_mask: np.ndarray) -> tuple[int, int] | None:
    """(x, y) of the pixel furthest from the boundary of `binary_mask`.
    Returns None for empty masks."""
    import cv2

    m = binary_mask.astype(np.uint8)
    if m.sum() == 0:
        return None
    dt = cv2.distanceTransform(m, cv2.DIST_L2, 5)
    flat_idx = int(np.argmax(dt))
    y, x = divmod(flat_idx, m.shape[1])
    return x, y


def next_correction_click(
    pred: np.ndarray, gt: np.ndarray
) -> tuple[int, int, int] | None:
    """Pick the next click for an interactive-seg simulator.

    Returns (x, y, label) where label=1 for a positive click (on a missed
    region) and 0 for a negative click (on an over-predicted region). The
    click is placed at the distance-transform peak of the larger error
    region. None if pred already matches gt exactly."""
    pred_b = pred.astype(bool)
    gt_b = gt.astype(bool)
    fn = np.logical_and(gt_b, np.logical_not(pred_b))
    fp = np.logical_and(np.logical_not(gt_b), pred_b)
    fn_count = int(fn.sum())
    fp_count = int(fp.sum())
    if fn_count == 0 and fp_count == 0:
        return None
    if fn_count >= fp_count:
        peak = distance_transform_peak(fn.astype(np.uint8))
        if peak is None:
            return None
        return peak[0], peak[1], 1
    else:
        peak = distance_transform_peak(fp.astype(np.uint8))
        if peak is None:
            return None
        return peak[0], peak[1], 0


# ---------------------------------------------------------------------------
# Per-instance evaluation
# ---------------------------------------------------------------------------


def text_prompt_zero_click(
    processor,
    image: Image.Image,
    text: str,
    gt_mask: np.ndarray,
) -> np.ndarray:
    """Run a text-only prompt and return the candidate mask with the highest
    IoU vs gt_mask. Returns an all-zeros mask if no candidates are returned."""
    state = processor.set_image(image)
    state = processor.set_text_prompt(text, state=state)
    masks = state.get("masks")
    if masks is None or len(masks) == 0:
        return np.zeros_like(gt_mask, dtype=np.uint8)
    if isinstance(masks, torch.Tensor):
        # Cast to fp32 first — autocast may return bf16, which numpy can't handle.
        masks = masks.detach().to(torch.float32).cpu().numpy()
    masks = np.asarray(masks).astype(np.uint8)
    # Sam3Processor.set_text_prompt returns masks as (N, 1, H, W); squeeze
    # the channel dim so each candidate is (H, W) and matches gt_mask.
    if masks.ndim == 4 and masks.shape[1] == 1:
        masks = masks[:, 0]
    if masks.ndim == 2:
        masks = masks[None]
    # pick the candidate with highest IoU vs GT
    best_iou = -1.0
    best = np.zeros_like(gt_mask, dtype=np.uint8)
    for m in masks:
        if m.shape != gt_mask.shape:
            # set_text_prompt returns masks at original image size; size mismatch
            # is unexpected but guard anyway.
            continue
        v = iou(m, gt_mask)
        if v > best_iou:
            best_iou = v
            best = m
    return best


def predict_with_clicks(
    model,
    inference_state,
    points: np.ndarray,
    labels: np.ndarray,
    prev_logits: np.ndarray | None,
    multimask: bool,
):
    """Wrapper around model.predict_inst → returns (best_mask uint8 HxW,
    best_score, best_logit 256x256)."""
    masks, scores, logits = model.predict_inst(
        inference_state,
        point_coords=points,
        point_labels=labels,
        mask_input=prev_logits,
        multimask_output=multimask,
    )
    if masks.ndim == 2:  # multimask_output=False returns (1,H,W) but defensive
        masks = masks[None]
        scores = np.array([scores]) if np.ndim(scores) == 0 else scores
        logits = logits[None]
    best_idx = int(np.argmax(scores))
    return (
        masks[best_idx].astype(np.uint8),
        float(scores[best_idx]),
        logits[best_idx],
    )


def evaluate_one_instance(
    model,
    processor,
    image: Image.Image,
    gt_mask: np.ndarray,
    category_name: str,
    max_clicks: int,
    iou_target: float,
    skip_zero_click: bool,
) -> dict:
    """Returns {'iou_at_clicks': {0,1,3,...}, 'biou_at_clicks': {...},
    'noc_to_target': int}."""
    iou_at = {}
    biou_at = {}

    # 0-click (text-only) — optional, expensive on big category sets.
    if not skip_zero_click:
        zc_mask = text_prompt_zero_click(processor, image, category_name, gt_mask)
        iou_at[0] = iou(zc_mask, gt_mask)
        biou_at[0] = boundary_iou(zc_mask, gt_mask)

    # Click-based prediction needs the interactive predictor's image embedding.
    inference_state = processor.set_image(image)

    # --- click 1 — center of GT, multimask_output=True, take best by score ---
    seed = distance_transform_peak(gt_mask)
    if seed is None:
        # GT empty (shouldn't happen, but guard).
        for k in (1, 3):
            iou_at[k] = 0.0
            biou_at[k] = 0.0
        return {
            "iou_at_clicks": iou_at,
            "biou_at_clicks": biou_at,
            "noc_to_target": max_clicks,
        }
    points = np.array([[seed[0], seed[1]]], dtype=np.float32)
    labels = np.array([1], dtype=np.int64)

    pred, _score, logit = predict_with_clicks(
        model, inference_state, points, labels, prev_logits=None, multimask=True
    )
    cur_iou = iou(pred, gt_mask)
    iou_at[1] = cur_iou
    biou_at[1] = boundary_iou(pred, gt_mask)
    noc_to_target = max_clicks
    if cur_iou >= iou_target:
        noc_to_target = 1

    # --- subsequent clicks — corrections on worst-error region ---
    for n_clicks in range(2, max_clicks + 1):
        cc = next_correction_click(pred, gt_mask)
        if cc is None:
            # Already matches.
            for k in (3,):
                if k not in iou_at:
                    iou_at[k] = cur_iou
                    biou_at[k] = boundary_iou(pred, gt_mask)
            if noc_to_target == max_clicks and cur_iou >= iou_target:
                noc_to_target = n_clicks - 1
            break
        x, y, lab = cc
        points = np.concatenate([points, [[x, y]]], axis=0).astype(np.float32)
        labels = np.concatenate([labels, [lab]], axis=0).astype(np.int64)
        pred, _score, logit = predict_with_clicks(
            model,
            inference_state,
            points,
            labels,
            prev_logits=logit[None] if logit.ndim == 2 else logit,
            multimask=False,
        )
        cur_iou = iou(pred, gt_mask)
        if n_clicks == 3:
            iou_at[3] = cur_iou
            biou_at[3] = boundary_iou(pred, gt_mask)
        if cur_iou >= iou_target and noc_to_target == max_clicks:
            noc_to_target = n_clicks
            # Continue ONLY if we still need to record IoU @ 3 clicks.
            if 3 in iou_at:
                break

    # If the loop never reached n_clicks==3 (e.g. converged at 2), backfill.
    if 3 not in iou_at:
        iou_at[3] = cur_iou
        biou_at[3] = boundary_iou(pred, gt_mask)

    return {
        "iou_at_clicks": iou_at,
        "biou_at_clicks": biou_at,
        "noc_to_target": noc_to_target,
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def iter_instances(
    coco: dict,
    image_root: Path,
    min_area_px: int,
    max_per_image: int | None,
) -> Iterable[tuple[int, dict, np.ndarray, str, Image.Image]]:
    cat_id_to_name = {c["id"]: c["name"] for c in coco["categories"]}
    images_by_id = {im["id"]: im for im in coco["images"]}
    anns_by_image: dict[int, list[dict]] = {}
    for ann in coco["annotations"]:
        anns_by_image.setdefault(ann["image_id"], []).append(ann)

    for image_id, anns in anns_by_image.items():
        im_info = images_by_id[image_id]
        img_path = image_root / im_info["file_name"]
        if not img_path.exists():
            continue
        try:
            pil = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"[eval] skipping {img_path}: {e}")
            continue
        h, w = im_info["height"], im_info["width"]
        kept = 0
        for ann in anns:
            if ann.get("iscrowd"):
                continue
            seg = ann.get("segmentation")
            if not seg:
                continue
            try:
                gt = polygons_to_mask(seg, h, w)
            except Exception:
                continue
            if int(gt.sum()) < min_area_px:
                continue
            yield (
                image_id,
                ann,
                gt,
                cat_id_to_name[ann["category_id"]],
                pil,
            )
            kept += 1
            if max_per_image is not None and kept >= max_per_image:
                break


def aggregate(
    rows: list[dict], iou_target: float, max_clicks: int
) -> dict:
    if not rows:
        return {"n_instances": 0}
    by_clicks_iou = {}
    by_clicks_biou = {}
    for k in (0, 1, 3):
        vals_iou = [r["iou_at_clicks"].get(k) for r in rows]
        vals_iou = [v for v in vals_iou if v is not None]
        vals_biou = [r["biou_at_clicks"].get(k) for r in rows]
        vals_biou = [v for v in vals_biou if v is not None]
        by_clicks_iou[str(k)] = float(np.mean(vals_iou)) if vals_iou else None
        by_clicks_biou[str(k)] = float(np.mean(vals_biou)) if vals_biou else None
    nocs = [r["noc_to_target"] for r in rows]
    reached = sum(1 for r in rows if r["noc_to_target"] < max_clicks)
    return {
        "n_instances": len(rows),
        "mIoU_at_clicks": by_clicks_iou,
        "BoundaryIoU_at_clicks": by_clicks_biou,
        f"NoC@{int(iou_target * 100)}": float(np.mean(nocs)),
        f"reach_rate_at_{int(iou_target * 100)}": reached / len(rows),
        "noc_cap": max_clicks,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coco", type=Path, default=Path("data/AWS_SAM_split/test.json"))
    parser.add_argument("--image-root", type=Path, default=Path("data/AWS_SAM"))
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Path to a fine-tuned SAM3 checkpoint. Omit to use facebook/sam3 from HF.",
    )
    parser.add_argument(
        "--pretrained-fallback",
        type=Path,
        default=None,
        help="Path to the pretrained SAM3 checkpoint to use as a fallback for "
        "submodules the fine-tuned checkpoint didn't include (e.g. "
        "sam2_convs, inst_interactive_predictor, tracker). The trainer only "
        "saves modules that were built during training, so the click "
        "predictor stays at random init unless we fill it from pretrained.",
    )
    parser.add_argument("--output", type=Path, default=Path("runs/eval/results.json"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--bpe-path", default="sam3/assets/bpe_simple_vocab_16e6.txt.gz")
    parser.add_argument(
        "--max-instances",
        type=int,
        default=None,
        help="Total cap across the whole eval set (for quick smoke-testing).",
    )
    parser.add_argument(
        "--max-per-image", type=int, default=None,
        help="Cap on instances sampled per image (default: all).",
    )
    parser.add_argument(
        "--min-area",
        type=int,
        default=200,
        help="Skip GT instances smaller than this many pixels.",
    )
    parser.add_argument("--max-clicks", type=int, default=20, help="NoC cap.")
    parser.add_argument("--iou-target", type=float, default=0.95)
    parser.add_argument(
        "--skip-zero-click",
        action="store_true",
        help="Skip the text-prompt 0-click path (useful when category names "
        "don't disambiguate well or you only care about click metrics).",
    )
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Late imports so --help works without train extras.
    from sam3 import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    # When a fine-tuned checkpoint is given, pre-load the pretrained weights
    # first so submodules the trainer never built (sam2_convs,
    # inst_interactive_predictor, tracker) start from sensible weights
    # rather than random init. Then layer the fine-tuned dict on top.
    initial_ckpt = args.pretrained_fallback if (args.checkpoint and args.pretrained_fallback) else args.checkpoint
    print(f"[eval] building model (initial_ckpt={initial_ckpt})")
    model = build_sam3_image_model(
        bpe_path=str(args.bpe_path),
        device=args.device,
        eval_mode=True,
        enable_segmentation=True,
        enable_inst_interactivity=True,
        checkpoint_path=str(initial_ckpt) if initial_ckpt else None,
        load_from_HF=initial_ckpt is None,
    )

    # Overlay fine-tuned weights on top of the pretrained-initialized model.
    if args.checkpoint and args.pretrained_fallback:
        print(f"[eval] overlaying fine-tuned weights from {args.checkpoint}")
        ft_ckpt = torch.load(str(args.checkpoint), map_location="cpu", weights_only=False)
        if isinstance(ft_ckpt, dict) and "model" in ft_ckpt and isinstance(ft_ckpt["model"], dict):
            ft_state = ft_ckpt["model"]
        else:
            ft_state = ft_ckpt
        # Make sure the keys are unprefixed (trainer-format usually are).
        if any(k.startswith("detector.") for k in ft_state):
            ft_state = {
                k.replace("detector.", ""): v for k, v in ft_state.items() if k.startswith("detector.")
            }
        # Move tensors to the model's device before load to avoid host->device
        # copies during load_state_dict (esp. on multi-GPU setups).
        target_device = next(model.parameters()).device
        ft_state = {k: v.to(target_device) if hasattr(v, "to") else v for k, v in ft_state.items()}
        missing, unexpected = model.load_state_dict(ft_state, strict=False)
        print(
            f"[eval] overlay: loaded={len(ft_state) - len(unexpected)}, "
            f"missing={len(missing)} (kept from pretrained), "
            f"unexpected={len(unexpected)}"
        )

    processor = Sam3Processor(model, device=args.device)

    # SAM3's fused MLP kernels (perflib.fused.addmm_act, in vitdet.Mlp.forward)
    # cast inputs to bf16 internally, so the surrounding tensors must be bf16
    # too. Wrap every forward pass in this context to avoid the
    # "mat1 and mat2 must have the same dtype, but got BFloat16 and Float"
    # crash that surfaces in some code paths.
    amp_ctx = (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if str(args.device).startswith("cuda")
        else torch.autocast("cpu", dtype=torch.bfloat16, enabled=False)
    )

    with open(args.coco, "r") as f:
        coco = json.load(f)

    rows: list[dict] = []
    t0 = time.time()
    n_seen = 0
    for image_id, ann, gt, cat_name, pil in iter_instances(
        coco,
        args.image_root,
        args.min_area,
        args.max_per_image,
    ):
        if args.max_instances is not None and len(rows) >= args.max_instances:
            break
        n_seen += 1
        try:
            with torch.inference_mode(), amp_ctx:
                row = evaluate_one_instance(
                    model,
                    processor,
                    pil,
                    gt,
                    cat_name,
                    max_clicks=args.max_clicks,
                    iou_target=args.iou_target,
                    skip_zero_click=args.skip_zero_click,
                )
            row["image_id"] = image_id
            row["ann_id"] = ann["id"]
            row["category"] = cat_name
            rows.append(row)
        except Exception as e:
            print(f"[eval] instance {ann['id']} failed: {e}")
            continue

        if len(rows) % 25 == 0:
            elapsed = time.time() - t0
            ips = len(rows) / max(elapsed, 1e-6)
            print(
                f"[eval] {len(rows)} instances done "
                f"({ips:.2f} inst/s, last cat={cat_name})"
            )

    summary = aggregate(rows, args.iou_target, args.max_clicks)
    summary["checkpoint"] = str(args.checkpoint) if args.checkpoint else "facebook/sam3"
    summary["coco"] = str(args.coco)
    summary["wall_time_sec"] = time.time() - t0

    out = {"summary": summary, "instances": rows}
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)

    print("\n=== Summary ===")
    print(json.dumps(summary, indent=2))
    print(f"[eval] wrote {args.output}")


if __name__ == "__main__":
    main()
