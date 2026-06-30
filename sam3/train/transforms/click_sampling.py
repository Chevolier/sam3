# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved
# pyre-unsafe
"""Synthetic click sampling for SAM-style interactive training.

These are pure-tensor utilities — they live separately from
`point_sampling.py` (which targets DETR-style geometric query inputs at
data-loading time). The click utilities here are called from inside
`Sam3Image.forward` during training, because correction clicks depend on
the model's prediction at the previous step.

Two utilities:

- `sample_seed_clicks(gt_masks)`: one positive click per GT instance at
  the most-interior pixel (distance-transform peak). Pure GT-driven, no
  model dependency.

- `sample_correction_click(pred_mask, gt_mask)`: given the model's
  prediction for one instance and the GT, pick the next (x, y, label)
  to feed back as a correction click. label=1 if pred missed GT (FN
  dominant); label=0 if pred over-segmented (FP dominant).

These mirror the protocol in `scripts/finetune/eval/evaluate_interactive.py`
so training-time and evaluation-time click distributions match.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch


def _distance_transform_peak_torch(mask: torch.Tensor) -> Tuple[int, int]:
    """Return (x, y) of the pixel furthest from any boundary of `mask`.

    `mask`: (H, W) bool / 0-1 tensor. Returns (0, 0) for empty masks
    rather than None so callers don't have to special-case empties.

    Uses the CV2 distance transform via numpy round-trip — fast enough
    for training where each instance is sampled once per step, and we
    don't need to differentiate through point coordinates anyway.
    """
    import cv2

    m = mask.detach().to(torch.uint8).cpu().numpy()
    if m.sum() == 0:
        return 0, 0
    dt = cv2.distanceTransform(m, cv2.DIST_L2, 5)
    flat_idx = int(np.argmax(dt))
    h, w = m.shape
    y, x = divmod(flat_idx, w)
    return x, y


def sample_seed_clicks(gt_masks: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sample one positive seed click per GT instance.

    Args:
        gt_masks: (N, H, W) bool/uint8 tensor of GT instance masks.

    Returns:
        points: (N, 1, 2) float32 xy in pixel coords.
        labels: (N, 1) int64 — all ones (positive).
    """
    n = gt_masks.shape[0]
    pts = torch.zeros((n, 1, 2), dtype=torch.float32, device=gt_masks.device)
    labels = torch.ones((n, 1), dtype=torch.int64, device=gt_masks.device)
    for i in range(n):
        x, y = _distance_transform_peak_torch(gt_masks[i])
        pts[i, 0, 0] = x
        pts[i, 0, 1] = y
    return pts, labels


def sample_correction_click(
    pred_mask: torch.Tensor, gt_mask: torch.Tensor
) -> Optional[Tuple[int, int, int]]:
    """Pick the next click for an interactive-seg simulator.

    Args:
        pred_mask: (H, W) binary current prediction.
        gt_mask:   (H, W) binary ground truth.

    Returns:
        (x, y, label) where label=1 if FN > FP (positive correction on
        missed region), 0 if FP > FN (negative correction on over-pred).
        None if pred matches GT exactly.
    """
    p = pred_mask.detach().to(torch.bool)
    g = gt_mask.detach().to(torch.bool)
    fn = g & ~p
    fp = p & ~g
    fn_n = int(fn.sum())
    fp_n = int(fp.sum())
    if fn_n == 0 and fp_n == 0:
        return None
    if fn_n >= fp_n:
        x, y = _distance_transform_peak_torch(fn)
        return x, y, 1
    else:
        x, y = _distance_transform_peak_torch(fp)
        return x, y, 0


def sample_correction_clicks_batch(
    pred_masks: torch.Tensor, gt_masks: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Vectorized wrapper that runs `sample_correction_click` per instance.

    Args:
        pred_masks: (N, H, W) binary predictions.
        gt_masks:   (N, H, W) binary GT masks.

    Returns:
        points: (N, 1, 2) float32 — xy, with (0, 0) for instances where
            no correction was needed (caller can mask these via the
            `valid` flag).
        labels: (N, 1) int64 — 1 or 0; default 1 for invalid slots.
        valid:  (N,) bool — True if a correction was actually selected.
    """
    n = pred_masks.shape[0]
    pts = torch.zeros((n, 1, 2), dtype=torch.float32, device=pred_masks.device)
    labels = torch.ones((n, 1), dtype=torch.int64, device=pred_masks.device)
    valid = torch.zeros((n,), dtype=torch.bool, device=pred_masks.device)
    for i in range(n):
        cc = sample_correction_click(pred_masks[i], gt_masks[i])
        if cc is None:
            continue
        x, y, lab = cc
        pts[i, 0, 0] = x
        pts[i, 0, 1] = y
        labels[i, 0] = lab
        valid[i] = True
    return pts, labels, valid
