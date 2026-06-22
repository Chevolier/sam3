# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved
"""SageMaker PyTorch container handler for SAM3.

The deploy.py driver packages this file under `code/` inside the model
tarball. SageMaker's PyTorch inference container imports this module and
calls model_fn / input_fn / predict_fn / output_fn.

Request schemas (JSON, content-type=application/json):

  Text prompt:
    {
      "mode": "text",
      "image_b64": "<base64 PNG/JPEG>",
      "text": "obstacle",
      "confidence": 0.5     # optional, default 0.5
    }

  Click prompt (SAM-1-style interactive):
    {
      "mode": "click",
      "image_b64": "<base64>",
      "points": [[x, y], [x, y], ...],   # pixel coords
      "labels": [1, 0, 1, ...],          # 1=foreground, 0=background
      "multimask_output": true,          # optional, default true
      "box": [x1, y1, x2, y2]            # optional, XYXY pixel
    }

Response (JSON):
  {
    "masks_rle":  [<COCO RLE>, ...],     # one per returned mask
    "scores":     [...],
    "boxes":      [[x1,y1,x2,y2], ...],  # text mode only
    "image_size": [H, W]
  }

The masks are returned as COCO-RLE strings to keep payloads small;
clients can decode with `pycocotools.mask.decode(rle)`.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import os
from typing import Any

import numpy as np
import torch
from PIL import Image
from pycocotools import mask as mask_utils

LOG = logging.getLogger("sam3.inference")
LOG.setLevel(logging.INFO)


# ---------------------------------------------------------------------------
# SageMaker hooks
# ---------------------------------------------------------------------------


def model_fn(model_dir: str) -> dict:
    """Loaded once per worker. SageMaker passes the extracted model dir."""
    from sam3 import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    bpe_path = _locate(
        model_dir,
        candidates=[
            "bpe_simple_vocab_16e6.txt.gz",
            "assets/bpe_simple_vocab_16e6.txt.gz",
        ],
    )
    if bpe_path is None:
        # Fall back to the one shipped inside the installed sam3 package.
        import sam3 as _sam3

        bpe_path = os.path.join(
            os.path.dirname(_sam3.__file__),
            "assets/bpe_simple_vocab_16e6.txt.gz",
        )

    ckpt = _locate(model_dir, candidates=["checkpoint.pt", "model.pt", "sam3.pt"])
    LOG.info(
        "model_fn: device=%s, bpe=%s, ckpt=%s, model_dir=%s",
        device, bpe_path, ckpt, model_dir,
    )
    model = build_sam3_image_model(
        bpe_path=bpe_path,
        device=device,
        eval_mode=True,
        enable_segmentation=True,
        enable_inst_interactivity=True,
        checkpoint_path=ckpt,
        load_from_HF=ckpt is None,
    )
    processor = Sam3Processor(model, device=device)
    return {"model": model, "processor": processor, "device": device}


def input_fn(request_body: bytes | str, request_content_type: str) -> dict:
    if request_content_type not in ("application/json", "text/json"):
        raise ValueError(f"Unsupported content type {request_content_type}")
    if isinstance(request_body, (bytes, bytearray)):
        request_body = request_body.decode("utf-8")
    return json.loads(request_body)


def predict_fn(payload: dict, ctx: dict) -> dict:
    model = ctx["model"]
    processor = ctx["processor"]

    image = _decode_image(payload["image_b64"])
    mode = payload.get("mode", "text")
    H, W = image.size[1], image.size[0]

    if mode == "text":
        return _predict_text(processor, image, payload, (H, W))
    if mode == "click":
        return _predict_click(model, processor, image, payload, (H, W))
    raise ValueError(f"unknown mode={mode!r}")


def output_fn(prediction: dict, accept: str) -> tuple[str, str]:
    if accept not in ("application/json", "text/json", "*/*"):
        raise ValueError(f"Unsupported accept {accept}")
    return json.dumps(prediction), "application/json"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _locate(model_dir: str, candidates: list[str]) -> str | None:
    for c in candidates:
        p = os.path.join(model_dir, c)
        if os.path.exists(p):
            return p
    return None


def _decode_image(b64: str) -> Image.Image:
    data = base64.b64decode(b64)
    return Image.open(io.BytesIO(data)).convert("RGB")


def _mask_to_rle(m: np.ndarray) -> dict:
    rle = mask_utils.encode(np.asfortranarray(m.astype(np.uint8)))
    rle["counts"] = rle["counts"].decode("ascii")
    return rle


def _predict_text(processor, image: Image.Image, payload: dict, hw: tuple[int, int]) -> dict:
    confidence = float(payload.get("confidence", 0.5))
    processor.confidence_threshold = confidence
    state = processor.set_image(image)
    state = processor.set_text_prompt(payload["text"], state=state)
    masks = state.get("masks")
    scores = state.get("scores")
    boxes = state.get("boxes")
    if masks is None or len(masks) == 0:
        return {
            "masks_rle": [],
            "scores": [],
            "boxes": [],
            "image_size": list(hw),
        }
    if isinstance(masks, torch.Tensor):
        masks = masks.detach().cpu().numpy()
    if isinstance(scores, torch.Tensor):
        scores = scores.detach().cpu().numpy()
    if isinstance(boxes, torch.Tensor):
        boxes = boxes.detach().cpu().numpy()
    masks = np.asarray(masks).astype(np.uint8)
    if masks.ndim == 2:
        masks = masks[None]
    return {
        "masks_rle": [_mask_to_rle(m) for m in masks],
        "scores": [float(s) for s in np.asarray(scores).tolist()] if scores is not None else [],
        "boxes": np.asarray(boxes).tolist() if boxes is not None else [],
        "image_size": list(hw),
    }


def _predict_click(model, processor, image: Image.Image, payload: dict, hw: tuple[int, int]) -> dict:
    points = np.asarray(payload.get("points", []), dtype=np.float32)
    labels = np.asarray(payload.get("labels", []), dtype=np.int64)
    box = payload.get("box")
    box = np.asarray(box, dtype=np.float32) if box else None
    multimask = bool(payload.get("multimask_output", True))

    state = processor.set_image(image)
    masks, scores, _ = model.predict_inst(
        state,
        point_coords=points if len(points) else None,
        point_labels=labels if len(labels) else None,
        box=box,
        multimask_output=multimask,
    )
    if masks.ndim == 2:
        masks = masks[None]
        scores = np.array([scores]) if np.ndim(scores) == 0 else scores
    return {
        "masks_rle": [_mask_to_rle(m) for m in masks.astype(np.uint8)],
        "scores": [float(s) for s in np.asarray(scores).tolist()],
        "boxes": [],
        "image_size": list(hw),
    }
