#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved
"""FastAPI server that loads both pretrained and fine-tuned SAM3 models
and serves a side-by-side comparison UI.

Both models are kept warm on the GPU so each request is just a forward
pass. For the click-prompt mode, image embeddings are cached per image
so additional clicks on the same image are cheap.

Usage:
  python scripts/finetune/compare_app/server.py \
      --pretrained-ckpt /home/ec2-user/SageMaker/efs/Models/sam3/sam3.pt \
      --finetuned-ckpt  runs/aws_sam_finetune/checkpoints/checkpoint_merged.pt \
      --port 8080

  IMPORTANT: pass the *merged* fine-tuned checkpoint (see
  scripts/finetune/merge_checkpoint.py). The raw trainer output is
  incomplete and yields catastrophically wrong masks.

Then visit http://<host>:8080/ — upload an image, enter a text prompt
or click points, and see both models' masks rendered overlaid on the
image, side by side.

For SSH port-forwarding from a remote box:
  ssh -L 8080:localhost:8080 <ec2-host>
  open http://localhost:8080
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import logging
import os
import time
import uuid
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel
from pycocotools import mask as mask_utils

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOG = logging.getLogger("sam3.compare_app")


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class UploadRequest(BaseModel):
    image_b64: str


class UploadResponse(BaseModel):
    image_id: str
    width: int
    height: int


class TextPromptRequest(BaseModel):
    image_id: str
    text: str
    confidence: float = 0.5


class ClickPromptRequest(BaseModel):
    image_id: str
    # Each click: [x, y, label] where label=1 positive, 0 negative.
    clicks: list[list[int]]
    multimask: bool = True


class ModelResult(BaseModel):
    # Each entry: base64-encoded PNG of an RGBA overlay (mask + label).
    mask_overlays: list[str]
    scores: list[float]
    n_masks: int
    elapsed_ms: float


class CompareResponse(BaseModel):
    pretrained: ModelResult
    finetuned: ModelResult


class TestImageSummary(BaseModel):
    image_index: int              # 0-indexed position in the test set
    file_name: str
    width: int
    height: int
    categories: list[str]         # distinct GT category names in this image


class TestSetListResponse(BaseModel):
    n_images: int                 # total matching the filter
    n_categories: int             # total distinct categories
    all_categories: list[str]
    items: list[TestImageSummary]


class TestImageLoadRequest(BaseModel):
    image_index: int
    # If given, render only GT instances of these categories. Empty = all.
    categories: Optional[list[str]] = None


class TestImageLoadResponse(BaseModel):
    image_id: str
    file_name: str
    width: int
    height: int
    # The image itself, base64-encoded PNG (so the UI can display it without
    # serving an extra HTTP route).
    image_b64: str
    # Per-category GT overlay (one base64 PNG per requested category).
    gt_overlays: list[str]
    gt_categories: list[str]      # names corresponding 1:1 with gt_overlays
    available_categories: list[str]   # everything this image has GT for


# ---------------------------------------------------------------------------
# Image / mask utilities
# ---------------------------------------------------------------------------


def b64_to_pil(b64: str) -> Image.Image:
    # accept both "data:image/png;base64,..." and raw base64
    if b64.startswith("data:"):
        b64 = b64.split(",", 1)[1]
    raw = base64.b64decode(b64)
    return Image.open(io.BytesIO(raw)).convert("RGB")


def pil_to_b64_png(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def color_for_index(i: int) -> tuple[int, int, int]:
    import colorsys

    hue = (i * 0.6180339887) % 1.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.75, 0.95)
    return int(r * 255), int(g * 255), int(b * 255)


def mask_to_overlay(mask: np.ndarray, color: tuple[int, int, int], alpha: int = 120) -> Image.Image:
    """Return an RGBA Image the size of the mask, with mask pixels filled
    in `color` at the given alpha, transparent elsewhere."""
    h, w = mask.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    m = mask.astype(bool)
    rgba[m, 0] = color[0]
    rgba[m, 1] = color[1]
    rgba[m, 2] = color[2]
    rgba[m, 3] = alpha
    return Image.fromarray(rgba, mode="RGBA")


# ---------------------------------------------------------------------------
# Test-set loader — exposes the COCO test split for browsing
# ---------------------------------------------------------------------------


def _polygons_to_mask(polys: list[list[float]], h: int, w: int) -> np.ndarray:
    rles = mask_utils.frPyObjects(polys, h, w)
    rle = mask_utils.merge(rles)
    return mask_utils.decode(rle).astype(np.uint8)


class TestSet:
    """Lazy-load COCO test.json + on-demand image/mask rendering."""

    def __init__(self, coco_path: Path, image_root: Path) -> None:
        if not coco_path.exists():
            raise FileNotFoundError(f"coco path not found: {coco_path}")
        if not image_root.exists():
            raise FileNotFoundError(f"image root not found: {image_root}")
        self.coco_path = coco_path
        self.image_root = image_root

        LOG.info("[testset] loading %s", coco_path)
        with open(coco_path, "r") as f:
            data = json.load(f)
        self.cat_id_to_name: dict[int, str] = {c["id"]: c["name"] for c in data["categories"]}
        self.all_categories: list[str] = sorted(self.cat_id_to_name.values())

        # Sort images by id for stable indexing.
        self._images_sorted: list[dict] = sorted(data["images"], key=lambda im: im["id"])

        # Annotations per image_id.
        self._anns_by_image: dict[int, list[dict]] = {}
        for ann in data["annotations"]:
            self._anns_by_image.setdefault(ann["image_id"], []).append(ann)

        # Per-image cached category set for fast filtering.
        self._cats_by_image: list[set[str]] = []
        for im in self._images_sorted:
            cats: set[str] = set()
            for ann in self._anns_by_image.get(im["id"], []):
                name = self.cat_id_to_name.get(ann["category_id"])
                if name:
                    cats.add(name)
            self._cats_by_image.append(cats)

        LOG.info(
            "[testset] ready: %d images, %d categories",
            len(self._images_sorted), len(self.all_categories),
        )

    def list_summaries(
        self,
        offset: int = 0,
        limit: int = 200,
        category_filter: Optional[str] = None,
    ) -> tuple[int, list[TestImageSummary]]:
        # Build the filtered index list.
        if category_filter:
            indices = [
                i for i, cats in enumerate(self._cats_by_image)
                if category_filter in cats
            ]
        else:
            indices = list(range(len(self._images_sorted)))
        n_total = len(indices)
        page = indices[offset : offset + limit]
        items = []
        for i in page:
            im = self._images_sorted[i]
            items.append(
                TestImageSummary(
                    image_index=i,
                    file_name=im["file_name"],
                    width=int(im["width"]),
                    height=int(im["height"]),
                    categories=sorted(self._cats_by_image[i]),
                )
            )
        return n_total, items

    def load_image(
        self, image_index: int, categories: Optional[list[str]] = None
    ) -> TestImageLoadResponse:
        if image_index < 0 or image_index >= len(self._images_sorted):
            raise HTTPException(404, f"image_index {image_index} out of range")
        im_info = self._images_sorted[image_index]
        path = self.image_root / im_info["file_name"]
        if not path.exists():
            raise HTTPException(404, f"image file missing: {path}")
        pil = Image.open(path).convert("RGB")
        w, h = pil.size

        # Use file content hash as the image_id so repeat loads reuse the
        # ModelHandle's embedding cache.
        with open(path, "rb") as f:
            digest = hashlib.sha1(f.read()).hexdigest()[:16]
        image_id = f"img_{digest}"

        # Build per-category union masks.
        anns = self._anns_by_image.get(im_info["id"], [])
        wanted = set(categories) if categories else None
        masks_by_cat: dict[str, np.ndarray] = {}
        for ann in anns:
            name = self.cat_id_to_name.get(ann["category_id"])
            if not name:
                continue
            if wanted is not None and name not in wanted:
                continue
            seg = ann.get("segmentation")
            if not seg:
                continue
            try:
                m = _polygons_to_mask(seg, int(im_info["height"]), int(im_info["width"]))
            except Exception as e:
                LOG.warning("[testset] mask decode failed for ann %s: %s", ann.get("id"), e)
                continue
            if name in masks_by_cat:
                masks_by_cat[name] = np.maximum(masks_by_cat[name], m)
            else:
                masks_by_cat[name] = m

        gt_overlays: list[str] = []
        gt_categories: list[str] = []
        # Stable order matching the global category list.
        for idx, name in enumerate(self.all_categories):
            if name not in masks_by_cat:
                continue
            color = color_for_index(idx)
            gt_overlays.append(pil_to_b64_png(mask_to_overlay(masks_by_cat[name], color)))
            gt_categories.append(name)

        return TestImageLoadResponse(
            image_id=image_id,
            file_name=im_info["file_name"],
            width=int(w),
            height=int(h),
            image_b64=pil_to_b64_png(pil),
            gt_overlays=gt_overlays,
            gt_categories=gt_categories,
            available_categories=sorted(self._cats_by_image[image_index]),
        ), pil


# ---------------------------------------------------------------------------
# Model wrapper — holds one SAM3 + processor and exposes (text|click) APIs
# ---------------------------------------------------------------------------


class ModelHandle:
    def __init__(self, name: str, checkpoint_path: Optional[Path], bpe_path: Path, device: str):
        from sam3 import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor

        self.name = name
        LOG.info("[%s] building model from %s ...", name, checkpoint_path or "facebook/sam3 (HF)")
        t0 = time.time()
        self.model = build_sam3_image_model(
            bpe_path=str(bpe_path),
            device=device,
            eval_mode=True,
            enable_segmentation=True,
            enable_inst_interactivity=True,
            checkpoint_path=str(checkpoint_path) if checkpoint_path else None,
            load_from_HF=checkpoint_path is None,
        )
        self.processor = Sam3Processor(self.model, device=device)
        self.device = device
        # Autocast context — SAM3's fused MLP kernels (perflib.fused.addmm_act
        # in vitdet.Mlp.forward) cast to bf16 internally, so the surrounding
        # tensors need to be bf16 too. The example notebooks set this globally
        # via torch.autocast(...).__enter__(); we use a per-call context.
        self._amp = (
            torch.autocast("cuda", dtype=torch.bfloat16)
            if device.startswith("cuda")
            else torch.autocast("cpu", dtype=torch.bfloat16, enabled=False)
        )
        # State cache, per image_id, so repeated clicks on the same image
        # don't recompute the ViT embedding.
        self._state_cache: dict[str, dict] = {}
        LOG.info("[%s] ready in %.1fs", name, time.time() - t0)

    def set_image(self, image_id: str, pil: Image.Image) -> dict:
        if image_id in self._state_cache:
            return self._state_cache[image_id]
        with self._amp:
            state = self.processor.set_image(pil)
        self._state_cache[image_id] = state
        return state

    def evict(self, image_id: str) -> None:
        self._state_cache.pop(image_id, None)

    @torch.inference_mode()
    def predict_text(self, image_id: str, pil: Image.Image, text: str, confidence: float) -> ModelResult:
        self.processor.confidence_threshold = float(confidence)
        # Text-prompt uses set_image + set_text_prompt; reuse the cached state
        # only by re-running set_text_prompt on the same state dict.
        t0 = time.time()
        state = self.set_image(image_id, pil)
        with self._amp:
            state = self.processor.set_text_prompt(text, state=state)
        elapsed = (time.time() - t0) * 1000.0

        masks = state.get("masks")
        scores = state.get("scores")
        if masks is None or len(masks) == 0:
            return ModelResult(mask_overlays=[], scores=[], n_masks=0, elapsed_ms=elapsed)
        # Cast to float32 before .numpy() — autocast may return bf16,
        # which numpy doesn't support.
        if isinstance(masks, torch.Tensor):
            masks = masks.detach().to(torch.float32).cpu().numpy()
        if isinstance(scores, torch.Tensor):
            scores = scores.detach().to(torch.float32).cpu().numpy()
        masks = np.asarray(masks).astype(np.uint8)
        # Sam3Processor returns masks as (N, 1, H, W) after the channel-dim
        # unsqueeze in set_text_prompt. Squeeze the channel dim so each
        # iteration produces a 2-D (H, W) mask.
        if masks.ndim == 4 and masks.shape[1] == 1:
            masks = masks[:, 0]
        if masks.ndim == 2:
            masks = masks[None]

        overlays = []
        for i, m in enumerate(masks):
            color = color_for_index(i)
            overlays.append(pil_to_b64_png(mask_to_overlay(m, color)))
        return ModelResult(
            mask_overlays=overlays,
            scores=[float(s) for s in np.asarray(scores).tolist()],
            n_masks=int(len(masks)),
            elapsed_ms=elapsed,
        )

    @torch.inference_mode()
    def predict_clicks(
        self,
        image_id: str,
        pil: Image.Image,
        clicks: list[list[int]],
        multimask: bool,
    ) -> ModelResult:
        if not clicks:
            return ModelResult(mask_overlays=[], scores=[], n_masks=0, elapsed_ms=0.0)
        points = np.array([[c[0], c[1]] for c in clicks], dtype=np.float32)
        labels = np.array([c[2] for c in clicks], dtype=np.int64)

        t0 = time.time()
        state = self.set_image(image_id, pil)
        with self._amp:
            masks, scores, _ = self.model.predict_inst(
                state,
                point_coords=points,
                point_labels=labels,
                multimask_output=multimask,
            )
        elapsed = (time.time() - t0) * 1000.0

        if masks.ndim == 2:
            masks = masks[None]
            scores = np.array([scores]) if np.ndim(scores) == 0 else scores

        # Sort by score so the best mask is first.
        order = np.argsort(-scores)
        masks = masks[order].astype(np.uint8)
        scores = scores[order]

        overlays = []
        for i, m in enumerate(masks):
            color = color_for_index(i)
            overlays.append(pil_to_b64_png(mask_to_overlay(m, color)))
        return ModelResult(
            mask_overlays=overlays,
            scores=[float(s) for s in scores.tolist()],
            n_masks=int(len(masks)),
            elapsed_ms=elapsed,
        )


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------


class CompareApp:
    def __init__(
        self,
        pretrained: ModelHandle,
        finetuned: ModelHandle,
        testset: Optional[TestSet] = None,
    ):
        self.pretrained = pretrained
        self.finetuned = finetuned
        self.testset = testset
        # Keep the PIL image around per image_id for re-prompting.
        self._images: dict[str, Image.Image] = {}

    def upload(self, req: UploadRequest) -> UploadResponse:
        pil = b64_to_pil(req.image_b64)
        # Use a hash of the bytes as the id so repeat uploads are cheap.
        h = hashlib.sha1(req.image_b64.encode("ascii")).hexdigest()[:16]
        image_id = f"img_{h}"
        self._images[image_id] = pil
        return UploadResponse(image_id=image_id, width=pil.size[0], height=pil.size[1])

    def text_prompt(self, req: TextPromptRequest) -> CompareResponse:
        pil = self._images.get(req.image_id)
        if pil is None:
            raise HTTPException(404, f"Unknown image_id {req.image_id}; re-upload.")
        return CompareResponse(
            pretrained=self.pretrained.predict_text(req.image_id, pil, req.text, req.confidence),
            finetuned=self.finetuned.predict_text(req.image_id, pil, req.text, req.confidence),
        )

    def click_prompt(self, req: ClickPromptRequest) -> CompareResponse:
        pil = self._images.get(req.image_id)
        if pil is None:
            raise HTTPException(404, f"Unknown image_id {req.image_id}; re-upload.")
        return CompareResponse(
            pretrained=self.pretrained.predict_clicks(req.image_id, pil, req.clicks, req.multimask),
            finetuned=self.finetuned.predict_clicks(req.image_id, pil, req.clicks, req.multimask),
        )

    def testset_list(self, offset: int, limit: int, category: Optional[str]) -> TestSetListResponse:
        if self.testset is None:
            raise HTTPException(503, "Test set not configured. Start the server with --test-coco/--test-image-root.")
        n_total, items = self.testset.list_summaries(offset, limit, category)
        return TestSetListResponse(
            n_images=n_total,
            n_categories=len(self.testset.all_categories),
            all_categories=self.testset.all_categories,
            items=items,
        )

    def testset_load(self, req: TestImageLoadRequest) -> TestImageLoadResponse:
        if self.testset is None:
            raise HTTPException(503, "Test set not configured.")
        resp, pil = self.testset.load_image(req.image_index, req.categories)
        # Cache the PIL image so subsequent /api/text and /api/click calls
        # work with the same image_id as if it had been uploaded.
        self._images[resp.image_id] = pil
        return resp


def build_app(
    pretrained: ModelHandle,
    finetuned: ModelHandle,
    static_dir: Path,
    testset: Optional[TestSet] = None,
) -> FastAPI:
    compare = CompareApp(pretrained, finetuned, testset)
    app = FastAPI(title="SAM3 Compare", version="1.0")

    @app.post("/api/upload", response_model=UploadResponse)
    def upload(req: UploadRequest) -> UploadResponse:
        return compare.upload(req)

    @app.post("/api/text", response_model=CompareResponse)
    def text(req: TextPromptRequest) -> CompareResponse:
        return compare.text_prompt(req)

    @app.post("/api/click", response_model=CompareResponse)
    def click(req: ClickPromptRequest) -> CompareResponse:
        return compare.click_prompt(req)

    @app.get("/api/testset/list", response_model=TestSetListResponse)
    def testset_list(offset: int = 0, limit: int = 200, category: Optional[str] = None) -> TestSetListResponse:
        return compare.testset_list(offset, limit, category)

    @app.post("/api/testset/load", response_model=TestImageLoadResponse)
    def testset_load(req: TestImageLoadRequest) -> TestImageLoadResponse:
        return compare.testset_load(req)

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(static_dir / "index.html")

    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
    return app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pretrained-ckpt",
        type=Path,
        default=Path("/home/ec2-user/SageMaker/efs/Models/sam3/sam3.pt"),
        help="Local SAM3 pretrained checkpoint. Pass an empty string to use HF download.",
    )
    parser.add_argument(
        "--finetuned-ckpt",
        type=Path,
        required=True,
        help="MERGED fine-tuned checkpoint (run merge_checkpoint.py first). "
        "The raw trainer output is missing the SAM-1 click predictor and "
        "SAM-2 neck convs, which makes the click-mode predictions garbage.",
    )
    parser.add_argument(
        "--bpe-path",
        type=Path,
        default=Path("sam3/assets/bpe_simple_vocab_16e6.txt.gz"),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument(
        "--test-coco",
        type=Path,
        default=Path("data/AWS_SAM_split/test.json"),
        help="COCO test JSON to expose via the test-set browser. Pass empty "
        "string to disable.",
    )
    parser.add_argument(
        "--test-image-root",
        type=Path,
        default=Path("data/AWS_SAM"),
        help="Directory containing test images. file_name entries in --test-coco "
        "are resolved relative to this.",
    )
    args = parser.parse_args()

    pre_ckpt = args.pretrained_ckpt if str(args.pretrained_ckpt) else None
    pretrained = ModelHandle("pretrained", pre_ckpt, args.bpe_path, args.device)
    finetuned = ModelHandle("finetuned", args.finetuned_ckpt, args.bpe_path, args.device)

    testset: Optional[TestSet] = None
    if str(args.test_coco) and args.test_coco.exists():
        try:
            testset = TestSet(args.test_coco, args.test_image_root)
        except Exception as e:
            LOG.warning("[testset] disabled: %s", e)
            testset = None
    else:
        LOG.info("[testset] disabled (no --test-coco given or path missing)")

    static_dir = Path(__file__).resolve().parent / "static"
    app = build_app(pretrained, finetuned, static_dir, testset=testset)

    LOG.info("Server starting on http://%s:%d", args.host, args.port)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
