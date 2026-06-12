#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved
"""Sample N images from the AWS_SAM test split and render polygon masks on top.

Reads the COCO file written by `prepare_aws_sam.py`, samples N image ids
deterministically (seeded), and writes RGBA overlays into <out-dir>.

Each polygon is filled with a per-category color and the label is drawn at
the polygon centroid.
"""

from __future__ import annotations

import argparse
import colorsys
import json
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def category_color(cat_id: int) -> tuple[int, int, int]:
    # Distribute hues over the unit circle, vary saturation/value mildly.
    h = (cat_id * 0.6180339887) % 1.0  # golden-ratio hash → well-spread hues
    s = 0.65 + 0.2 * ((cat_id % 3) / 2.0)
    v = 0.85 + 0.1 * ((cat_id % 2))
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return int(r * 255), int(g * 255), int(b * 255)


def polygon_centroid(flat_xy: list[float]) -> tuple[float, float]:
    xs = flat_xy[0::2]
    ys = flat_xy[1::2]
    return sum(xs) / len(xs), sum(ys) / len(ys)


def get_font(size: int) -> ImageFont.ImageFont:
    for candidate in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        if Path(candidate).exists():
            try:
                return ImageFont.truetype(candidate, size)
            except OSError:
                pass
    return ImageFont.load_default()


def render_one(
    img_path: Path,
    anns: list[dict],
    cat_id_to_name: dict[int, str],
    alpha: int = 110,
) -> Image.Image:
    base = Image.open(img_path).convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    for ann in anns:
        cat_id = ann["category_id"]
        color = category_color(cat_id)
        for poly in ann.get("segmentation", []):
            if len(poly) < 6:
                continue
            pts = [(poly[i], poly[i + 1]) for i in range(0, len(poly), 2)]
            draw.polygon(pts, fill=color + (alpha,), outline=color + (255,))

    composite = Image.alpha_composite(base, overlay)
    text_layer = ImageDraw.Draw(composite)
    font = get_font(max(14, base.size[0] // 80))
    seen_pos: list[tuple[float, float]] = []
    for ann in anns:
        cat_id = ann["category_id"]
        color = category_color(cat_id)
        for poly in ann.get("segmentation", []):
            if len(poly) < 6:
                continue
            cx, cy = polygon_centroid(poly)
            # Skip near-duplicates so labels don't pile up.
            if any(abs(cx - x) + abs(cy - y) < 25 for x, y in seen_pos):
                continue
            seen_pos.append((cx, cy))
            label = cat_id_to_name.get(cat_id, str(cat_id))
            try:
                bbox = text_layer.textbbox((cx, cy), label, font=font)
                pad = 2
                text_layer.rectangle(
                    (bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad),
                    fill=(0, 0, 0, 180),
                )
            except AttributeError:
                pass
            text_layer.text((cx, cy), label, fill=color + (255,), font=font)

    return composite.convert("RGB")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--coco",
        default="data/AWS_SAM_split/test.json",
        type=Path,
        help="COCO test file produced by prepare_aws_sam.py.",
    )
    parser.add_argument(
        "--image-root",
        default="data/AWS_SAM",
        type=Path,
        help="Directory containing the actual image files (matches COCO file_name).",
    )
    parser.add_argument(
        "--out-dir",
        default="data/AWS_SAM_split/test_vis",
        type=Path,
    )
    parser.add_argument("--num", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    with open(args.coco, "r") as f:
        coco = json.load(f)

    cat_id_to_name = {c["id"]: c["name"] for c in coco["categories"]}
    anns_by_image: dict[int, list[dict]] = {}
    for ann in coco["annotations"]:
        anns_by_image.setdefault(ann["image_id"], []).append(ann)

    images_with_anns = [im for im in coco["images"] if anns_by_image.get(im["id"])]
    if not images_with_anns:
        raise SystemExit("No annotated test images found.")

    rng = random.Random(args.seed)
    sample = rng.sample(images_with_anns, k=min(args.num, len(images_with_anns)))

    for im in sample:
        img_path = args.image_root / im["file_name"]
        if not img_path.exists():
            print(f"[viz] missing image: {img_path}, skipping")
            continue
        out_path = args.out_dir / f"{Path(im['file_name']).stem}_vis.jpg"
        rendered = render_one(img_path, anns_by_image[im["id"]], cat_id_to_name)
        rendered.save(out_path, quality=92)
        print(f"[viz] wrote {out_path}")

    print(f"[viz] done — {len(sample)} images written to {args.out_dir}")


if __name__ == "__main__":
    main()
