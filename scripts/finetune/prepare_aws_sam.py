#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved
"""Convert AWS_SAM LabelMe-format polygons into COCO JSON files and split 0.9/0.1.

The AWS_SAM dataset stores per-image LabelMe annotations alongside the image:
    data/AWS_SAM/foo.png
    data/AWS_SAM/foo.json   # {"shapes": [{"label": ..., "points": [[x,y], ...]}, ...]}

This script:
  1. Walks the dataset, pairs each .json with the matching image,
  2. Converts polygons to COCO annotations (segmentation = list-of-polygon-list,
     bbox = xywh, area = polygon area),
  3. Writes train.json / test.json into <out_dir> (default: data/AWS_SAM_split/),
  4. Optionally drops categories whose total instance count is below --min-count.

Output layout:
    <out_dir>/train.json
    <out_dir>/test.json
    <out_dir>/categories.json   # id -> name, for inspection

Image folder is left in place — the COCO files reference images via the
original directory (default: data/AWS_SAM) using relative file_name.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import random
from pathlib import Path
from typing import Iterable

# Image extensions to probe for each .json (case-insensitive).
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def find_image_for_json(json_path: Path) -> Path | None:
    # Strip ONLY the trailing ".json" — many AWS_SAM filenames contain
    # additional dots (e.g. 1730876719.714330.json), so Path.with_suffix
    # would mangle them.
    base = json_path.parent / json_path.name[: -len(".json")]
    for ext in IMAGE_EXTS:
        for cased in (ext, ext.upper()):
            cand = base.with_name(base.name + cased)
            if cand.exists():
                return cand
    return None


def polygon_bbox_xywh(points: list[list[float]]) -> tuple[float, float, float, float]:
    xs = [float(p[0]) for p in points]
    ys = [float(p[1]) for p in points]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    return x_min, y_min, x_max - x_min, y_max - y_min


def polygon_area(points: list[list[float]]) -> float:
    # Shoelace.
    n = len(points)
    if n < 3:
        return 0.0
    s = 0.0
    for i in range(n):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % n]
        s += x1 * y2 - x2 * y1
    return abs(s) * 0.5


def collect_records(data_dir: Path) -> list[dict]:
    records = []
    for jp in sorted(data_dir.glob("*.json")):
        ip = find_image_for_json(jp)
        if ip is None:
            continue
        records.append({"json": jp, "image": ip})
    return records


def labels_in_record(rec: dict) -> Iterable[str]:
    with open(rec["json"], "r") as f:
        data = json.load(f)
    for s in data.get("shapes", []):
        lbl = s.get("label")
        if lbl is not None:
            yield lbl


def build_label_set(
    records: list[dict], min_count: int
) -> tuple[list[str], collections.Counter]:
    counter: collections.Counter = collections.Counter()
    for rec in records:
        for lbl in labels_in_record(rec):
            counter[lbl] += 1
    kept = sorted([k for k, v in counter.items() if v >= min_count])
    return kept, counter


def make_coco(
    records: list[dict],
    label_to_id: dict[str, int],
    image_root: Path,
    keep_only_polygon: bool = True,
) -> dict:
    images = []
    annotations = []
    next_image_id = 1
    next_ann_id = 1

    for rec in records:
        with open(rec["json"], "r") as f:
            data = json.load(f)
        h = data.get("imageHeight")
        w = data.get("imageWidth")
        if h is None or w is None:
            # Fall back to PIL only if the JSON didn't carry dims.
            from PIL import Image as _PILImage  # local import — optional dep

            with _PILImage.open(rec["image"]) as im:
                w, h = im.size

        # COCO file_name is relative to the image folder we will pass to the trainer.
        rel = os.path.relpath(rec["image"], image_root)
        image_id = next_image_id
        next_image_id += 1
        images.append(
            {
                "id": image_id,
                "file_name": rel,
                "height": int(h),
                "width": int(w),
            }
        )

        for shape in data.get("shapes", []):
            label = shape.get("label")
            if label is None or label not in label_to_id:
                continue
            stype = shape.get("shape_type", "polygon")
            if keep_only_polygon and stype != "polygon":
                continue
            points = shape.get("points") or []
            if len(points) < 3:
                continue
            x, y, bw, bh = polygon_bbox_xywh(points)
            if bw <= 0 or bh <= 0:
                continue
            flat = [float(c) for p in points for c in p[:2]]
            annotations.append(
                {
                    "id": next_ann_id,
                    "image_id": image_id,
                    "category_id": label_to_id[label],
                    "bbox": [float(x), float(y), float(bw), float(bh)],
                    "area": float(polygon_area(points)),
                    "iscrowd": 0,
                    "segmentation": [flat],
                }
            )
            next_ann_id += 1

    categories = [
        {"id": cid, "name": name, "supercategory": "aws_sam"}
        for name, cid in sorted(label_to_id.items(), key=lambda kv: kv[1])
    ]
    return {
        "info": {"description": "AWS_SAM (LabelMe -> COCO)"},
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        default="data/AWS_SAM",
        type=Path,
        help="Folder with paired image+json LabelMe annotations.",
    )
    parser.add_argument(
        "--out-dir",
        default="data/AWS_SAM_split",
        type=Path,
        help="Where to write train.json / test.json / categories.json.",
    )
    parser.add_argument(
        "--train-frac", type=float, default=0.9, help="Train fraction (default 0.9)."
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--min-count",
        type=int,
        default=20,
        help="Drop categories with fewer than this many instances across the dataset.",
    )
    args = parser.parse_args()

    data_dir = args.data_dir.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    records = collect_records(data_dir)
    if not records:
        raise SystemExit(f"No paired image+json found under {data_dir}")
    print(f"[prepare_aws_sam] paired records: {len(records)}")

    kept_labels, counter = build_label_set(records, args.min_count)
    print(
        f"[prepare_aws_sam] labels kept (>= {args.min_count} instances): "
        f"{len(kept_labels)} / {len(counter)} total"
    )
    label_to_id = {name: i + 1 for i, name in enumerate(kept_labels)}

    rng = random.Random(args.seed)
    indices = list(range(len(records)))
    rng.shuffle(indices)
    n_train = int(round(len(indices) * args.train_frac))
    train_idx = set(indices[:n_train])
    train_records = [records[i] for i in range(len(records)) if i in train_idx]
    test_records = [records[i] for i in range(len(records)) if i not in train_idx]
    print(
        f"[prepare_aws_sam] split: train={len(train_records)} test={len(test_records)}"
    )

    train_coco = make_coco(train_records, label_to_id, data_dir)
    test_coco = make_coco(test_records, label_to_id, data_dir)

    train_out = out_dir / "train.json"
    test_out = out_dir / "test.json"
    cat_out = out_dir / "categories.json"

    with open(train_out, "w") as f:
        json.dump(train_coco, f)
    with open(test_out, "w") as f:
        json.dump(test_coco, f)
    with open(cat_out, "w") as f:
        json.dump(
            {
                "label_to_id": label_to_id,
                "instance_counts": counter.most_common(),
                "image_root": str(data_dir),
            },
            f,
            indent=2,
        )

    print(f"[prepare_aws_sam] wrote {train_out}")
    print(f"[prepare_aws_sam] wrote {test_out}")
    print(f"[prepare_aws_sam] wrote {cat_out}")
    print(
        f"[prepare_aws_sam] image_root for trainer: {data_dir}  "
        "(file_name in COCO is relative to this)"
    )


if __name__ == "__main__":
    main()
