#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved
"""Smoke-test invocation of a deployed SAM3 SageMaker endpoint.

Usage:
  python sagemaker/deploy/invoke_example.py \
      --endpoint sam3-aws-sam \
      --image data/AWS_SAM/companypremises2025101600217.png \
      --text grass

  python sagemaker/deploy/invoke_example.py \
      --endpoint sam3-aws-sam \
      --image <path> \
      --click 520 375
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--text", default=None, help="Text noun phrase prompt.")
    parser.add_argument(
        "--click", nargs=2, type=int, default=None, metavar=("X", "Y"),
        help="Single positive click instead of text.",
    )
    parser.add_argument("--region", default=None)
    args = parser.parse_args()

    if args.text is None and args.click is None:
        parser.error("pass --text or --click")

    img_b64 = base64.b64encode(args.image.read_bytes()).decode("ascii")

    if args.text is not None:
        body = {"mode": "text", "image_b64": img_b64, "text": args.text}
    else:
        body = {
            "mode": "click",
            "image_b64": img_b64,
            "points": [args.click],
            "labels": [1],
            "multimask_output": True,
        }

    import boto3

    session = boto3.Session(region_name=args.region)
    smr = session.client("sagemaker-runtime")
    resp = smr.invoke_endpoint(
        EndpointName=args.endpoint,
        ContentType="application/json",
        Body=json.dumps(body),
    )
    out = json.loads(resp["Body"].read())
    n_masks = len(out.get("masks_rle", []))
    print(f"endpoint={args.endpoint} n_masks={n_masks} scores={out.get('scores')}")
    print(f"image_size={out.get('image_size')}")
    if n_masks == 0:
        sys.exit(2)


if __name__ == "__main__":
    main()
