#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved
"""Package a SAM3 checkpoint and deploy it as a SageMaker real-time endpoint.

What this does:
  1. Builds a model.tar.gz containing
       checkpoint.pt
       bpe_simple_vocab_16e6.txt.gz
       code/inference.py
       code/requirements.txt
     (the `code/` subtree is what SageMaker auto-installs and imports.)
  2. Uploads model.tar.gz to s3://<bucket>/<prefix>/model.tar.gz
  3. Calls sagemaker.pytorch.PyTorchModel.deploy(...) to create the endpoint.

Run from the repo root, on a host with AWS creds + the `sagemaker` SDK
installed (`pip install sagemaker boto3`).

Example:
  python sagemaker/deploy/deploy.py \
      --checkpoint runs/aws_sam_finetune/checkpoints/checkpoint.pt \
      --role arn:aws:iam::123456789012:role/SageMakerRole \
      --bucket my-sagemaker-bucket \
      --prefix sam3/aws_sam_v1 \
      --instance-type ml.g5.xlarge \
      --endpoint-name sam3-aws-sam

Use --pretrained instead of --checkpoint to deploy facebook/sam3 with no
custom weights (the inference handler will pull them from HF on container
boot).
"""

from __future__ import annotations

import argparse
import os
import shutil
import tarfile
import tempfile
import time
from pathlib import Path


def build_model_tar(
    checkpoint: Path | None,
    bpe_path: Path,
    code_dir: Path,
    out_path: Path,
) -> Path:
    """Assemble model.tar.gz with the layout SageMaker's PyTorch container expects."""
    with tempfile.TemporaryDirectory() as td:
        staging = Path(td)
        # Top-level model artifacts
        if checkpoint is not None:
            shutil.copy(checkpoint, staging / "checkpoint.pt")
        shutil.copy(bpe_path, staging / Path(bpe_path).name)

        # code/ subdir — SageMaker auto-installs requirements.txt from here
        code_dst = staging / "code"
        code_dst.mkdir()
        for fname in ("inference.py", "requirements.txt"):
            src = code_dir / fname
            if src.exists():
                shutil.copy(src, code_dst / fname)

        with tarfile.open(out_path, "w:gz") as tf:
            for p in staging.rglob("*"):
                tf.add(p, arcname=p.relative_to(staging))
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=None,
                        help="Local SAM3 checkpoint to bundle. Omit to use --pretrained.")
    parser.add_argument("--pretrained", action="store_true",
                        help="Deploy facebook/sam3 from HF (no local checkpoint).")
    parser.add_argument("--role", required=True,
                        help="SageMaker IAM execution role ARN.")
    parser.add_argument("--bucket", required=True, help="S3 bucket for the tarball.")
    parser.add_argument("--prefix", default="sam3/deploy",
                        help="S3 key prefix.")
    parser.add_argument("--instance-type", default="ml.g5.xlarge")
    parser.add_argument("--instance-count", type=int, default=1)
    parser.add_argument("--endpoint-name", default=None,
                        help="Endpoint name. Default: sam3-<timestamp>.")
    parser.add_argument("--region", default=None,
                        help="AWS region (default: from boto/SageMaker session).")
    parser.add_argument("--framework-version", default="2.3.0",
                        help="PyTorch container version.")
    parser.add_argument("--py-version", default="py311")
    parser.add_argument("--bpe-path", type=Path,
                        default=Path("sam3/assets/bpe_simple_vocab_16e6.txt.gz"))
    parser.add_argument("--code-dir", type=Path,
                        default=Path("sagemaker/deploy"))
    parser.add_argument("--out-tar", type=Path,
                        default=Path("runs/deploy/model.tar.gz"))
    parser.add_argument("--dry-run", action="store_true",
                        help="Build the tarball but skip upload/deploy.")
    args = parser.parse_args()

    if not args.pretrained and args.checkpoint is None:
        parser.error("provide --checkpoint or --pretrained")

    args.out_tar.parent.mkdir(parents=True, exist_ok=True)
    print(f"[deploy] building model tarball -> {args.out_tar}")
    build_model_tar(
        checkpoint=args.checkpoint,
        bpe_path=args.bpe_path,
        code_dir=args.code_dir,
        out_path=args.out_tar,
    )
    print(f"[deploy] tarball size: {args.out_tar.stat().st_size / 1e9:.2f} GB")

    if args.dry_run:
        print("[deploy] --dry-run: stopping before upload/deploy")
        return

    import boto3
    import sagemaker
    from sagemaker.pytorch import PyTorchModel

    session = sagemaker.Session(boto_session=boto3.Session(region_name=args.region))
    s3_key = f"{args.prefix}/model.tar.gz"
    print(f"[deploy] uploading to s3://{args.bucket}/{s3_key}")
    model_data = session.upload_data(
        path=str(args.out_tar), bucket=args.bucket, key_prefix=args.prefix
    )
    print(f"[deploy] uploaded: {model_data}")

    endpoint_name = args.endpoint_name or f"sam3-{int(time.time())}"

    model = PyTorchModel(
        model_data=model_data,
        role=args.role,
        framework_version=args.framework_version,
        py_version=args.py_version,
        entry_point="inference.py",
        sagemaker_session=session,
        env={
            # Pre-cache HF home so first request doesn't time out on facebook/sam3
            # download. Set HF_TOKEN here only if the role can't read from HF.
            "TRANSFORMERS_CACHE": "/opt/ml/model/.hf_cache",
            "HF_HOME": "/opt/ml/model/.hf_cache",
        },
    )
    print(f"[deploy] creating endpoint {endpoint_name} on {args.instance_type}")
    predictor = model.deploy(
        initial_instance_count=args.instance_count,
        instance_type=args.instance_type,
        endpoint_name=endpoint_name,
    )
    print(f"[deploy] endpoint ready: {predictor.endpoint_name}")
    print(
        "[deploy] invoke with the boto3 SageMaker runtime client; see "
        "sagemaker/deploy/invoke_example.py for a working sample."
    )


if __name__ == "__main__":
    main()
