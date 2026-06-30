#!/usr/bin/env python3
"""Generate launch_sagemaker_training.ipynb from a Python definition.

Cleaner than hand-editing JSON. Run once after you change the cell text:
    python scripts/finetune/sagemaker/build_notebook.py
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path


def md(src: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": textwrap.dedent(src).strip().splitlines(keepends=True),
    }


def code(src: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": textwrap.dedent(src).strip().splitlines(keepends=True),
    }


cells = [
    md("""
        # SAM3 fine-tuning on SageMaker TrainingJobs

        This notebook launches a SageMaker TrainingJob that fine-tunes SAM3 on
        AWS_SAM using the recipe in `sam3/train/configs/aws_sam/aws_sam_finetune.yaml`.

        **Inputs** (S3 channels passed to the job):

        - `train`  → contains both `AWS_SAM/` (images + LabelMe JSONs) and
          `AWS_SAM_split/` (COCO train.json / test.json from `prepare_aws_sam.py`)
        - `pretrained` → contains `sam3.pt` (the local SAM3 weights)

        **Outputs**:

        - `s3://<bucket>/<prefix>/output/model.tar.gz` — contains
          `checkpoint_merged.pt` (the self-contained fine-tuned checkpoint) +
          the BPE vocab. Ready to download and pass to eval / SageMaker deploy.

        ## Prerequisites

        ```bash
        pip install sagemaker boto3
        aws configure   # or assume an IAM role with sagemaker:* + s3 access
        ```

        Run this notebook on a machine that has the SAM3 source tree (so the
        SDK can package it) and AWS credentials. A SageMaker notebook
        instance or your laptop both work.
    """),

    code("""
        # Pin sagemaker<3 — the v3 SDK (sagemaker>=3.0) is a major rewrite
        # that moves PyTorch estimator + Session + get_execution_role out
        # of their canonical paths. This notebook targets the long-stable
        # v2 API.
        %pip install --quiet 'sagemaker>=2.230,<3' boto3
        # IMPORTANT: after installing, restart the kernel before running
        # the next cell — otherwise a previously-imported v3 module stays
        # cached.
    """),

    code("""
        # ---------- Configuration: edit these ----------
        import os
        import sagemaker
        import boto3

        # Sanity-check SDK version. The v3 SDK has a different layout and
        # this notebook is written against v2. The cell above pins
        # sagemaker<3.
        _v = getattr(sagemaker, "__version__", "unknown")
        if _v.startswith("3.") or _v.startswith("4."):
            raise RuntimeError(
                f"Detected sagemaker {_v}, but this notebook requires v2.x. "
                f"Run the previous cell, then RESTART THE KERNEL, then retry."
            )
        print("sagemaker version:", _v)

        # v2-style imports (re-exported at top-level).
        from sagemaker import get_execution_role

        sess = sagemaker.Session()
        role = get_execution_role()
        sagemaker_default_bucket = sess.default_bucket()
        region = sess.boto_session.region_name
        print("sagemaker_default_bucket:", sagemaker_default_bucket)
        print("sagemaker_region:", region)

        S3_PREFIX        = "sam3/aws_sam_finetune_v1"

        # Local paths (these get uploaded to s3 once at launch time)
        LOCAL_REPO_ROOT  = "/home/ec2-user/SageMaker/efs/Projects/sam3"
        LOCAL_DATA       = "/home/ec2-user/SageMaker/efs/Projects/sam3/data"             # contains AWS_SAM/ and AWS_SAM_split/
        LOCAL_PRETRAINED = "/home/ec2-user/SageMaker/efs/Models/sam3"                    # contains sam3.pt

        # Instance & training settings
        INSTANCE_TYPE    = "ml.p4de.24xlarge"     # 8 × A100 80GB. Use ml.p5.48xlarge for 8 × H100.
        INSTANCE_COUNT   = 1                     # multi-node not wired up yet
        NUM_GPUS         = 8                     # per-instance
        MAX_RUNTIME_S    = 24 * 3600
        VOLUME_SIZE_GB   = 500

        # Hyperparameter overrides (None = use yaml defaults)
        HP = {
            "num-gpus":           NUM_GPUS,
            # Optional — set/uncomment to override config defaults:
            # "max-epochs":         20,
            # "train-batch-size":   16,
            # "lr-scale":           0.1,
            # "num-train-workers":  0,
        }

        # Job name (timestamp will be appended automatically by the SDK)
        JOB_NAME_PREFIX = "sam3-aws-sam-ft"

        print("repo root:", LOCAL_REPO_ROOT)
    """),

    md("""
        ## 1. Upload data + pretrained weights to S3

        The training job needs the dataset and pretrained checkpoint pre-staged
        in S3. This step is **one-time** — subsequent runs reuse the same S3 paths.
    """),

    code("""
        import boto3
        import sagemaker
        from sagemaker.s3 import S3Uploader

        boto_session = boto3.Session(region_name=REGION)
        sagemaker_session = sagemaker.Session(boto_session=boto_session)

        s3_train_uri      = f"s3://{S3_BUCKET}/{S3_PREFIX}/input/train"
        s3_pretrained_uri = f"s3://{S3_BUCKET}/{S3_PREFIX}/input/pretrained"
        s3_output_uri     = f"s3://{S3_BUCKET}/{S3_PREFIX}/output"

        print("train data    →", s3_train_uri)
        print("pretrained    →", s3_pretrained_uri)
        print("output        →", s3_output_uri)
    """),

    code("""
        # Upload the dataset (AWS_SAM/ + AWS_SAM_split/). Comment out after the
        # first successful upload — the data is large.
        S3Uploader.upload(
            local_path=LOCAL_DATA,
            desired_s3_uri=s3_train_uri,
            sagemaker_session=sagemaker_session,
        )
        print("✓ data uploaded")
    """),

    code("""
        # Upload the pretrained checkpoint (~9 GB). Comment out after the
        # first successful upload.
        S3Uploader.upload(
            local_path=LOCAL_PRETRAINED,
            desired_s3_uri=s3_pretrained_uri,
            sagemaker_session=sagemaker_session,
        )
        print("✓ pretrained uploaded")
    """),

    md("""
        ## 2. Launch the TrainingJob

        SageMaker's PyTorch estimator packages the local source tree (the
        sam3 repo) and ships it to the container, then runs `train_entry.py`
        as the entry point. The script installs deps, patches the config to
        use SageMaker channel paths, runs `sam3/train/train.py`, then merges
        the resulting checkpoint and writes it to `/opt/ml/model/`.
    """),

    code("""
        from sagemaker.pytorch import PyTorch
        from sagemaker.inputs import TrainingInput

        estimator = PyTorch(
            entry_point="train_entry.py",
            source_dir=LOCAL_REPO_ROOT,                       # whole repo goes into the container
            role=ROLE,
            framework_version="2.4.0",                        # cu124 base image
            py_version="py311",
            instance_type=INSTANCE_TYPE,
            instance_count=INSTANCE_COUNT,
            volume_size=VOLUME_SIZE_GB,
            max_run=MAX_RUNTIME_S,
            output_path=s3_output_uri,
            base_job_name=JOB_NAME_PREFIX,
            sagemaker_session=sagemaker_session,
            hyperparameters=HP,                                # forwarded as CLI args
            environment={
                "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
                # Tells train_entry.py where to find its own script. SageMaker
                # by default chdirs into /opt/ml/code/ which contains the
                # repo; the entry script lives at the same path it has in
                # the repo so this is mostly informational.
                "SAM3_REPO_ROOT": "/opt/ml/code",
            },
            disable_profiler=True,
            # Tip: set this if you want fast restart on interruption.
            keep_alive_period_in_seconds=1800,
        )

        # The SDK looks for source_dir/entry_point. Because train_entry.py
        # lives under scripts/finetune/sagemaker/, we need to tell the SDK
        # to look there for the entry point. The cleanest way is to set
        # entry_point as a relative path under source_dir:
        estimator.entry_point = "scripts/finetune/sagemaker/train_entry.py"

        print("Estimator built. ImageURI:", estimator.training_image_uri())
    """),

    code("""
        # Kick it off. estimator.fit() is blocking — pass wait=False if you
        # want the cell to return immediately and monitor via the SageMaker
        # console.
        estimator.fit(
            inputs={
                "train":      TrainingInput(s3_data=s3_train_uri),
                "pretrained": TrainingInput(s3_data=s3_pretrained_uri),
            },
            wait=True,        # set False to background the job
            logs="All",       # stream all container logs to this notebook
        )
        print("Job name:", estimator.latest_training_job.job_name)
        print("Model artifact:", estimator.model_data)
    """),

    md("""
        ## 3. Download the merged checkpoint back to local disk

        SageMaker uploads anything written to `/opt/ml/model/` as a
        `model.tar.gz` in the output S3 path. `train_entry.py` writes
        `checkpoint_merged.pt` (the self-contained fine-tuned weights) +
        the BPE vocab there.
    """),

    code("""
        from sagemaker.s3 import S3Downloader

        # estimator.model_data is the s3 URI of model.tar.gz
        local_artifact = "runs/sagemaker_artifact"
        os.makedirs(local_artifact, exist_ok=True)
        S3Downloader.download(
            s3_uri=estimator.model_data,
            local_path=local_artifact,
            sagemaker_session=sagemaker_session,
        )
        print("downloaded to:", local_artifact)

        # Unpack
        import tarfile
        tar_path = os.path.join(local_artifact, "model.tar.gz")
        with tarfile.open(tar_path) as t:
            t.extractall(local_artifact)
        print("contents:", os.listdir(local_artifact))
    """),

    md("""
        ## 4. Run eval against the downloaded checkpoint

        The merged file is identical to what `merge_checkpoint.py` would
        produce locally, so you can plug it straight into the eval harness:

        ```bash
        python scripts/finetune/eval/evaluate_interactive.py \\
            --coco data/AWS_SAM_split/test.json \\
            --image-root data/AWS_SAM \\
            --checkpoint runs/sagemaker_artifact/checkpoint_merged.pt \\
            --output runs/eval/finetuned_sagemaker.json
        ```

        ## Notes / gotchas

        - **First run is slow** because `pip install -e ".[dev,train]"` runs
          inside the container. If you do many runs, consider building a
          custom Docker image that has SAM3 pre-installed and pass it via
          `image_uri=...` on the estimator.
        - **S3 charges and bandwidth**: data + pretrained ≈ 10 GB, model
          artifact ≈ 9 GB. Cleanup with `aws s3 rm --recursive` when done.
        - **Spot training**: add `use_spot_instances=True, max_wait=...` to
          the estimator to drop costs ~60% on long jobs at the risk of
          preemption. SageMaker auto-resumes from the trainer's last
          checkpoint on the next replica.
        - **Multi-instance**: set `instance_count > 1` only after you've
          wired up the trainer's distributed init for multi-node SLURM-less
          launch. The current trainer uses single-node-multi-GPU via
          `mp.spawn`; multi-node SageMaker would need the `submitit` path
          or a `torchrun`-style wrapper around `train_entry.py`.
    """),
]


nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.12",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}


out = Path(__file__).resolve().parent / "launch_sagemaker_training.ipynb"
out.write_text(json.dumps(nb, indent=1))
print(f"wrote {out}")
