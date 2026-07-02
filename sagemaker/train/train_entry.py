#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved
"""SageMaker training entry point for SAM3 fine-tuning.

The notebook in this directory packages the SAM3 source + this script and
launches a SageMaker TrainingJob. SageMaker drops your training data into
/opt/ml/input/data/<channel>/ and expects checkpoints + model artifacts
under /opt/ml/model/ when the job finishes. This script:

  1. Installs missing runtime deps that aren't in the base PyTorch image.
  2. Resolves $PWD-relative defaults in aws_sam_finetune.yaml to the
     SageMaker channel paths via Hydra dotted-key overrides… but
     train.py doesn't accept CLI overrides, so we write a derived YAML.
  3. Invokes sam3/train/train.py with the resulting config.
  4. After training finishes, runs merge_checkpoint.py and copies the
     merged checkpoint into /opt/ml/model/ so SageMaker uploads it to S3.

Hyperparameters passed to the SageMaker Estimator are forwarded as
argparse args (max-epochs, train-batch-size, etc.).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


# SageMaker conventions
SM_MODEL_DIR = Path(os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
SM_OUTPUT_DIR = Path(os.environ.get("SM_OUTPUT_DATA_DIR", "/opt/ml/output/data"))
SM_CHANNEL_TRAIN = Path(os.environ.get("SM_CHANNEL_TRAIN", "/opt/ml/input/data/train"))
SM_CHANNEL_PRETRAINED = Path(
    os.environ.get("SM_CHANNEL_PRETRAINED", "/opt/ml/input/data/pretrained")
)
# `/opt/ml/checkpoints/` — anything written here is CONTINUOUSLY mirrored
# to `checkpoint_s3_uri` by SageMaker (background rsync throughout the
# job). SM_MODEL_DIR only uploads once at job end; SM_OUTPUT_DATA_DIR is
# ephemeral and never leaves the container. So per-epoch checkpoints
# MUST be written under SM_CHECKPOINT_DIR to survive to S3.
# Requires the estimator to be built with:
#   checkpoint_s3_uri="s3://<bucket>/<prefix>/checkpoints",
#   checkpoint_local_path="/opt/ml/checkpoints",
# (see launch_training.ipynb cell 062be268).
SM_CHECKPOINT_DIR = Path(os.environ.get("SM_CHECKPOINT_DIR", "/opt/ml/checkpoints"))


def ensure_shm_capacity() -> None:
    """Remount /dev/shm larger if it's the SageMaker default (~64 MB).

    torch.multiprocessing.spawn workers hit /dev/shm hard during startup
    (module import + IPC handle setup). 8 ranks * ~150 MB imports easily
    trips ENOSPC on the default 64 MB tmpfs — the child then aborts with
    SIGABRT before its Python traceback ever prints. We bump the tmpfs to
    16 GB (well under the 200+ GB of RAM on p4de/p5); a no-op if the
    container was already launched with `--shm-size` big enough.
    """
    try:
        st = os.statvfs("/dev/shm")
        total = st.f_frsize * st.f_blocks
    except FileNotFoundError:
        print("[train_entry] /dev/shm missing — skipping remount")
        return
    if total >= 4 * 1024**3:
        print(f"[train_entry] /dev/shm is already {total/1e9:.1f} GB — leaving alone")
        return
    print(f"[train_entry] /dev/shm is only {total/1e6:.0f} MB, remounting to 16 GB")
    rc = subprocess.run(
        ["mount", "-o", "remount,size=16g", "/dev/shm"],
        check=False,
    )
    if rc.returncode != 0:
        print("[train_entry] WARNING: could not remount /dev/shm — training may abort")


def install_runtime_deps() -> None:
    """Install SAM3 runtime deps that the base image may be missing."""
    # Force a CUDA-12.8-matching torch first. The SageMaker PyTorch 2.4
    # base image ships a CUDA 12.8 driver (`torch._C._cuda_init` reports
    # driver version 12080). `pip install -e ".[dev,train]"` alone would
    # let pip resolve torch>=2.7 to the default PyPI wheel — which for
    # recent torch releases is built against CUDA 12.9+, triggering
    # "The NVIDIA driver on your system is too old (found version 12080)".
    #
    # Pin to a KNOWN-GOOD cu128 wheel. `torch>=2.7,<2.12` resolves to
    # whatever's latest on the cu128 index, which drifts and can change
    # SDPA / activation-ckpt memory usage between runs. Pinning to
    # 2.11.0 matches the local ec2 env that was measured to fit
    # batch_size=8 in ~70 GB/rank on the same p4de.24xlarge silicon.
    subprocess.run(
        [
            sys.executable, "-m", "pip", "install",
            "--index-url", "https://download.pytorch.org/whl/cu128",
            "torch==2.11.0", "torchvision==0.26.0",
            "--quiet",
        ],
        check=True,
    )
    # Now install SAM3 + its extras. torch/torchvision constraints are
    # already satisfied, so this step won't try to re-download them.
    subprocess.run(
        [
            sys.executable, "-m", "pip", "install",
            "-e", ".[dev,train]",
            "--quiet",
        ],
        check=True,
    )
    # Some transitive deps that pyproject lists as base deps but were
    # discovered later — make sure they're present even on older base images.
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "einops", "psutil", "--quiet"],
        check=False,
    )


def write_runtime_config(args: argparse.Namespace) -> Path:
    """Write a config file with SageMaker-channel paths and any hyperparam
    overrides patched in. Returns the path of the derived file.

    src == dst: we read `_sm.yaml`, patch it in-memory (SageMaker channel
    paths, plus any HP overrides passed as CLI args), and write it back
    to the same location. That means any hand-edits committed to
    `_sm.yaml` locally ship in the tarball and stick on SageMaker — the
    patcher only rewrites the ${oc.env:PWD} placeholders and the four
    hyperparameters below, leaving every other line alone.
    """
    src = Path(args.config_template)
    if not src.exists():
        raise SystemExit(f"config template not found: {src}")

    text = src.read_text()

    # Replace the ${oc.env:PWD}-relative paths with SageMaker channel paths.
    # experiment_log_dir points at SM_CHECKPOINT_DIR (=/opt/ml/checkpoints/)
    # so per-epoch checkpoints AND tensorboard logs stream continuously to
    # `checkpoint_s3_uri`. WITHOUT this remap the trainer writes to
    # /opt/ml/output/data/... which is ephemeral (survives only until job
    # end) and NOT synced to S3, so per-epoch checkpoints appear to vanish.
    replacements = {
        "${oc.env:PWD}/data/AWS_SAM": str(SM_CHANNEL_TRAIN / "AWS_SAM"),
        "${oc.env:PWD}/data/AWS_SAM_split": str(SM_CHANNEL_TRAIN / "AWS_SAM_split"),
        "${oc.env:PWD}/runs/aws_sam_finetune": str(SM_CHECKPOINT_DIR / "aws_sam_finetune"),
        "${oc.env:PWD}/sam3/assets/bpe_simple_vocab_16e6.txt.gz":
            "sam3/assets/bpe_simple_vocab_16e6.txt.gz",
        "/home/ec2-user/SageMaker/efs/Models/sam3/sam3.pt":
            str(SM_CHANNEL_PRETRAINED / "sam3.pt"),
    }
    for old, new in replacements.items():
        text = text.replace(old, new)

    # Per-job hyperparameter overrides (simple line patches; the trainer
    # has no CLI override support, so we mutate the YAML directly).
    if args.max_epochs is not None:
        text = _replace_yaml_scalar(text, "max_epochs:", args.max_epochs)
    if args.train_batch_size is not None:
        text = _replace_yaml_scalar(text, "train_batch_size:", args.train_batch_size)
    if args.lr_scale is not None:
        text = _replace_yaml_scalar(text, "lr_scale:", args.lr_scale)
    if args.num_train_workers is not None:
        text = _replace_yaml_scalar(text, "num_train_workers:", args.num_train_workers)

    # src == dst (both point at _sm.yaml). Explicit here so a future
    # reader doesn't wonder about the mismatch.
    out = src
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    print(f"[train_entry] wrote derived config -> {out}")
    return out


def _replace_yaml_scalar(text: str, key: str, value) -> str:
    """Replace the first '  key: <old>' line with the supplied value. Naive
    line-based patcher — fine for top-level scalars (max_epochs, lr_scale)."""
    out_lines = []
    replaced = False
    for line in text.splitlines():
        if not replaced and key in line:
            indent = line.split(key)[0]
            out_lines.append(f"{indent}{key} {value}")
            replaced = True
        else:
            out_lines.append(line)
    if not replaced:
        print(f"[train_entry] WARNING: key {key!r} not found in template")
    return "\n".join(out_lines) + "\n"


def _resolve_multinode_env() -> dict:
    """Read SageMaker's multi-node env vars and translate them into the
    `SAM3_*` vars our train.py::single_node_runner expects.

    SageMaker sets these on every training instance:
        SM_HOSTS         JSON list, e.g. '["algo-1","algo-2"]'
        SM_CURRENT_HOST  this instance's hostname, e.g. "algo-2"
        SM_NUM_HOSTS     integer stringified

    Single-instance jobs still get these (with a length-1 host list),
    so the logic below degrades cleanly to single-node.

    Master is always the FIRST host in SM_HOSTS (sorted lexicographically
    by SageMaker), so every worker resolves the same rendezvous target.
    """
    hosts_json = os.environ.get("SM_HOSTS")
    current = os.environ.get("SM_CURRENT_HOST")
    if not hosts_json or not current:
        # Not running under SageMaker (or single-node local dev) —
        # leave envs unset, single_node_runner falls back to localhost.
        return {}
    try:
        hosts = json.loads(hosts_json)
    except json.JSONDecodeError as e:
        print(f"[train_entry] malformed SM_HOSTS ({e}); falling back to single-node")
        return {}
    if not isinstance(hosts, list) or not hosts:
        return {}

    master = hosts[0]
    num_nodes = len(hosts)
    try:
        node_rank = hosts.index(current)
    except ValueError:
        print(
            f"[train_entry] SM_CURRENT_HOST={current!r} not in SM_HOSTS={hosts!r}; "
            "assuming node_rank=0"
        )
        node_rank = 0

    env_out = {
        "SAM3_MASTER_ADDR": master,
        "SAM3_MASTER_PORT": os.environ.get("SAM3_MASTER_PORT", "29500"),
        "SAM3_NODE_RANK":   str(node_rank),
        "SAM3_NUM_NODES":   str(num_nodes),
    }
    if num_nodes > 1:
        print(
            f"[train_entry] multi-node: {node_rank+1}/{num_nodes} — "
            f"master={master}:{env_out['SAM3_MASTER_PORT']} "
            f"(this host: {current})"
        )
    return env_out


def run_training(config_path: Path, args: argparse.Namespace) -> None:
    """Invoke sam3/train/train.py with the derived config."""
    # train.py calls initialize_config_module("sam3.train", ...), so
    # Hydra's config search path is rooted at pkg://sam3.train (the
    # Python package dir). Config names passed to -c must include the
    # "configs/" prefix so Hydra can resolve them under sam3/train/
    # configs/.
    relative = config_path.relative_to(Path("sam3/train"))

    # Detect multi-node topology from SageMaker's SM_HOSTS / SM_CURRENT_HOST.
    # Adds SAM3_MASTER_ADDR / SAM3_NODE_RANK / SAM3_NUM_NODES to the env
    # we hand to train.py.
    multinode_env = _resolve_multinode_env()
    num_nodes = int(multinode_env.get("SAM3_NUM_NODES", "1"))

    cmd = [
        sys.executable, "-u", "sam3/train/train.py",  # -u = unbuffered stdout/stderr
        "-c", str(relative),
        "--use-cluster", "0",
        "--num-gpus", str(args.num_gpus),
        "--num-nodes", str(num_nodes),
    ]
    print(f"[train_entry] launching: {' '.join(cmd)}")

    # Turn up torch-side logging so a rank that aborts silently (e.g. rank
    # 4 SIGABRT during spawn) surfaces its real error site instead of just
    # the parent's ProcessExitedException. These add ~2 lines/rank at
    # startup, negligible during steady-state training.
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("PYTHONFAULTHANDLER", "1")
    env.setdefault("TORCH_SHOW_CPP_STACKTRACES", "1")
    env.setdefault("TORCH_CPP_LOG_LEVEL", "INFO")
    # NCCL_DEBUG is inert on the gloo backend but harmless.
    env.setdefault("NCCL_DEBUG", "INFO")

    # Force allocator fragmentation off. The SageMaker training toolkit
    # sometimes doesn't forward `environment={}` from the estimator all
    # the way through `mp.spawn`'s child inherit path, so set it here
    # explicitly for the training subprocess (this env is `subprocess.run`'d
    # with cmd, then mp.spawn inherits from *this* env).
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:512"

    # Prevent ghost CUDA contexts on GPU 0.
    #
    # When N ranks are spawned by mp.spawn, each child inherits the
    # parent's CUDA state. If ANYTHING in the parent (importing torch,
    # a transitive that reads torch.cuda.is_available(), a logging
    # sanity check) has touched CUDA before torch.multiprocessing.spawn
    # forks, every child inherits a context on GPU 0 in addition to
    # its own local_rank device. On the observed p4de OOM, all 7
    # sibling ranks held ~414 MB on GPU 0 = ~3 GB of stolen headroom
    # before training even started.
    #
    # CUDA_DEVICE_MAX_CONNECTIONS=1 + lazy init keeps children from
    # materializing contexts on non-owned devices. And CUDA_VISIBLE_DEVICES
    # per-rank is set inside single_proc_run itself (train.py:44), but
    # only *after* the child has already imported torch.
    env.setdefault("CUDA_DEVICE_MAX_CONNECTIONS", "1")
    # Delay CUDA init until torch.cuda is actually used. Without this,
    # the parent process's CUDA context (materialized during `import
    # torch` on some builds) gets inherited by every child fork.
    env.setdefault("CUDA_MODULE_LOADING", "LAZY")

    # Prepend the wheel-bundled cuDNN to LD_LIBRARY_PATH.
    #
    # The SageMaker PyTorch base image ships a system cuDNN at
    # /lib/x86_64-linux-gnu/libcudnn_graph.so.9 that predates the
    # cudnnGetLibConfig symbol our cu128 torch>=2.7 wheel needs. On the
    # first conv forward, the loader picks the system copy, fails a
    # symbol lookup, and calls abort() — visible in CloudWatch as:
    #   "Could not load symbol cudnnGetLibConfig ..."
    #   "Fatal Python error: Aborted" @ conv.py:548 in _conv_forward
    # nvidia-cudnn-cu12 (pulled in transitively by the torch wheel)
    # installs its own libcudnn*.so.9 under site-packages/nvidia/cudnn/lib.
    # Prepending that directory to LD_LIBRARY_PATH resolves the graph
    # frontend against the wheel's cuDNN instead of the system one.
    cudnn_lib = _find_bundled_cudnn_dir()
    if cudnn_lib is not None:
        prev = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = f"{cudnn_lib}:{prev}" if prev else cudnn_lib
        print(f"[train_entry] LD_LIBRARY_PATH prepended with {cudnn_lib}")
    else:
        print("[train_entry] WARNING: bundled nvidia-cudnn-cu12 not found; conv may abort")

    # Multi-node rendezvous info (empty dict when single-node, so this
    # is a no-op there). SAM3_* takes precedence over any legacy
    # MASTER_ADDR / RANK values that might already be in os.environ.
    env.update(multinode_env)

    subprocess.run(cmd, check=True, env=env)


def _find_bundled_cudnn_dir() -> str | None:
    """Return the site-packages path where nvidia-cudnn-cu12 installed
    its shared libraries, or None if the package isn't installed.

    `nvidia.cudnn` is an implicit namespace package (no __init__.py), so
    __file__ is None; we have to walk __path__ instead.
    """
    try:
        import nvidia.cudnn  # type: ignore[import-not-found]
    except ModuleNotFoundError:
        return None
    for base in getattr(nvidia.cudnn, "__path__", []):
        lib = Path(base) / "lib"
        if lib.is_dir():
            return str(lib.resolve())
    return None


def merge_and_export(args: argparse.Namespace) -> None:
    """After training, merge the trainer's checkpoint with the pretrained
    weights and stage the result for SageMaker to upload.

    Reads from SM_CHECKPOINT_DIR (=/opt/ml/checkpoints/), matching where
    write_runtime_config remaps experiment_log_dir. That's also the tree
    SageMaker's checkpoint sync watches, so the raw per-epoch files are
    already in S3 by the time this runs — the merge step just produces
    the self-contained artifact for model.tar.gz.
    """
    del args  # currently unused; kept for symmetry with the other hooks
    ckpt_dir = SM_CHECKPOINT_DIR / "aws_sam_finetune" / "checkpoints"
    raw_ckpt = ckpt_dir / "checkpoint.pt"
    if not raw_ckpt.exists():
        print(f"[train_entry] WARNING: no checkpoint at {raw_ckpt} — skipping merge")
        return
    pretrained = SM_CHANNEL_PRETRAINED / "sam3.pt"
    merged = SM_MODEL_DIR / "checkpoint_merged.pt"
    SM_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "scripts/finetune/merge_checkpoint.py",
        "--finetuned", str(raw_ckpt),
        "--pretrained", str(pretrained),
        "--output", str(merged),
    ]
    print(f"[train_entry] merging: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    print(f"[train_entry] merged checkpoint -> {merged}")

    # Also copy the BPE vocab so the model artifact is self-contained.
    bpe = Path("sam3/assets/bpe_simple_vocab_16e6.txt.gz")
    if bpe.exists():
        shutil.copy(bpe, SM_MODEL_DIR / bpe.name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-template",
        # Read from `_sm.yaml` so hand-edits in that file stick on
        # SageMaker (matching local behavior). write_runtime_config
        # then patches SageMaker-channel paths + any HP overrides
        # in-place, so `_sm.yaml` acts as both the input and output
        # of the derivation step — the container reads it, rewrites
        # it, and then invokes train.py against it.
        default="sam3/train/configs/aws_sam/aws_sam_finetune_sm.yaml",
        help="Path to the template Hydra config to derive from.",
    )
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--train-batch-size", type=int, default=None)
    parser.add_argument("--lr-scale", type=float, default=None)
    parser.add_argument("--num-train-workers", type=int, default=None)
    parser.add_argument(
        "--skip-install", action="store_true",
        help="Skip pip install (useful when the base image already has SAM3 baked in).",
    )
    parser.add_argument(
        "--skip-merge", action="store_true",
        help="Skip the merge step (export raw checkpoint.pt instead).",
    )
    args, _ = parser.parse_known_args()

    print("[train_entry] SageMaker training job started")
    print(f"[train_entry] SM_MODEL_DIR={SM_MODEL_DIR}")
    print(f"[train_entry] SM_CHECKPOINT_DIR={SM_CHECKPOINT_DIR}")
    print(f"[train_entry] SM_CHANNEL_TRAIN={SM_CHANNEL_TRAIN}")
    print(f"[train_entry] SM_CHANNEL_PRETRAINED={SM_CHANNEL_PRETRAINED}")

    ensure_shm_capacity()
    if not args.skip_install:
        install_runtime_deps()

    cfg_path = write_runtime_config(args)
    run_training(cfg_path, args)

    if args.skip_merge:
        # Just copy the raw checkpoint to the model dir.
        raw = SM_CHECKPOINT_DIR / "aws_sam_finetune" / "checkpoints" / "checkpoint.pt"
        if raw.exists():
            SM_MODEL_DIR.mkdir(parents=True, exist_ok=True)
            shutil.copy(raw, SM_MODEL_DIR / raw.name)
    else:
        merge_and_export(args)

    print("[train_entry] done")


if __name__ == "__main__":
    main()
