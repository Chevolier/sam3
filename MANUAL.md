# Fine-tuning SAM3 on AWS_SAM

End-to-end recipe for fine-tuning SAM3 on the local LabelMe dataset under
`data/AWS_SAM/`. Three stages: **prepare → fine-tune → merge → eval/deploy**.
All commands assume repo root as CWD.

---

## 0. Install

```bash
pip install --index-url https://download.pytorch.org/whl/cu128 torch torchvision
pip install -e ".[dev,train,notebooks]"
pip install einops psutil
```

**Pretrained weights**: loaded from `/home/ec2-user/SageMaker/efs/Models/sam3/sam3.pt`
by default. Set `paths.sam3_pretrained_ckpt=null` + `trainer.model.load_from_HF=True`
+ `hf auth login` to pull `facebook/sam3` instead.

Hardware: single GPU ≥ 24 GB is enough at `train_batch_size=1`, resolution 1008.

---

## 1. Prepare the dataset

```bash
python scripts/finetune/prepare_aws_sam.py \
    --data-dir data/AWS_SAM \
    --out-dir  data/AWS_SAM_split \
    --train-frac 0.9 --seed 42 --min-count 20
```

Walks `data/AWS_SAM/`, drops labels below `--min-count`, writes:

```
data/AWS_SAM_split/
  train.json test.json categories.json
```

COCO `file_name` is relative to `data/AWS_SAM/` — no file copies.
Expected: `4684 records → labels 66 / 112 → train=4216 test=468`.

**Sanity-check the conversion**:

```bash
python scripts/finetune/visualize_test_masks.py \
    --coco       data/AWS_SAM_split/test.json \
    --image-root data/AWS_SAM \
    --out-dir    data/AWS_SAM_split/test_vis \
    --num 20 --seed 0
```

---

## 2. Fine-tune

Two recipes, both under `sam3/train/configs/aws_sam/`:

| Recipe | Config | Trains | Best for |
|---|---|---|---|
| **A** — text-only | `aws_sam_finetune.yaml` | detector + segmentation head | 0-click / text prompt |
| **B** — + click branch | `aws_sam_finetune_click.yaml` | A + SAM-1-style click predictor | 1-/3-click interactive |

Recipe B ports SAM-1's multi-step click loss into SAM3 (`ClickMaskLoss` —
best-of-3 mask selection + focal + dice + MSE on IoU head). Weights:
`loss_click_mask: 200`, `loss_click_dice: 10`, `loss_click_iou: 1`.
Adds ~30% memory; drop `train_batch_size` if tight.

**Local run**:

```bash
python sam3/train/train.py \
    -c configs/aws_sam/aws_sam_finetune.yaml \
    --use-cluster 0 --num-gpus 8
# or _click.yaml for recipe B
```

**SageMaker**: open `sagemaker/train/launch_training.ipynb`, edit S3 /
instance / HP cells, Run All. Multi-instance DDP works via `INSTANCE_COUNT>1`
on the `multi-instance-train` branch.

**Path / hyperparameter overrides** (Hydra dotted-key):

```bash
python sam3/train/train.py -c configs/aws_sam/aws_sam_finetune.yaml \
    paths.aws_sam_image_root=/abs/path/AWS_SAM \
    scratch.train_batch_size=2 trainer.max_epochs=40
```

Auto-resume from `${experiment_log_dir}/checkpoints/`. Explicit resume:
`trainer.checkpoint.resume_from=/path/to/checkpoint.pt`.

**TensorBoard**: `tensorboard --logdir runs/aws_sam_finetune/tensorboard --host 0.0.0.0 --port 6006`.
Watch `Losses/train_all_miou_semantic_seg` — the cleanest health signal
(0-1 scale, should climb steadily; raw `train_all_loss` is ~300 due to
loss weights and hard to read).

---

## 3. Merge the checkpoint (mandatory after training)

The trainer only saves parameters of modules it built. Recipe A skips
the SAM-1 click predictor + SAM-2 tracker → those keys are missing from
`checkpoint.pt` → inference builds them with random init → mIoU collapses
to ~0.02. Merge fixes this by copying missing keys from the pretrained file:

```bash
python scripts/finetune/merge_checkpoint.py \
    --finetuned  runs/aws_sam_finetune_click/checkpoints/checkpoint_8.pt \
    --pretrained /home/ec2-user/SageMaker/efs/Models/sam3/sam3.pt \
    --output     runs/aws_sam_finetune_click/checkpoints/checkpoint_8_merged.pt
```

`checkpoint_merged.pt` is the **canonical artifact** for eval / compare /
deploy. Re-run after every training job. SageMaker's `train_entry.py`
does this automatically.

---

## 4. Evaluate — mIoU / Boundary IoU / NoC@95

`scripts/finetune/eval/evaluate_interactive.py` runs each test instance
through 3 settings and reports:

| Metric | Description |
|---|---|
| **mIoU @ k** | Mean IoU after k clicks (k ∈ {0, 1, 3}) |
| **Boundary IoU @ k** | IoU on a boundary band (Cheng et al. 2021) |
| **NoC@95** | Mean clicks to reach IoU ≥ 0.95, capped at 20 |

```bash
# Pretrained baseline
nohup python -u scripts/finetune/eval/evaluate_interactive.py \
      --checkpoint runs/aws_sam_finetune_sm/checkpoint_7.pt \
      --pretrained-fallback /home/ec2-user/SageMaker/efs/Models/sam3/sam3.pt \
      --coco data/AWS_SAM_split/test.json --image-root data/AWS_SAM \
      --dump-predictions \
      --output runs/eval/finetuned_sm_ckpt7.json > logs/finetuned_sm_ckpt7.out 2>&1 &

# Markdown comparison table
python scripts/finetune/eval/compare_results.py \
    --pretrained runs/eval/pretrained.json \
    --finetuned  runs/eval/finetuned.json \
    --out-md     runs/eval/compare.md
```

Useful flags: `--max-instances N` (quick smoke test), `--max-per-image N`
(balance across images), `--min-area 200` (drop tiny GTs), `--skip-zero-click`
(halves runtime if you only need click metrics).

---

## 5. Interactive web compare-app

Qualitative side-by-side of GT / pretrained / fine-tuned:

```bash
pip install fastapi uvicorn pydantic
python scripts/finetune/compare_app/server.py \
    --pretrained-ckpt /home/ec2-user/SageMaker/efs/Models/sam3/sam3.pt \
    --finetuned-ckpt  runs/aws_sam_finetune/checkpoints/checkpoint_merged.pt \
    --port 8080
```
runs/aws_sam_finetune_click/checkpoints/checkpoint_8_merged.pt

SSH tunnel: `ssh -L 8080:localhost:8080 <ec2-host>` → open `http://localhost:8080`.
Both models stay warm; image embedding cached per upload.

---

## 6. SageMaker real-time endpoint

Full workflow: `sagemaker/deploy/launch_deploy.ipynb`. It vendors the
current fork's `sam3/` source into the tarball (private-repo safe, avoids
upstream state-dict mismatch), timestamps each endpoint for zero-collision
redeploys, and includes text+click smoke-test cells.

Handler (`sagemaker/deploy/inference.py`) accepts JSON in two modes:

```jsonc
// Text mode
{"mode": "text", "image_b64": "...", "text": "grass", "confidence": 0.5}

// Click mode (SAM-1 compatible)
{"mode": "click", "image_b64": "...", "points": [[x,y]], "labels": [1],
 "multimask_output": true}
```

Response: `masks_rle` (COCO-RLE, decode with `pycocotools.mask.decode`),
`scores`, `boxes`, `image_size`.

**Verify a deployed endpoint matches local weights**:

```bash
python scripts/finetune/eval/compare_local_vs_endpoint.py \
    --checkpoint runs/aws_sam_finetune/checkpoints/checkpoint_merged.pt \
    --endpoint   sam3-aws-sam-<STAMP> \
    --image      data/AWS_SAM/companypremises2025101600217.png \
    --text       grass --confidence 0.5 \
    --output     runs/eval/compare
```

Should print IoU ≈ 0.999 per pair — anything lower indicates a
state-dict mismatch or the container is running upstream `sam3`.

Recommended instance: `ml.g5.xlarge` (1 × A10G, 24 GB) — warm p50 ~250 ms
for a 1008×1008 text prompt.

---

## 7. Benchmark (local GPU and deployed endpoint)

**Local GPU** — `benchmark_throughput.py` reports p50/p90/p95/p99 for
`set_image` (ViT embedding), text prompt (full + amortized), and click_n:

```bash
python scripts/finetune/benchmark/benchmark_throughput.py \
    --checkpoint runs/aws_sam_finetune/checkpoints/checkpoint_merged.pt \
    --image-dir data/AWS_SAM \
    --num-iters 50 --warmup 5 --click-points 1 3 5 \
    --output runs/bench/finetuned.json
```

**Deployed endpoint** — `benchmark_endpoint.py` covers end-to-end latency
including network + TLS, plus sustained QPS under `--concurrency`:

```bash
python scripts/finetune/benchmark/benchmark_endpoint.py \
    --endpoint sam3-aws-sam-<STAMP> \
    --num-iters 100 --warmup 5 --concurrency 4 --modes text click \
    --output runs/bench/endpoint.json
```

---

## File layout produced

```
data/AWS_SAM_split/{train,test,categories}.json + test_vis/
runs/
  aws_sam_finetune/checkpoints/{checkpoint,checkpoint_merged}.pt
  aws_sam_finetune/{tensorboard,logs,dumps}/
  aws_sam_finetune_click/…                      # recipe B
  eval/{pretrained,finetuned}.json + compare.md
  bench/{pretrained,finetuned,endpoint}.json
```

---

## Troubleshooting

- **`No module named 'hydra'`** — `pip install -e ".[train]"`.
- **HF auth error** — SAM3 weights are gated; `hf auth login`.
- **OOM at resolution 1008 on 24 GB** — keep `train_batch_size=1`, or
  drop to `scratch.resolution=672` (must stay ÷14).
- **"No find queries" spam at startup** — normal; empty category chunks
  get filtered and the loader retries.
- **Eval mIoU catastrophically low (~0.02)** — you're loading an unmerged
  `checkpoint.pt`. Merge it (§3) or pass `--pretrained-fallback`.
- **SageMaker endpoint returns 0 masks / first invoke times out** —
  ~30-60 s cold start. Use the notebook's warm-up + 300 s read timeout.
  Verify state-dict match with `compare_local_vs_endpoint.py`.
- **Benchmark variance high** — raise `--warmup` to 20; pin GPU clock
  via `nvidia-smi -lgc <freq>` for reproducible numbers.
