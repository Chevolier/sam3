# Fine-tuning SAM3 on AWS_SAM

End-to-end recipe for fine-tuning the SAM3 image model on the local
LabelMe-format dataset under `data/AWS_SAM/`.

The pipeline has three stages:

1. **Prepare** — convert LabelMe per-image JSONs to two COCO files
   (train.json / test.json) at a 0.9 / 0.1 split.
2. **Visualize** — sample 20 images from the test split and overlay the
   ground-truth polygon masks (sanity check on the conversion).
3. **Fine-tune** — run `sam3/train/train.py` with the new
   `configs/aws_sam/aws_sam_finetune.yaml` config, which reuses SAM3's
   existing `Sam3ImageDataset` + `COCO_FROM_JSON` machinery and enables the
   detection + mask + semantic-segmentation losses.

All commands assume the repo root is the working directory.

---

## 0. Install

```bash
pip install --index-url https://download.pytorch.org/whl/cu128 \
      torch torchvision
pip install -e ".[dev,train]"
pip install -e ".[notebooks]"   # PIL is in numpy/jupyter, but pin versions if missing

pip install einops psutil
```

### Pretrained weights

By default this workflow loads SAM3 weights from a local checkpoint at
`/home/ec2-user/SageMaker/efs/Models/sam3/sam3.pt` — no HuggingFace
download needed. Override per-run:

- **Training**: `paths.sam3_pretrained_ckpt=/abs/path/to/sam3.pt`
- **Eval / benchmark**: `--checkpoint /abs/path/to/sam3.pt`
- **SageMaker deploy**: `--checkpoint /abs/path/to/sam3.pt`

If you want to pull `facebook/sam3` from HuggingFace instead, set the
training config `paths.sam3_pretrained_ckpt=null` and
`trainer.model.load_from_HF=True`, and run `hf auth login` first. For the
eval/benchmark scripts, omit `--checkpoint` entirely.

Hardware: a single GPU with ≥ 24 GB VRAM is enough at `train_batch_size=1`,
resolution 1008. Multi-GPU works the same way — just bump `--num-gpus`.

---

## 1. Prepare the dataset (LabelMe → COCO + 0.9/0.1 split)

```bash
python scripts/finetune/prepare_aws_sam.py \
    --data-dir data/AWS_SAM \
    --out-dir  data/AWS_SAM_split \
    --train-frac 0.9 \
    --seed 42 \
    --min-count 1
```

What it does:

- Walks `data/AWS_SAM/`, pairs every `*.json` (LabelMe export) with its
  matching image (`.jpg`/`.png`/`.jpeg`/`.bmp`/`.tif`).
- Drops categories with fewer than `--min-count` instances dataset-wide
  (raise this for a smaller, cleaner label set; lower it to keep the long
  tail).
- Writes:

  ```
  data/AWS_SAM_split/
    train.json        # COCO: ~90 % of images, polygon segmentations
    test.json         # COCO: ~10 % of images
    categories.json   # label_to_id + per-label instance counts (audit file)
  ```

- COCO `file_name` is **relative** to `data/AWS_SAM/`; the trainer uses
  that folder as `img_folder` (no copies, no symlinks).

The split is deterministic given `--seed`. Re-running with the same seed
reproduces the same split.

Expected output (current dataset):

```
[prepare_aws_sam] paired records: 4684
[prepare_aws_sam] labels kept (>= 20 instances): 66 / 112 total
[prepare_aws_sam] split: train=4216 test=468
```

---

## 2. Visualize ground-truth masks on 20 test images

```bash
python scripts/finetune/visualize_test_masks.py \
    --coco       data/AWS_SAM_split/test.json \
    --image-root data/AWS_SAM \
    --out-dir    data/AWS_SAM_split/test_vis \
    --num 20 \
    --seed 0
```

Renders 20 randomly-sampled test images with each polygon filled in a
per-category color and labeled at its centroid. Inspect a few before
training — if labels are misplaced or polygons look wrong, the conversion
needs investigation.

---

## 3. Fine-tune

The config lives at `sam3/train/configs/aws_sam/aws_sam_finetune.yaml`.
Hydra resolves `-c` relative to `sam3/train/configs/`.

### Single-GPU local run

```bash
nohup python sam3/train/train.py \
    -c configs/aws_sam/aws_sam_finetune.yaml \
    --use-cluster 0 \
    --num-gpus 8 > logs/train2.out 2>&1 &
```

### Multi-GPU local run

```bash
python sam3/train/train.py \
    -c configs/aws_sam/aws_sam_finetune.yaml \
    --use-cluster 0 \
    --num-gpus 8
```

### Click-prompt fine-tune (SAM-2-style)

The default `aws_sam_finetune.yaml` builds the model with
`enable_inst_interactivity=False`, so the SAM-1-style click predictor is
**not trained** — only its text/detector pathway gets gradient signal.
This means click-mode metrics (NoC@95, mIoU @ 1 click) gain only
indirectly through the shared image encoder. To get the full click-mode
benefit, use the new preset:

```bash
nohup python sam3/train/train.py \
    -c configs/aws_sam/aws_sam_finetune_click.yaml \
    --use-cluster 0 \
    --num-gpus 8 > logs/train_click.out 2>&1 &
```

What's different from the base recipe:

- `enable_inst_interactivity: True` builds the `SAM3InteractiveImagePredictor`
  submodule (~150 M params) at training time.
- `Sam3Image._forward_click_branch_train` runs inside `forward()` per step:
  samples a seed click at each GT centroid, then 2 correction clicks at the
  worst-error region of the previous step's prediction, calls the mask
  decoder iteratively, and returns lists of `(N, M, H, W)` multistep mask
  logits + `(N, M)` IoU predictions.
- `ClickMaskLoss` (`sam3.train.loss.click_loss.ClickMaskLoss`) is added
  to `loss_fns_find`. Supervises best-of-3 candidate mask with focal+dice
  and the iou_predictions head with MSE. Weights: `loss_click_mask: 200`,
  `loss_click_dice: 10`, `loss_click_iou: 1`.
- Default `train_batch_size: 4` (vs 8 in the base preset) since the click
  branch adds ~30% memory pressure per step. Tune up if you have headroom.

The merge step (§3.5) works identically — `merge_checkpoint.py` produces
a `checkpoint_merged.pt` that the eval / compare-app / deploy scripts
consume without changes.

### Remote training on SageMaker

`scripts/finetune/sagemaker/launch_sagemaker_training.ipynb` packages the
repo + this config and launches a SageMaker TrainingJob. The entry-point
script (`train_entry.py`) patches the YAML to use SageMaker channel paths,
runs `sam3/train/train.py`, then merges the resulting checkpoint into a
self-contained `checkpoint_merged.pt` that SageMaker uploads to S3.
Use this when you want managed multi-GPU training without the on-prem
GPUs. Open the notebook, edit the AWS / S3 / instance settings in the
first cell, then run all.

### Override paths from the command line

By default the config reads `${oc.env:PWD}/data/AWS_SAM`,
`${oc.env:PWD}/data/AWS_SAM_split`, etc. — i.e. the dataset is expected at
`<repo>/data/AWS_SAM` when you launch from the repo root. Override per-run
with Hydra dotted-key syntax:

```bash
python sam3/train/train.py \
    -c configs/aws_sam/aws_sam_finetune.yaml \
    --use-cluster 0 --num-gpus 1 \
    paths.aws_sam_image_root=/abs/path/to/AWS_SAM \
    paths.aws_sam_split_dir=/abs/path/to/AWS_SAM_split \
    paths.experiment_log_dir=/abs/path/to/run_dir \
    scratch.train_batch_size=2 \
    trainer.max_epochs=40
```

### What the config actually does

- **Model** — `sam3.model_builder.build_sam3_image_model` is instantiated
  with `eval_mode=false` and `enable_segmentation=true`. Because
  `load_from_HF=True` is the default and `checkpoint_path` is unset, SAM3's
  pretrained weights from HuggingFace `facebook/sam3` are loaded as the
  starting point — that's the "fine-tune" part. Run `hf auth login` first.
- **Data** — `Sam3ImageDataset` with `COCO_FROM_JSON`. Polygon
  segmentations from the prep step are converted to RLE on the fly by
  `ann_to_rle` and decoded inside the training transform pipeline
  (`segmentation.DecodeRle`).
- **Losses** — bbox + GIoU + IABCE classification + presence + mask
  (focal + dice) + semantic-segmentation. This is the "with segmentation"
  block from the roboflow config, which is the reference setup for joint
  detection + segmentation training.
- **Optimizer** — AdamW with three LR groups (transformer / vision
  backbone / language backbone) and inverse-square-root schedules; vision
  backbone uses 0.9 layer-decay. All LRs are scaled by `scratch.lr_scale =
  0.1` (full-FT defaults are 10× higher; we drop to 0.1× for fine-tuning).
- **Eval** — every 5 epochs against `test.json`, COCO bbox AP via
  `CocoEvaluatorOfflineWithPredFileEvaluators`. Predictions are dumped to
  `${experiment_log_dir}/dumps/aws_sam/`.
- **Checkpoints** — written to `${experiment_log_dir}/checkpoints/`.
  Default `save_freq: 0` keeps only the last epoch's checkpoint; raise it
  to keep periodic snapshots.

### Resume / re-run

The trainer auto-resumes from the latest checkpoint in
`${experiment_log_dir}/checkpoints/` when the directory exists. To
explicitly resume from another file, override:

```bash
... trainer.checkpoint.resume_from=/path/to/checkpoint.pt
```

### Bbox-only (skip mask losses)

Open `aws_sam_finetune.yaml`, set `scratch.enable_segmentation: False`,
and remove the `Masks` and `loss_fn_semantic_seg` entries from
`aws_sam_train.loss` (or copy the bbox-only loss block from
`configs/roboflow_v100/roboflow_v100_full_ft_100_images.yaml` verbatim).
Keeps memory ~30 % lower.

### Monitor training with TensorBoard

The trainer writes scalars (per-component losses, LR per group, gradient
norms, iter time) to `${experiment_log_dir}/tensorboard/` every
`log_freq: 10` iterations. To watch them live:

```bash
pip install tensorboard   # ships with the [train] extra
tensorboard --logdir runs/aws_sam_finetune/tensorboard \
    --host 0.0.0.0 --port 6006
```

Then open `http://<host>:6006/`. From a laptop with SSH:

```bash
ssh -L 6006:localhost:6006 <ec2-host>
open http://localhost:6006
```

What to look at:

| Tag prefix | What it tells you |
|---|---|
| `Losses/train_all_loss` | Overall objective. Headline number; should trend down. |
| `Losses/train_all_loss_bbox`, `loss_giou` | Detection box quality. |
| `Losses/train_all_loss_ce`, `presence_loss` | Classification & presence head. |
| `Losses/train_all_loss_mask`, `loss_dice` | Per-instance mask quality. |
| `Losses/train_all_loss_semantic_seg`, `loss_semantic_dice` | Semantic-seg head. |
| `Losses/train_all_miou_semantic_seg` | **The cleanest sanity-check signal** — running mIoU of the seg head on the training batch. Should climb steadily. |
| `Trainer/where` | Fraction of the schedule completed (0 → 1 over `max_epochs`). |
| `Trainer/epoch`, `steps_train` | Position counters. |

Practical tips:

- Watch `train_all_miou_semantic_seg` rather than the raw `train_all_loss`
  — the total is heavily weighted (200 × mask + 30 × semantic_dice + …)
  and looks alarming in absolute terms (≈ 300 is normal). mIoU is on a
  [0, 1] scale and is the better health signal.
- TensorBoard auto-refreshes; no need to restart on each epoch.
- Multiple runs side-by-side: keep `experiment_log_dir` distinct per
  run, then point `tensorboard --logdir runs/` at the parent directory.

---

## 3.5 Merge the checkpoint (do this once after training)

The trainer only saves parameters of modules it actually built. Because
training runs with `enable_inst_interactivity=False` (default), the
SAM-1 click predictor, the SAM-2 neck convs, and the tracker get
**omitted** from `checkpoint.pt`. Inference code that builds those
modules will fill them with random init → catastrophic eval regression
(observed mIoU ≈ 0.02 vs. 0.57 pretrained).

`scripts/finetune/merge_checkpoint.py` produces a self-contained file:

```bash
python scripts/finetune/merge_checkpoint.py \
    --finetuned  runs/aws_sam_finetune/checkpoints/checkpoint.pt \
    --pretrained /home/ec2-user/SageMaker/efs/Models/sam3/sam3.pt \
    --output     runs/aws_sam_finetune/checkpoints/checkpoint_merged.pt
```

Expected output:

```
[merge]   pretrained keys: ~1465
[merge]   fine-tuned keys: 1134
[merge] result: total=~1465, updated_from_ft=1134, kept_from_pretrained=~331, only_in_ft=0
[merge] wrote runs/aws_sam_finetune/checkpoints/checkpoint_merged.pt
```

After this step, **`checkpoint_merged.pt` is the canonical artifact**
to pass to eval, compare-app, benchmark, and SageMaker deploy. The
unmerged `checkpoint.pt` is incomplete on its own.

Re-run the merge after every fresh training run (or wrap both into a
shell script):

```bash
# scripts/finetune/train_and_merge.sh
python sam3/train/train.py -c configs/aws_sam/aws_sam_finetune.yaml \
    --use-cluster 0 --num-gpus 8
python scripts/finetune/merge_checkpoint.py \
    --finetuned  runs/aws_sam_finetune/checkpoints/checkpoint.pt \
    --pretrained /home/ec2-user/SageMaker/efs/Models/sam3/sam3.pt \
    --output     runs/aws_sam_finetune/checkpoints/checkpoint_merged.pt
```

---

## 4. Eval-only against the test split

```bash
python sam3/train/train.py \
    -c configs/aws_sam/aws_sam_eval.yaml \
    --use-cluster 0 --num-gpus 1 \
    trainer.checkpoint.resume_from=runs/aws_sam_finetune/checkpoints/checkpoint_merged.pt
```

The eval config inherits from the fine-tune one and just flips
`trainer.mode: val`. AP is printed and the per-category dump lands in
`${experiment_log_dir}/dumps/aws_sam/`.

---

---

## 4.1 Interactive web UI: side-by-side compare

For a qualitative side-by-side comparison of the pretrained and fine-tuned
models on arbitrary images / prompts, run:

```bash
pip install fastapi uvicorn pydantic   # if not already installed

python scripts/finetune/compare_app/server.py \
    --pretrained-ckpt /home/ec2-user/SageMaker/efs/Models/sam3/sam3.pt \
    --finetuned-ckpt  runs/aws_sam_finetune/checkpoints/checkpoint_merged.pt \
    --port 8080
```

Then open `http://<host>:8080/`. From a laptop with SSH:

```bash
ssh -L 8080:localhost:8080 <ec2-host>
# then on the laptop:
open http://localhost:8080
```

The UI lets you:

- Upload any image.
- **Text mode**: type a noun phrase (e.g. `obstacle`, `grass`, `trunk`) and
  see what each model returns. Adjust the confidence threshold inline.
- **Click mode**: click points on the image (positive/negative label
  selectable) and watch both models' masks update side-by-side.

Both models stay warm on the GPU; the image embedding is cached per
upload, so additional prompts on the same image take ~tens of ms.

---

## 5. Interactive evaluation: mIoU / Boundary IoU / NoC@95

`scripts/finetune/eval/evaluate_interactive.py` runs each test instance
through the same model under three settings — 0 corrective clicks (text
prompt only), 1 click (positive click at the GT centroid), and 3 clicks
(positive seed + 2 corrections placed on the worst-error region) — then
reports:

| Metric | Description |
|---|---|
| **mIoU @ k clicks** | mean Intersection-over-Union after k clicks (k ∈ {0, 1, 3}) |
| **Boundary IoU @ k clicks** | IoU restricted to a band along the mask boundary (Cheng et al., CVPR 2021), dilation = 2 % of the image diagonal |
| **NoC@95** | mean number of clicks needed to reach IoU ≥ 0.95, capped at 20 |

### Run pretrained vs. fine-tuned

```bash
# Pretrained (local SAM3 weights)
python scripts/finetune/eval/evaluate_interactive.py \
    --coco data/AWS_SAM_split/test.json \
    --image-root data/AWS_SAM \
    --checkpoint /home/ec2-user/SageMaker/efs/Models/sam3/sam3.pt \
    --output runs/eval/pretrained.json

# Fine-tuned weights from the run above (use the MERGED checkpoint — see §3.5)
python scripts/finetune/eval/evaluate_interactive.py \
    --coco data/AWS_SAM_split/test.json \
    --image-root data/AWS_SAM \
    --checkpoint runs/aws_sam_finetune/checkpoints/checkpoint_merged.pt \
    --output runs/eval/finetuned.json

# Alternative: if you haven't merged, point at the raw checkpoint and
# supply the pretrained file as fallback for modules training omitted.
python scripts/finetune/eval/evaluate_interactive.py \
    --coco data/AWS_SAM_split/test.json \
    --image-root data/AWS_SAM \
    --checkpoint           runs/aws_sam_finetune_click/checkpoints/checkpoint_1.pt \
    --pretrained-fallback  /home/ec2-user/SageMaker/efs/Models/sam3/sam3.pt \
    --output               runs/eval/finetuned_click_ckpt1.json

# Side-by-side markdown table
python scripts/finetune/eval/compare_results.py \
    --pretrained runs/eval/pretrained.json \
    --finetuned  runs/eval/finetuned.json \
    --out-md     runs/eval/compare.md
```

Omit `--checkpoint` to fall back to a HuggingFace download of
`facebook/sam3`. The two checkpoints share the same state-dict layout
so swapping is just a path change.

Useful flags:

- `--max-instances N` — total cap (use 100–500 for a quick smoke test).
- `--max-per-image N` — keeps eval balanced across images instead of
  letting a few very crowded scenes dominate.
- `--min-area 200` — drops tiny GTs (default 200 px²) where boundary IoU
  is meaningless.
- `--skip-zero-click` — skip the text-prompt path if you only care about
  click metrics (saves ~half the wall-time).
- `--max-clicks 20 --iou-target 0.95` — defaults match the standard
  interactive-segmentation protocol.

The output JSON contains both the per-instance breakdown and an
aggregated summary; `compare_results.py` only reads the summary.

---

## 6. SageMaker real-time endpoint

`scripts/finetune/deploy/` contains a SageMaker PyTorch deployment for
both the pretrained and the fine-tuned model. The handler
(`inference.py`) accepts JSON requests in two modes — text prompt or
SAM-1-style click prompt — and returns COCO-RLE encoded masks.

### Prerequisites

```bash
pip install sagemaker boto3
aws configure                # or assume an IAM role with sagemaker:*
```

### Deploy

```bash
# Fine-tuned model — use the MERGED checkpoint (see §3.5) so the
# container has every parameter the inference handler needs.
python scripts/finetune/deploy/deploy.py \
    --checkpoint runs/aws_sam_finetune/checkpoints/checkpoint_merged.pt \
    --role arn:aws:iam::<acct>:role/SageMakerRole \
    --bucket <my-sagemaker-bucket> \
    --prefix sam3/aws_sam_v1 \
    --instance-type ml.g5.xlarge \
    --endpoint-name sam3-aws-sam

# Pretrained (downloads facebook/sam3 inside the container on cold start)
python scripts/finetune/deploy/deploy.py --pretrained --role <ARN> --bucket <BUCKET>
```

What it does:

1. Builds `runs/deploy/model.tar.gz` containing
   `checkpoint.pt`, the BPE vocab, and `code/{inference.py,requirements.txt}`.
2. Uploads it to S3.
3. Creates a `PyTorchModel` and `.deploy(...)` an endpoint.

### Invoke

```bash
# Text prompt
python scripts/finetune/deploy/invoke_example.py \
    --endpoint sam3-aws-sam \
    --image data/AWS_SAM/companypremises2025101600217.png \
    --text grass

# Click prompt
python scripts/finetune/deploy/invoke_example.py \
    --endpoint sam3-aws-sam --image <path> --click 520 375
```

Request schemas, including the optional `box` and `multimask_output`
fields, are documented at the top of `inference.py`. Masks come back as
COCO-RLE strings that `pycocotools.mask.decode(rle)` turns into binary
arrays.

Recommended instance: `ml.g5.xlarge` (1 × A10G, 24 GB) for the standard
SAM3 model at 1008 resolution. Use `ml.g5.2xlarge` if you also need
client-side preprocessing on the same host.

---

## 7. Throughput / latency benchmark

`scripts/finetune/benchmark/benchmark_throughput.py` separately measures
the three components of an inference call:

- **set_image** — image preprocessing + ViT embedding (the dominant cost).
- **text** — `set_image` + `set_text_prompt`, both in full-pipeline form
  and "amortized" (prompt only, with the embedding excluded — what
  matters when the same image is queried repeatedly).
- **click_n** — `set_image` + `predict_inst` with N positive clicks
  (sweeps N via `--click-points`).

Stats reported per setting: median + p50/p90/p95/p99 latency, mean,
throughput per second.

```bash
# Pretrained (local SAM3 weights)
python scripts/finetune/benchmark/benchmark_throughput.py \
    --image-dir data/AWS_SAM \
    --checkpoint /home/ec2-user/SageMaker/efs/Models/sam3/sam3.pt \
    --num-iters 50 --warmup 5 \
    --click-points 1 3 5 \
    --output runs/bench/pretrained.json

# Fine-tuned (use the merged checkpoint — see §3.5)
python scripts/finetune/benchmark/benchmark_throughput.py \
    --checkpoint runs/aws_sam_finetune/checkpoints/checkpoint_merged.pt \
    --output runs/bench/finetuned.json
```

Skip flags: `--skip-set-image`, `--skip-text`, `--skip-click` if you
only want a subset.

The benchmark inserts `torch.cuda.synchronize()` around every timed
region — numbers reflect device wall-time, not the host-side launch
queue.

---

## File layout produced by this workflow

```
data/
  AWS_SAM/                          # original LabelMe images + JSONs (unchanged)
  AWS_SAM_split/
    train.json                      # COCO
    test.json                       # COCO
    categories.json                 # audit
    test_vis/<stem>_vis.jpg         # 20 ground-truth overlays
runs/
  aws_sam_finetune/                 # default ${paths.experiment_log_dir}
    checkpoints/
      checkpoint.pt                 # raw trainer output (incomplete on its own)
      checkpoint_merged.pt          # ← canonical artifact, see §3.5 — use this
    tensorboard/
    logs/
    dumps/aws_sam/<predictions>.json
  eval/
    pretrained.json finetuned.json compare.md
  bench/
    pretrained.json finetuned.json
  deploy/
    model.tar.gz                    # bundle uploaded to S3
```

## Troubleshooting

- **`No module named 'hydra'`** — `[train]` extras weren't installed. Run
  `pip install -e ".[train]"`.
- **HF auth error on first run** — SAM3 weights are gated. `hf auth login`
  with a token that has access to `facebook/sam3`.
- **OOM at resolution 1008 on a 24 GB GPU** — keep `train_batch_size=1`,
  or override `scratch.resolution=672` (still divisible by 14, the ViT
  patch stride).
- **Many "No find queries" errors at startup** — normal: the dataset
  groups categories into chunks (see `scratch.collate_fn` /
  `category_chunk_size`); chunks with no GT instances are filtered by
  `FilterEmptyTargets` and the loader retries the next index.
- **Polygons look wrong in the visualizer** — re-run `prepare_aws_sam.py`
  and check `categories.json` instance counts; LabelMe sometimes exports
  empty `points` arrays (we skip those automatically).
- **Eval is too slow** — start with `--max-instances 200` to verify the
  pipeline, then scale up. The 0-click path is the most expensive
  (text prompts run a full DETR forward); pass `--skip-zero-click` if
  you only need click metrics.
- **SageMaker container OOMs on first request** — `facebook/sam3` is
  ~2 GB; on cold start the HF download competes with model loading.
  Either bundle the checkpoint via `--checkpoint` or pre-warm by
  invoking the endpoint once and waiting ~60 s before real traffic.
- **Benchmark numbers swing wildly** — raise `--warmup` (default 5,
  bump to 20 for a fresh CUDA context) and pin the GPU via
  `nvidia-smi -lgc <freq>` if you need clock-stable measurements.
