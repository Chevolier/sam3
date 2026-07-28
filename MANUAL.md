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
| **C** — click-only | `aws_sam_finetune_click.yaml` + overrides | click predictor (+ ViT) only | 1-/3-click, no text prompt |

Recipe C is B with the text-side loss weights zeroed and those modules
frozen — see "Click-only fine-tuning" below. There is no separate config.

Recipe B ports SAM-1's multi-step click loss into SAM3 (`ClickMaskLoss` —
best-of-3 mask selection + focal + dice + MSE on IoU head). Weights:
`loss_click_mask: 200`, `loss_click_dice: 10`, `loss_click_iou: 1`.
Adds ~30% memory; drop `train_batch_size` if tight.

Both recipes fine-tune the full 840M model including the ViT and text
encoder — see "Freeze the pretrained encoders" below to train only the
task-specific parts instead.

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

**Click-only fine-tuning (no text supervision)**

Recipe B trains the click branch *and* the text-conditioned detector
together. To train **only** the click path, zero out every text-side loss
weight and freeze the text-side modules. Nothing needs to change in the
forward pass — the text tower still runs (`Sam3Image.forward` always calls
`backbone.forward_text`, `sam3_image.py:562`), it just receives no gradient.

```bash
python sam3/train/train.py -c configs/aws_sam/aws_sam_finetune_click.yaml \
    --use-cluster 0 --num-gpus 8 \
    paths.experiment_log_dir='${oc.env:PWD}'/runs/aws_sam_clickonly \
    aws_sam_train.loss.loss_fns_find.0.weight_dict.loss_bbox=0.0 \
    aws_sam_train.loss.loss_fns_find.0.weight_dict.loss_giou=0.0 \
    aws_sam_train.loss.loss_fns_find.1.weight_dict.loss_ce=0.0 \
    aws_sam_train.loss.loss_fns_find.1.weight_dict.presence_loss=0.0 \
    aws_sam_train.loss.loss_fns_find.2.weight_dict.loss_mask=0.0 \
    aws_sam_train.loss.loss_fns_find.2.weight_dict.loss_dice=0.0 \
    aws_sam_train.loss.loss_fn_semantic_seg.weight_dict.loss_semantic_seg=0.0 \
    aws_sam_train.loss.loss_fn_semantic_seg.weight_dict.loss_semantic_presence=0.0 \
    aws_sam_train.loss.loss_fn_semantic_seg.weight_dict.loss_semantic_dice=0.0 \
    trainer.model._target_=sam3.model_builder.build_sam3_image_model_with_freeze \
    '+trainer.model.freeze_patterns=[backbone.language_backbone.,backbone.vision_backbone.convs.,transformer.,geometry_encoder.,segmentation_head.,dot_prod_scoring.,class_embed.]'
```

The list indices `loss_fns_find.{0,1,2}` are positional — `0`=`Boxes`,
`1`=`IABCEMdetr`, `2`=`Masks`, `3`=`ClickMaskLoss` (left at full weight).
Check the config if you reorder them. `reduce_loss` skips any key whose
weight is `0` (`loss_fns.py:260`), so the zeroed heads still *run* (cheap
tensor ops on already-computed logits) but contribute nothing and propagate
no gradient.

`class_embed.` in the freeze list is a harmless no-op — the builder sets
`use_dot_prod_scoring=True`, so that head is never constructed. Freeze
patterns that match nothing are silently ignored, so check the
`[freeze] froze N/M parameter tensors` log line to confirm your set landed.

| Freeze set | Trainable | % of 860.1M |
|---|---|---|
| text-side frozen, ViT trainable | 458.1M | 53.3% |
| + `backbone.vision_backbone.trunk.` | 11.9M | 1.4% |

With the ViT trainable, the encoder adapts to your imagery through the click
loss alone: `vision_backbone.trunk` 446.2M + `sam2_convs` 7.8M +
`inst_interactive_predictor.model` 4.1M. Add the trunk to `freeze_patterns`
and you train just `sam2_convs` + the click predictor's prompt encoder and
mask decoder — fast and very low-memory, but the ViT can't adapt, so expect
smaller gains. Start with the ViT trainable unless memory forces otherwise.

Two things to know:

- **Keep `enable_segmentation: True`.** The zeroed `Masks` / semantic-seg
  losses don't need it, but `enable_inst_interactivity` shares the FPN and
  disabling the seg head changes which modules get built — a needless
  divergence from the checkpoint layout the merge step (§3) expects.
- **Evaluate with click metrics only.** The config's `meters.val` runs a
  COCO *bbox* evaluator, which now measures an untrained detection head.
  Its numbers will be poor and that's expected. Judge these runs by
  `Losses/train_all_loss_click_mask` / `_dice` / `_iou` in TensorBoard and
  by §4 with `--skip-zero-click`.

Also worth doing: drop `scratch.lr_vision_backbone` further (the ViT is now
the *only* large trainable block, and the click loss is a much weaker signal
than the full detection objective), and raise `scratch.train_batch_size`
since the frozen text tower and DETR transformer no longer retain
activations.

**Freeze the pretrained encoders** (train only the fusion encoder / DETR
decoder / seg head / geometry encoder / FPN necks / click predictor).

Both recipes fine-tune *everything* by default — the ViT and text tower are
not frozen, just given smaller LRs (`lr_vision_backbone` 2.5e-5,
`lr_language_backbone` 5e-6, plus layer-decay 0.9 on the ViT trunk). To
freeze them instead, swap the builder for `build_sam3_image_model_with_freeze`
and pass name prefixes — no config edit needed:

```bash
python sam3/train/train.py -c configs/aws_sam/aws_sam_finetune_click.yaml \
    --use-cluster 0 --num-gpus 8 \
    trainer.model._target_=sam3.model_builder.build_sam3_image_model_with_freeze \
    '+trainer.model.freeze_patterns=[backbone.vision_backbone.trunk.,backbone.language_backbone.encoder.]'
```

For recipe A add `trainer.distributed.find_unused_parameters=False
+trainer.distributed.static_graph=True` (recipe B already sets both). DDP
drops `requires_grad=False` params from its reducer, so this isn't strictly
required, but it's faster once the trainable set is fixed.

| Recipe | Total | Trainable | % |
|---|---|---|---|
| A — text-only | 840.5M | 40.8M | 4.9% |
| B — + click branch | 860.1M | 52.7M | 6.1% |

Trainable in B: `transformer` 21.1M, `geometry_encoder` 8.2M,
`vision_backbone.convs` 7.8M, `vision_backbone.sam2_convs` 7.8M,
`inst_interactive_predictor` 4.1M, `segmentation_head` 2.3M,
`dot_prod_scoring` 1.2M, `language_backbone.resizer` 0.26M.

The trailing dots matter — `backbone.language_backbone.encoder.` freezes the
24-layer text tower but leaves the `.resizer` 1024→256 adapter trainable;
likewise the SimpleFPN `convs` stay trainable under
`backbone.vision_backbone.trunk.`. Both are projections into `d_model`, not
part of the pretrained encoders proper. Matching is `fnmatch` *or*
`startswith` (`model_builder.py:759`), so bare prefixes work and globs are
optional.

Raise `scratch.train_batch_size` afterwards — frozen subtrees don't retain
activations for backward, which is where most of the memory goes.

Two things freezing does *not* turn off: the ViT keeps `drop_path_rate=0.1`
(`model_builder.py:88`) and the text tower keeps gradient checkpointing
(`text_encoder_ve.py:266`), both gated on `.training` rather than
`requires_grad`. So frozen encoder features still vary run-to-run (can't
cache them across epochs) and the text tower still recomputes its forward
for nothing. Harmless for convergence; fixing either needs a builder arg.

The LR groups for `backbone.vision_backbone.*` / `backbone.language_backbone.*`
then govern almost nothing (just the necks and `resizer`), and
`layer_decay_param_modifier` becomes a no-op. Both are harmless —
`validate_param_group_params` only requires full coverage, not that every
param is trainable.

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
