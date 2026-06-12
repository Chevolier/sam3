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
pip install -e ".[dev,train]"
pip install -e ".[notebooks]"   # PIL is in numpy/jupyter, but pin versions if missing
hf auth login                    # required: training pulls facebook/sam3 from HF
```

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
    --min-count 20
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
python sam3/train/train.py \
    -c configs/aws_sam/aws_sam_finetune.yaml \
    --use-cluster 0 \
    --num-gpus 1
```

### Multi-GPU local run

```bash
python sam3/train/train.py \
    -c configs/aws_sam/aws_sam_finetune.yaml \
    --use-cluster 0 \
    --num-gpus 4
```

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

---

## 4. Eval-only against the test split

```bash
python sam3/train/train.py \
    -c configs/aws_sam/aws_sam_eval.yaml \
    --use-cluster 0 --num-gpus 1 \
    trainer.checkpoint.resume_from=runs/aws_sam_finetune/checkpoints/checkpoint.pt
```

The eval config inherits from the fine-tune one and just flips
`trainer.mode: val`. AP is printed and the per-category dump lands in
`${experiment_log_dir}/dumps/aws_sam/`.

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
    checkpoints/checkpoint.pt
    tensorboard/
    logs/
    dumps/aws_sam/<predictions>.json
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
