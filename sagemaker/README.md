# SAM3 on SageMaker

Everything you need to fine-tune or deploy SAM3 on AWS SageMaker,
grouped by workflow.

## Layout

```
sagemaker/
├── README.md                       ← you are here
├── train/
│   ├── launch_training.ipynb       Notebook: submits a SageMaker TrainingJob
│   ├── train_entry.py              Container entry point (runs INSIDE the job)
│   └── build_notebook.py           Regenerates the launcher notebook from Python
└── deploy/
    ├── deploy.py                   CLI: package a checkpoint + launch an endpoint
    ├── inference.py                Container handler for the real-time endpoint
    ├── invoke_example.py           Smoke-test client for a deployed endpoint
    └── requirements.txt            Container deps installed at endpoint boot
```

## Training

Open `sagemaker/train/launch_training.ipynb` from Jupyter, edit the
config cell (S3 bucket / prefix / instance type), and run all cells.

Under the hood:

1. **Config cell** — checks the SageMaker SDK version (pins v2), reads
   the IAM role, picks up the default bucket.
2. **Upload cells** — one-time push of `data/AWS_SAM*` and the local
   `sam3.pt` pretrained checkpoint to S3.
3. **Staging cell** — builds a slim source-bundle directory
   (`~/sam3_sagemaker_staging`) containing just `sam3/`, `sagemaker/`,
   `scripts/`, and package metadata. **This is important**: SDK v2 tars
   `source_dir` in full, and pointing it at the repo root would upload
   the entire `data/` and `runs/` trees — silently hangs the launch.
4. **Estimator + fit** — constructs `sagemaker.pytorch.PyTorch` pointing
   at the staging dir and `sagemaker/train/train_entry.py`, then calls
   `.fit()` with the two S3 channels.

Inside the container, `train_entry.py`:

1. `pip install -e .` (installs SAM3 + train extras).
2. Rewrites the Hydra YAML paths to point at `/opt/ml/input/data/train/*`
   (the SageMaker channels) and the pretrained file.
3. Invokes `python sam3/train/train.py -c configs/aws_sam/aws_sam_finetune_sm.yaml`.
4. After training, calls `scripts/finetune/merge_checkpoint.py` to
   produce a self-contained `checkpoint_merged.pt` and writes it to
   `/opt/ml/model/` (which SageMaker uploads to
   `s3://<bucket>/<prefix>/output/model.tar.gz`).

## Deployment

Once training has produced a merged checkpoint, deploy it as a real-time
endpoint:

```bash
python sagemaker/deploy/deploy.py \
    --checkpoint runs/aws_sam_finetune/checkpoints/checkpoint_merged.pt \
    --role arn:aws:iam::<acct>:role/SageMakerRole \
    --bucket <my-sagemaker-bucket> \
    --prefix sam3/aws_sam_v1 \
    --instance-type ml.g5.xlarge \
    --endpoint-name sam3-aws-sam
```

The handler at `sagemaker/deploy/inference.py` accepts both text-prompt
and click-prompt JSON requests; masks come back as COCO-RLE strings.
See `sagemaker/deploy/invoke_example.py` for a working client.

## Notes

- The launcher notebook targets **SageMaker SDK v2**. The first cell
  pins `sagemaker>=2.230,<3` because the v3 SDK is a major rewrite that
  moves `PyTorch` estimator + `Session` + `get_execution_role` out of
  their canonical paths.
- `train_entry.py` and `inference.py` are **container-side** scripts —
  they run inside the SageMaker instance, not on your laptop.
- If you change `train_entry.py`, no rebuild is needed — it ships in
  the source bundle on every `.fit()` call.
- For details of the training config itself, see
  `sam3/train/configs/aws_sam/aws_sam_finetune_sm.yaml` and the
  discussion in `MANUAL.md`.
