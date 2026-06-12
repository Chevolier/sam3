# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

SAM 3 (Segment Anything with Concepts) is a unified foundation model for promptable segmentation in images and videos using text or visual prompts (points, boxes, masks). The repo also contains SAM 3.1 ("Object Multiplex"), a shared-memory variant for joint multi-object tracking. Both ship as a single installable package `sam3`.

Requires Python ≥ 3.12, PyTorch ≥ 2.7, CUDA ≥ 12.6.

## Common commands

```bash
# Install (editable) with dev + train deps; add `notebooks` to run examples/*.ipynb
pip install -e ".[dev,train]"
pip install -e ".[notebooks]"   # decord, matplotlib, ipycanvas, ... — required by examples/

# Optional fast inference path
pip install einops ninja && pip install flash-attn-3 --no-deps --index-url https://download.pytorch.org/whl/cu128
pip install git+https://github.com/ronghanghu/cc_torch.git

# Format (CI uses ufmt with ruff-api 0.1.0, black 24.2.0, usort 1.0.2 over `sam3 scripts`)
ufmt format .

# Tests (only directory currently is `test/`; pyproject's testpaths default to `tests`)
pytest test/
pytest test/test_io_utils.py::TestLoadVideoFramesRouting::test_mp4_extension_routes_to_video_loader

# Train / eval (Hydra config-driven, single entry point)
python sam3/train/train.py -c configs/roboflow_v100/roboflow_v100_full_ft_100_images.yaml
python sam3/train/train.py -c configs/odinw13/odinw_text_only.yaml          # eval (trainer.mode=val in config)
python sam3/train/train.py -c CONFIG --use-cluster 0 --num-gpus 4            # local multi-GPU
python sam3/train/train.py -c CONFIG --use-cluster 1 --partition P --num-nodes 2  # SLURM via submitit
```

`-c` paths are resolved relative to `sam3/train/configs/` by Hydra (`initialize_config_module`). See `README_TRAIN.md` for the full launcher arg list and `experiment_log_dir` layout.

## Architecture

**Model assembly is centralized in `sam3/model_builder.py`.** It is the canonical entry point for users; it wires every component manually (no Hydra at inference time) and is the file to read first when tracing how parts connect. Public builders:

- `build_sam3_image_model(...)` → `Sam3Image` — DETR-like detector conditioned on text/geometry/exemplars.
- `build_sam3_video_model(...)` → `Sam3VideoInferenceWithInstanceInteractivity` (SAM 3 video).
- `build_sam3_video_predictor(...)` → `Sam3VideoPredictorMultiGPU` wrapper around the above.
- `build_sam3_multiplex_video_predictor(...)` → `Sam3MultiplexVideoPredictor` (SAM 3.1).
- `build_sam3_predictor(version="sam3"|"sam3.1", ...)` → unified entry; both versions share the same `handle_request` / `handle_stream_request` API (`start_session`, `add_prompt`, `propagate_in_video`, `remove_object`, `reset_session`, `close_session`).

Checkpoints auto-download from HuggingFace (`facebook/sam3`, `facebook/sam3.1`) via `download_ckpt_from_hf`; users must `hf auth login` first. Local checkpoints with internal `sam3_model.` / `sam2_predictor.` key prefixes are remapped to OSS `detector.` / `tracker.` on load (see `build_sam3_multiplex_video_predictor` for the remap logic).

### Component layout (`sam3/`)

- `model/` — all inference modules.
  - **Detector path:** `vitdet.py` (ViT trunk) → `necks.py` (`Sam3DualViTDetNeck` / `Sam3TriViTDetNeck`) → `vl_combiner.py` (vision+text fusion) → `encoder.py` + `decoder.py` (DETR transformer with presence token) → `maskformer_segmentation.py` (`UniversalSegmentationHead`, `PixelDecoder`).
  - **Tracker path (SAM 2-style):** `sam3_tracker_base.py`, `sam3_tracker_utils.py`, `sam3_tracking_predictor.py`, `memory.py` (mask memory encoder).
  - **Video orchestration:** `sam3_video_base.py`, `sam3_video_inference.py`, `sam3_video_predictor.py` (multi-GPU split: detector and tracker can live on different GPUs).
  - **Multiplex (SAM 3.1):** `multiplex_utils.py`, `sam3_multiplex_*.py`, `video_tracking_multiplex*.py`. Multiplex groups objects into fixed-capacity buckets so memory attention is shared across them — see Appendix H of the paper.
  - **Text:** `text_encoder_ve.py` + `tokenizer_ve.py` (BPE vocab shipped at `sam3/assets/bpe_simple_vocab_16e6.txt.gz`).
  - **Image entry points:** `sam3_image.py` (`Sam3Image`, `Sam3ImageOnVideoMultiGPU`), `sam3_image_processor.py` (`Sam3Processor` with `set_image`/`set_text_prompt`/`add_geometric_prompt`), `sam1_task_predictor.py` (interactive SAM 1-style point/box prompting).
  - `io_utils.py` — video frame loading; routes by extension (mp4/mov go through `load_video_frames_from_video_file`, JPEG dirs through a separate path).
- `sam/` — SAM 2 transformer/RoPE/mask-decoder primitives reused by the tracker.
- `perflib/` — performance kernels (Triton NMS / connected components, FA3, fused ops). `perflib/compile.py` centralizes `torch.compile` modes.
- `train/` — Hydra-driven training/eval; `train.py` is launcher (single-node `mp.spawn` or `submitit` SLURM), `trainer.py` is the loop, `configs/` holds yaml configs grouped by task (`roboflow_v100/`, `odinw13/`, `gold_image_evals/`, `silver_image_evals/`, `saco_video_evals/`).
- `eval/` — metric implementations: `cgf1_eval.py` (official SA-Co metric), `coco_eval*.py`, `saco_veval_*.py`, vendored `hota_eval_toolkit/` and `teta_eval_toolkit/`.
- `agent/` — SAM 3 Agent (LLM tool-use over SAM 3); see `examples/sam3_agent.ipynb`.
- `scripts/eval/{gold,silver,veval}/` — dataset prep + standalone benchmark runners with their own READMEs (SA-Co/Gold, SA-Co/Silver, SA-Co/VEval).
- `examples/*.ipynb` — canonical usage references; read these before reverse-engineering call sites. Notable: `sam3_image_predictor_example`, `sam3_video_predictor_example`, `sam3.1_video_predictor_example`, `sam3_for_sam1_task_example` / `sam3_for_sam2_video_task_example` (SAM 1/2 compatibility shims), `sam3_agent`, and the four `saco_{gold_silver,veval}_{eval,vis}_example` notebooks for benchmarking.

### Conventions worth knowing

- `# pyre-unsafe` headers are intentional (Meta-internal Pyre type-checker) — do not remove.
- Files often gate optional accelerated paths with `use_fa3` (FlashAttention-3) and `use_rope_real` (real-valued RoPE for `torch.compile` compat). SAM 3.1 defaults both to `True`; SAM 3 base predictor does not.
- Image size is fixed at 1008 throughout; ViT patch size 14, stride 14, internal feature size 72×72.
- The "presence token" in `decoder.py` is the discriminative head described in the paper — preserve it when modifying the decoder.
