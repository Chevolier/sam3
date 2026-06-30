# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved
# pyre-unsafe
"""Click-prompt training loss for SAM3's interactive predictor.

This is a port of SAM 2's MultiStepMultiMasksAndIous, originally written
inside this repo at sam3/train/loss/loss_fns.py:782 (commented out) but
never wired into the training pipeline. It supervises:

  1. The mask decoder's per-pixel mask logits via focal BCE + Dice.
  2. The IoU-prediction head via MSE/L1 vs the actually-computed IoU
     between the predicted mask and GT.

The "multistep" terminology refers to SAM-style iterative click training:
at each training step, multiple clicks are accumulated (seed click, then
correction clicks on the worst-error region), and the loss is computed
for the mask produced after each click. We back-propagate through every
step of the iteration.

When `multimask_output=True`, the predictor emits 3 candidate masks per
click step. We pick the candidate with the **lowest focal + dice loss**
vs GT and supervise only that one (and its corresponding iou_prediction
slot). This is the standard SAM training recipe — using the best
candidate avoids forcing the model to learn ambiguity from a single
click.

The expected `outputs` dict from the click branch (see Sam3Image's
training-time click forward):

    outputs["click_multistep_pred_multimasks_high_res"]   list of (N, M, H, W)
    outputs["click_multistep_pred_ious"]                  list of (N, M)
    outputs["click_multistep_object_score_logits"]        list of (N, 1) [optional]

where N = total GT instances across the batch, M = 3 if multimask, 1
otherwise. The list has one entry per click step (1, 3, etc.).

The expected `targets` dict (built by Sam3Image.back_convert):

    targets["masks"]            (N, H, W) bool GT instance masks
    targets["is_valid_mask"]    (N,)      bool validity flags
"""

from __future__ import annotations

import torch

from .loss_fns import (
    CORE_LOSS_KEY,
    LossWithWeights,
    dice_loss,
    iou_loss,
    sigmoid_focal_loss,
)


class ClickMaskLoss(LossWithWeights):
    """Click-prompt loss head for SAM3's interactive predictor.

    Args:
      weight_dict: keys "loss_click_mask", "loss_click_dice", "loss_click_iou"
        (and optionally "loss_click_class" if pred_obj_scores=True).
      compute_aux: forwarded to LossWithWeights; usually False for this
        head since the click branch doesn't itself have aux decoder layers.
      focal_alpha, focal_gamma: focal-BCE knobs on the mask logits.
      supervise_all_iou: if True, supervise iou_predictions on every
        candidate mask. If False (default), only on the best-loss
        candidate (the SAM convention).
      iou_use_l1_loss: use L1 instead of MSE on iou_predictions.
      pred_obj_scores: if True, supervise object_score_logits (SAM 2's
        "is the click on a real object" head). Our setup doesn't use it.
    """

    def __init__(
        self,
        weight_dict=None,
        compute_aux=False,
        focal_alpha=0.25,
        focal_gamma=2,
        supervise_all_iou=False,
        iou_use_l1_loss=False,
        pred_obj_scores=False,
        focal_gamma_obj_score=0.0,
        focal_alpha_obj_score=-1,
    ):
        super().__init__(weight_dict, compute_aux, supports_o2m_loss=False)
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.target_keys.extend(["masks", "is_valid_mask"])
        # Sanity-check required weight_dict keys at construction time so
        # config typos surface immediately, not 200 iters into training.
        for required in ("loss_click_mask", "loss_click_dice", "loss_click_iou"):
            if required not in self.weight_dict:
                raise ValueError(
                    f"ClickMaskLoss requires weight_dict[{required!r}]; "
                    f"got {list(self.weight_dict.keys())}"
                )
        if pred_obj_scores and "loss_click_class" not in self.weight_dict:
            self.weight_dict["loss_click_class"] = 0.0
        self.focal_alpha_obj_score = focal_alpha_obj_score
        self.focal_gamma_obj_score = focal_gamma_obj_score
        self.supervise_all_iou = supervise_all_iou
        self.iou_use_l1_loss = iou_use_l1_loss
        self.pred_obj_scores = pred_obj_scores

    def get_loss(self, outputs, targets, indices=None, num_boxes=None):
        """
        Computes per-pixel mask losses on the predicted click masks +
        MSE on iou_predictions. Iterates over click steps.

        `indices` and `num_boxes` are part of the standard LossWithWeights
        interface but the click branch operates on its own pre-aligned
        (N, ...) tensors, so we don't use them directly. `num_boxes` is
        used as the per-instance normalization scalar.
        """
        # The click branch may be skipped this step (e.g., no GT instances
        # in the batch). Return zero losses in that case so the wrapper's
        # weighted sum still produces a valid scalar.
        if (
            "click_multistep_pred_multimasks_high_res" not in outputs
            or len(outputs["click_multistep_pred_multimasks_high_res"]) == 0
        ):
            zero = torch.zeros((), device=targets["masks"].device if "masks" in targets else "cpu")
            return {
                "loss_click_mask": zero,
                "loss_click_dice": zero,
                "loss_click_iou": zero,
                **({"loss_click_class": zero} if self.pred_obj_scores else {}),
            }

        target_masks = targets["masks"].unsqueeze(1).float()
        if target_masks.dim() != 4:
            raise ValueError(
                f"targets['masks'] should be (N, H, W); got shape {target_masks.shape[:-1]}"
            )

        src_masks_list = outputs["click_multistep_pred_multimasks_high_res"]
        ious_list = outputs["click_multistep_pred_ious"]
        object_score_logits_list = outputs.get(
            "click_multistep_object_score_logits",
            # Default to length-matched list of None so the loop runs uniformly.
            [None] * len(src_masks_list),
        )

        if len(src_masks_list) != len(ious_list):
            raise ValueError(
                f"click outputs length mismatch: "
                f"masks={len(src_masks_list)} ious={len(ious_list)}"
            )

        # Only count valid (non-padded) instances toward the loss.
        keep = targets.get("is_valid_mask")
        if keep is not None:
            target_masks = target_masks[keep]

        losses = {
            "loss_click_mask": 0.0,
            "loss_click_dice": 0.0,
            "loss_click_iou": 0.0,
            "loss_click_class": 0.0,
        }
        # num_boxes is the normalization scalar used by sigmoid_focal_loss /
        # dice_loss / iou_loss. Use the per-step instance count if not given.
        n_steps = len(src_masks_list)
        for step_masks, step_ious, step_obj_logits in zip(
            src_masks_list, ious_list, object_score_logits_list
        ):
            if keep is not None:
                step_masks = step_masks[keep]
                step_ious = step_ious[keep]
                if step_obj_logits is not None:
                    step_obj_logits = step_obj_logits[keep]
            self._accumulate(
                losses,
                step_masks,
                target_masks,
                step_ious,
                step_obj_logits,
                num_boxes=num_boxes if num_boxes is not None else max(target_masks.size(0), 1),
            )

        # Average over click steps so loss scale is independent of how
        # many clicks per instance we chose to simulate.
        for k in losses:
            losses[k] = losses[k] / max(n_steps, 1)

        if not self.pred_obj_scores:
            losses.pop("loss_click_class")
        return losses

    def _accumulate(
        self,
        losses,
        src_masks,        # (N, M, H, W)  predicted mask logits at full res
        target_masks,     # (N, 1, H, W)  GT binary masks
        ious,             # (N, M)        predicted IoUs
        object_score_logits,  # (N, 1) or None
        num_boxes,
    ):
        # Broadcast GT to all M candidates so per-candidate losses align.
        target_masks_bcast = target_masks.expand_as(src_masks)

        loss_multimask = sigmoid_focal_loss(
            src_masks,
            target_masks_bcast,
            num_boxes,
            alpha=self.focal_alpha,
            gamma=self.focal_gamma,
            loss_on_multimask=True,
        )
        loss_multidice = dice_loss(
            src_masks, target_masks_bcast, num_boxes, loss_on_multimask=True
        )
        loss_multiiou = iou_loss(
            src_masks,
            target_masks_bcast,
            ious,
            num_boxes,
            loss_on_multimask=True,
            use_l1_loss=self.iou_use_l1_loss,
        )
        assert loss_multimask.dim() == 2  # (N, M)
        assert loss_multidice.dim() == 2
        assert loss_multiiou.dim() == 2

        if loss_multimask.size(1) > 1:
            # Pick the candidate with lowest focal+dice loss; supervise
            # iou_predictions at the same index. Standard SAM trick to
            # avoid forcing the model to memorize ambiguity.
            loss_combo = (
                loss_multimask * self.weight_dict["loss_click_mask"]
                + loss_multidice * self.weight_dict["loss_click_dice"]
            )
            best_loss_inds = torch.argmin(loss_combo, dim=-1)
            batch_inds = torch.arange(loss_combo.size(0), device=loss_combo.device)
            loss_mask = loss_multimask[batch_inds, best_loss_inds].unsqueeze(1)
            loss_dice = loss_multidice[batch_inds, best_loss_inds].unsqueeze(1)
            if self.supervise_all_iou:
                loss_iou = loss_multiiou.mean(dim=-1).unsqueeze(1)
            else:
                loss_iou = loss_multiiou[batch_inds, best_loss_inds].unsqueeze(1)
        else:
            loss_mask = loss_multimask
            loss_dice = loss_multidice
            loss_iou = loss_multiiou

        # Object-score head (optional; off in our config).
        if self.pred_obj_scores and object_score_logits is not None:
            target_obj = torch.any(
                (target_masks[:, 0] > 0).flatten(1), dim=-1
            )[..., None].float()
            loss_class = sigmoid_focal_loss(
                object_score_logits,
                target_obj,
                num_boxes,
                alpha=self.focal_alpha_obj_score,
                gamma=self.focal_gamma_obj_score,
            )
            # Gate per-pixel losses on presence — only count loss where an
            # object exists, so empty instances don't drown out real ones.
            loss_mask = loss_mask * target_obj
            loss_dice = loss_dice * target_obj
            loss_iou = loss_iou * target_obj
            losses["loss_click_class"] = losses["loss_click_class"] + loss_class.sum()

        losses["loss_click_mask"] = losses["loss_click_mask"] + loss_mask.sum()
        losses["loss_click_dice"] = losses["loss_click_dice"] + loss_dice.sum()
        losses["loss_click_iou"] = losses["loss_click_iou"] + loss_iou.sum()
