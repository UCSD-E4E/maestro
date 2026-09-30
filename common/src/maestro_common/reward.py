"""Per-image reward: how well one model's instances match a person's annotation.

Predicted and annotated instances are matched one-to-one by mask IoU with the
Hungarian algorithm. Pairs below ``match_threshold`` do not count as matches.
Unmatched predictions (false positives) and unmatched annotations (misses)
contribute 0, so the score is

    sum of matched IoUs / max(#predictions, #annotations)

which lies in [0, 1]. An image with nothing predicted and nothing annotated
scores 1. Matching is class-agnostic: v1 has a single "fish" class (PLAN.md).
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

from maestro_common.instances import Instance, stack_masks

MATCH_THRESHOLD = 0.5


def iou_matrix(pred_masks: np.ndarray, gt_masks: np.ndarray) -> np.ndarray:
    """IoU between every predicted and every annotated mask, shape (n_pred, n_gt)."""
    if len(pred_masks) == 0 or len(gt_masks) == 0:
        return np.zeros((len(pred_masks), len(gt_masks)))
    pred = pred_masks.reshape(len(pred_masks), -1).astype(np.float32)
    gt = gt_masks.reshape(len(gt_masks), -1).astype(np.float32)
    intersection = pred @ gt.T
    union = pred.sum(1)[:, None] + gt.sum(1)[None, :] - intersection
    return np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)


def match(
    predictions: list[Instance],
    annotations: list[Instance],
    match_threshold: float = MATCH_THRESHOLD,
) -> list[tuple[int, int, float]]:
    """One-to-one matches as (prediction index, annotation index, IoU)."""
    if not predictions or not annotations:
        return []
    shape = annotations[0].mask.shape
    ious = iou_matrix(stack_masks(predictions, shape), stack_masks(annotations, shape))
    rows, cols = linear_sum_assignment(ious, maximize=True)
    return [(r, c, float(ious[r, c])) for r, c in zip(rows, cols) if ious[r, c] >= match_threshold]


def image_iou(
    predictions: list[Instance],
    annotations: list[Instance],
    match_threshold: float = MATCH_THRESHOLD,
) -> float:
    """The per-image reward in [0, 1] described in the module docstring."""
    denominator = max(len(predictions), len(annotations))
    if denominator == 0:
        return 1.0
    return sum(iou for _, _, iou in match(predictions, annotations, match_threshold)) / denominator
