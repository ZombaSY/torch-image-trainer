"""Shared helpers: reproducibility, logging, segmentation metrics. **PLACEHOLDER.**

``set_seed``, ``run_timestamp`` and ``get_logger`` are task-agnostic — copy them
verbatim from ``../../classification/src/utils.py``. Only the metric changes:
classification computes accuracy/macro-F1; segmentation computes per-class IoU
(intersection-over-union) + mean IoU, and optionally Dice, from a pixel
confusion matrix. Keep it dependency-light (plain numpy) like classification so
train and eval share identical math.
"""

from __future__ import annotations

import numpy as np


def run_timestamp() -> str:
    """yymmdd-hhmmss run name — copy from classification."""
    raise NotImplementedError("Copy run_timestamp from ../../classification/src/utils.py")


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Seed every RNG — copy from classification."""
    raise NotImplementedError("Copy set_seed from ../../classification/src/utils.py")


def get_logger(name: str, log_file=None):
    """Console + optional file logger — copy from classification."""
    raise NotImplementedError("Copy get_logger from ../../classification/src/utils.py")


def segmentation_metrics(
    targets: np.ndarray, preds: np.ndarray, num_classes: int, ignore_index: int = 255
) -> dict:
    """Per-class IoU + mean IoU (+ Dice) from a pixel confusion matrix.

    Exclude ``ignore_index`` pixels first. For each class c:
        IoU_c  = TP / (TP + FP + FN)
        Dice_c = 2*TP / (2*TP + FP + FN)
    Return {"mean_iou", "dice", "per_class_iou": [...]} — return per-class IoU so
    a mean never hides a collapsing class (same rationale as classification's
    per_class_f1).
    """
    raise NotImplementedError(
        "Accumulate a num_classes x num_classes confusion matrix over valid "
        "pixels and derive IoU/Dice. Keep it numpy-only like classification."
    )
