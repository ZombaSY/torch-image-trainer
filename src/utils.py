"""Small shared helpers: reproducibility, logging, metrics."""

from __future__ import annotations

import logging
import os
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch


def run_timestamp() -> str:
    """Return the current local time as a ``yymmdd-hhmmss`` run name."""
    return datetime.now().strftime("%y%m%d-%H%M%S")


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Seed every RNG that affects training for reproducible runs."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True


def get_logger(name: str, log_file: str | Path | None = None) -> logging.Logger:
    """Console logger, optionally tee'd to a file. Idempotent per name."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%H:%M:%S")

    stream = logging.StreamHandler()
    stream.setFormatter(fmt)
    logger.addHandler(stream)

    if log_file is not None:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)
    return logger


def classification_metrics(
    targets: np.ndarray, preds: np.ndarray, num_classes: int
) -> dict:
    """Accuracy plus macro precision/recall/F1, computed without sklearn deps.

    (sklearn is available, but keeping this dependency-light makes the metric
    path trivial to audit and reproduce.)
    """
    targets = np.asarray(targets)
    preds = np.asarray(preds)
    accuracy = float((targets == preds).mean()) if len(targets) else 0.0

    per_class_f1, per_class_p, per_class_r = [], [], []
    for c in range(num_classes):
        tp = int(((preds == c) & (targets == c)).sum())
        fp = int(((preds == c) & (targets != c)).sum())
        fn = int(((preds != c) & (targets == c)).sum())
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        per_class_p.append(precision)
        per_class_r.append(recall)
        per_class_f1.append(f1)

    return {
        "accuracy": accuracy,
        "macro_f1": float(np.mean(per_class_f1)),
        "macro_precision": float(np.mean(per_class_p)),
        "macro_recall": float(np.mean(per_class_r)),
        "per_class_f1": [round(x, 4) for x in per_class_f1],
    }
