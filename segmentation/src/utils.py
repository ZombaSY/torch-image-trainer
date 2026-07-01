"""Small shared helpers: reproducibility, logging, matting metrics.

``run_timestamp`` / ``set_seed`` / ``get_logger`` are identical to the
classification trainer (task-agnostic). Only the metric differs: classification
reports accuracy/F1; this regression task reports pixel error (MAE/MSE).
"""

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


def matting_metrics(targets: np.ndarray, preds: np.ndarray) -> dict:
    """Pixel-error metrics for alpha-matte regression.

    ``targets`` and ``preds`` are alpha values in ``[0, 1]`` (any shape; they
    are flattened). Returns MAE and MSE in ``[0, 1]`` plus ``mae_255`` for a
    human-readable 0-255 error. Kept dependency-light (plain numpy) so train and
    eval share identical math — the same rationale as classification's metrics.
    """
    targets = np.asarray(targets, dtype=np.float64).ravel()
    preds = np.asarray(preds, dtype=np.float64).ravel()
    if targets.size == 0:
        return {"mae": 0.0, "mse": 0.0, "mae_255": 0.0}
    diff = preds - targets
    mae = float(np.abs(diff).mean())
    mse = float((diff ** 2).mean())
    return {"mae": mae, "mse": mse, "mae_255": round(mae * 255.0, 3)}
