"""Small shared helpers: reproducibility, logging, metrics."""

from __future__ import annotations

import logging
import math
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


def _corr(x: np.ndarray, y: np.ndarray) -> float | None:
    """Pearson r, or None when undefined (< 3 points or a constant axis)."""
    if x.size < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return None
    return round(float(np.corrcoef(x, y)[0, 1]), 4)


def _ranks(a: np.ndarray) -> np.ndarray:
    """1-based ranks with ties averaged (turns Pearson into Spearman)."""
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(a.size, dtype=np.float64)
    ranks[order] = np.arange(1, a.size + 1, dtype=np.float64)
    for v in np.unique(a):
        tied = a == v
        if tied.sum() > 1:
            ranks[tied] = ranks[tied].mean()
    return ranks


def sweep_correlations(rows: list[dict]) -> dict:
    """Correlate swept parameters with per-trial scores across sweep runs.

    ``rows`` holds one entry per completed trial:
    ``{"params": {axis: value}, "metrics": {name: score}}``.

    Numeric axes get Pearson + Spearman coefficients (Spearman is rank-based,
    so it is the one to read for log-sampled axes like learning rates).
    Categorical axes (e.g. the backbone) get per-value group stats instead,
    since linear correlation is undefined for unordered categories. Plain
    numpy, for the same auditability reason as the other metrics here.
    """
    out: dict = {"n_trials": len(rows), "metrics": {}}
    param_names = list(dict.fromkeys(k for r in rows for k in r["params"]))
    metric_names = list(dict.fromkeys(k for r in rows for k in r["metrics"]))
    for metric in metric_names:
        per_param: dict = {}
        for param in param_names:
            pairs = [
                (r["params"][param], r["metrics"][metric])
                for r in rows
                if param in r["params"]
                and isinstance(r["metrics"].get(metric), (int, float))
                and math.isfinite(r["metrics"][metric])
            ]
            if not pairs:
                continue
            values = [v for v, _ in pairs]
            scores = np.asarray([s for _, s in pairs], dtype=np.float64)
            if all(isinstance(v, (int, float)) and not isinstance(v, bool)
                   for v in values):
                xs = np.asarray(values, dtype=np.float64)
                per_param[param] = {
                    "kind": "numeric",
                    "n": len(pairs),
                    "pearson": _corr(xs, scores),
                    "spearman": _corr(_ranks(xs), _ranks(scores)),
                }
            else:
                groups = {}
                for v in sorted({str(v) for v in values}):
                    g = scores[np.asarray([str(x) == v for x in values])]
                    groups[v] = {
                        "n": int(g.size),
                        "mean": round(float(g.mean()), 4),
                        "min": round(float(g.min()), 4),
                        "max": round(float(g.max()), 4),
                    }
                per_param[param] = {
                    "kind": "categorical", "n": len(pairs), "groups": groups,
                }
        out["metrics"][metric] = per_param
    return out
