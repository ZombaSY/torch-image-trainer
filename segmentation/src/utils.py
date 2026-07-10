"""Small shared helpers: reproducibility, logging, matting metrics.

``run_timestamp`` / ``set_seed`` / ``get_logger`` / ``sweep_correlations`` are
identical to the classification trainer (task-agnostic). Only the metric
differs: classification reports accuracy/F1; this regression task reports
pixel error (MAE/MSE).
"""

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
