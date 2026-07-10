#!/usr/bin/env python
"""Random hyper-parameter sweep for the segmentation trainer.

Search space (fixed — the only CLI knob is the number of trials):

* **model** — one of the four backbone configs, picked uniformly per trial:
  ``dinov2`` / ``eva`` / ``internimage`` / ``swin``.
* **optim.lr** — log-uniform in ``[0.1x, 10x]`` of the chosen config's own
  ``optim.lr`` (``optim.backbone_lr`` is scaled by the same factor so the
  head/backbone ratio is preserved where layer_decay == 1).
* **aug.cutmix_p** — uniform in ``[0.0, 0.5]``.

Each trial shells out to ``train.py`` (one subprocess per trial, so CUDA
memory is freed between runs) and is scored by the best monitored metric in
that run's ``history.json``. The monitor (``mae``) is *minimized*.

Disk policy (same as the classification sweep): only the run holding the
current best metric is kept; every other trial directory is pruned right
after its result is archived to ``<sweep>/sweep_results.json``, so no
information is lost when the heavy run directory is deleted.

After every completed trial the parameter<->score correlations (each swept
axis vs the best monitored metric and the best validation loss, across all
trials so far) are recomputed and rewritten to
``<sweep>/sweep_correlation.json``. Both files live on disk and are updated
per trial, so a sweep killed mid-way still leaves its analysis behind.

Example
-------
Run a 100-trial sweep::

    python sweeper.py --trials 100
"""

from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import subprocess
import sys
from pathlib import Path

from src.config import MINIMIZE_METRICS, load_config
from src.utils import get_logger, run_timestamp, sweep_correlations

# Swept models: name -> base YAML config (each carries its own tuned lr).
MODEL_CONFIGS: dict[str, str] = {
    "dinov2": "configs/dinov2_dpt.yaml",
    "eva": "configs/eva02_dpt.yaml",
    "internimage": "configs/internimage_uper.yaml",
    "swin": "configs/swin_uper.yaml",
}

# LR is sampled log-uniformly in [LR_FACTOR_RANGE[0] * base_lr,
# LR_FACTOR_RANGE[1] * base_lr] of the chosen config's value.
LR_FACTOR_RANGE = (0.1, 10.0)
# aug.cutmix_p is sampled uniformly from this range.
CUTMIX_P_RANGE = (0.0, 0.5)

SWEEP_ROOT_PARENT = Path("runs")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Random sweep over model / lr / cutmix_p "
                    "(fixed search space; only the trial count is exposed).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--trials", type=int, default=100,
        help="Number of random trials to run.",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Sampler seed for the trial plan (default: fresh entropy each "
             "run; the seed is logged so a sweep can be replayed).",
    )
    args = parser.parse_args()
    if args.trials < 1:
        parser.error("--trials must be >= 1.")
    return args


def _round_sig(x: float, sig: int = 3) -> float:
    """Round to ``sig`` significant figures for tidy logs and filenames."""
    if x == 0:
        return 0.0
    return round(x, -int(math.floor(math.log10(abs(x)))) + (sig - 1))


def build_trials(trials: int, rng: random.Random) -> list[dict]:
    """Draw ``trials`` samples: model uniform, lr factor log-uniform,
    cutmix_p uniform."""
    lo, hi = LR_FACTOR_RANGE
    plan = []
    for _ in range(trials):
        model = rng.choice(sorted(MODEL_CONFIGS))
        base = load_config(MODEL_CONFIGS[model])
        factor = math.exp(rng.uniform(math.log(lo), math.log(hi)))
        plan.append(
            {
                "model": model,
                "optim.lr": _round_sig(base.optim.lr * factor),
                "optim.backbone_lr": _round_sig(base.optim.backbone_lr * factor),
                "aug.cutmix_p": round(rng.uniform(*CUTMIX_P_RANGE), 3),
            }
        )
    return plan


def find_run_dir(trial_base: Path) -> Path | None:
    """train.py writes into <trial_base>/<timestamp>; return that child."""
    candidates = [p.parent for p in trial_base.glob("*/history.json")]
    if not candidates:
        return None
    # Newest, in case a crashed retry left more than one.
    return max(candidates, key=lambda p: p.stat().st_mtime)


def best_metric_from_history(
    run_dir: Path, monitor: str, minimize: bool
) -> tuple[float, int]:
    """Return (best score, best epoch) for ``monitor`` from history.json."""
    with open(run_dir / "history.json") as fh:
        history = json.load(fh)
    best_score, best_epoch = math.inf if minimize else -math.inf, -1
    for record in history:
        score = record.get(monitor)
        if score is None:
            continue
        if (score < best_score) if minimize else (score > best_score):
            best_score, best_epoch = score, record["epoch"]
    return best_score, best_epoch


def best_loss_from_history(history: list[dict]) -> float | None:
    """Lowest validation loss in a run, for the correlation report."""
    losses = [rec["loss"] for rec in history
              if isinstance(rec.get("loss"), (int, float))]
    return round(min(losses), 4) if losses else None


def update_correlations(results: list[dict], monitor: str, path: Path, logger) -> None:
    """Recompute parameter<->score correlations from every archived trial and
    rewrite them to ``path``. Called after each trial so a sweep killed by the
    user still leaves the analysis on disk."""
    rows = [
        {
            "params": r["params"],
            "metrics": {
                f"best_{monitor}": r.get(f"best_{monitor}"),
                "best_loss": r.get("best_loss"),
            },
        }
        for r in results if r.get("status") == "ok"
    ]
    report = sweep_correlations(rows)
    with open(path, "w") as fh:
        json.dump(report, fh, indent=2)

    stats = report["metrics"].get(f"best_{monitor}", {})
    parts = [
        f"{param} rho={s['spearman']:+.2f}"
        for param, s in stats.items()
        if s["kind"] == "numeric" and s["spearman"] is not None
    ]
    if parts:
        logger.info(
            "correlation vs best_%s (n=%d): %s",
            monitor, report["n_trials"], " | ".join(parts),
        )


def fmt_params(params: dict) -> str:
    return (
        f"model={params['model']} "
        f"lr={params['optim.lr']:g} "
        f"backbone_lr={params['optim.backbone_lr']:g} "
        f"cutmix_p={params['aug.cutmix_p']:g}"
    )


def main() -> None:
    args = parse_args()
    seed = args.seed if args.seed is not None else random.SystemRandom().randrange(2**32)
    rng = random.Random(seed)

    # All base configs monitor the same (minimized) metric; verify and reuse.
    monitors = {load_config(path).run.monitor for path in MODEL_CONFIGS.values()}
    if len(monitors) != 1:
        raise SystemExit(f"Base configs disagree on run.monitor: {sorted(monitors)}")
    monitor = monitors.pop()
    minimize = monitor in MINIMIZE_METRICS

    sweep_root = SWEEP_ROOT_PARENT / f"sweep_{run_timestamp()}"
    plan = build_trials(args.trials, rng)

    sweep_root.mkdir(parents=True, exist_ok=True)
    logger = get_logger("sweep", sweep_root / "sweep.log")
    logger.info(
        "Sweep %s | %d trials | monitor=%s (%s) | seed=%d",
        sweep_root, len(plan), monitor, "min" if minimize else "max", seed,
    )
    for i, params in enumerate(plan):
        logger.info("  trial %03d: %s", i, fmt_params(params))

    results: list[dict] = []
    global_best = math.inf if minimize else -math.inf
    best_run_dir, best_trial = None, None
    results_path = sweep_root / "sweep_results.json"

    for i, params in enumerate(plan):
        trial_base = sweep_root / f"trial_{i:03d}"
        config = MODEL_CONFIGS[params["model"]]
        cmd = [sys.executable, "train.py", "--config", config]
        cmd += [f"{k}={v}" for k, v in params.items() if k != "model"]
        cmd += [f"run.output_dir={trial_base}"]

        logger.info("[%d/%d] running: %s", i + 1, len(plan), fmt_params(params))
        proc = subprocess.run(cmd, cwd=Path(__file__).resolve().parent)

        run_dir = find_run_dir(trial_base)
        if proc.returncode != 0 or run_dir is None:
            logger.warning(
                "[%d/%d] FAILED (rc=%s); pruning %s",
                i + 1, len(plan), proc.returncode, trial_base,
            )
            shutil.rmtree(trial_base, ignore_errors=True)
            results.append({"trial": i, "params": params, "status": "failed"})
            _dump(results_path, results)
            continue

        score, epoch = best_metric_from_history(run_dir, monitor, minimize)
        with open(run_dir / "history.json") as fh:
            history = json.load(fh)
        breaks_best = (score < global_best) if minimize else (score > global_best)
        results.append(
            {
                "trial": i,
                "params": params,
                "status": "ok",
                f"best_{monitor}": round(score, 4),
                "best_loss": best_loss_from_history(history),
                "best_epoch": epoch,
                "breaks_best": breaks_best,
                "history": history,
            }
        )
        _dump(results_path, results)  # archive before any deletion
        update_correlations(
            results, monitor, sweep_root / "sweep_correlation.json", logger
        )

        if breaks_best:
            logger.info(
                "[%d/%d] NEW BEST %s=%.4f (epoch %d) | %s",
                i + 1, len(plan), monitor, score, epoch, fmt_params(params),
            )
            if best_run_dir is not None:
                logger.info("  pruning previous best %s", best_run_dir.parent)
                shutil.rmtree(best_run_dir.parent, ignore_errors=True)
            global_best, best_run_dir, best_trial = score, run_dir, i
        else:
            logger.info(
                "[%d/%d] %s=%.4f did not beat best %.4f",
                i + 1, len(plan), monitor, score, global_best,
            )
            shutil.rmtree(trial_base, ignore_errors=True)

    if best_run_dir is not None:
        winner = next(r for r in results if r["trial"] == best_trial)
        logger.info(
            "Sweep complete | best %s=%.4f | %s | %s",
            monitor, global_best, fmt_params(winner["params"]), best_run_dir,
        )
    else:
        logger.warning("Sweep complete | no successful trial produced a metric.")


def _dump(path: Path, results: list[dict]) -> None:
    with open(path, "w") as fh:
        json.dump(results, fh, indent=2)


if __name__ == "__main__":
    main()
