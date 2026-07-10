#!/usr/bin/env python
"""Hyper-parameter sweep over ``optim.lr``, ``optim.backbone_lr`` and
``model.backbone``.

Two search strategies:

* ``--search grid`` (default): Cartesian product of discrete value lists.
* ``--search random``: ``--trials N`` samples drawn from ranges -- the LRs
  log-uniformly from a ``[low high]`` bound, the backbone uniformly from a
  given set.

Each trial shells out to ``train.py`` (one subprocess per trial, so CUDA
memory is freed between runs) and is scored by the best monitored metric
recorded in that run's ``history.json``.

Disk policy
-----------
Training checkpoints are ~1.2 GB each, so the sweep keeps only the single
run that currently holds the best metric.  After every trial:

* if the trial **breaks** the running best, the *previous* best run is
  removed and the new one is kept;
* if the trial does **not** break the running best, it is removed straight
  away.

Either way the trial's result (params, best score, best epoch, full
per-epoch history) is archived to ``<sweep>/sweep_results.json`` first, so
no information is lost when the heavy run directory is deleted.

After every completed trial the parameter<->score correlations (each swept
axis vs the best monitored metric and the best validation loss, across all
trials so far) are recomputed and rewritten to
``<sweep>/sweep_correlation.json``. Both files live on disk and are updated
per trial, so a sweep killed mid-way still leaves its analysis behind.

Examples
--------
Random search: 20 trials, lr log-uniform in [1e-5, 1e-3], backbone_lr
log-uniform in [1e-6, 1e-4], backbone picked at random each trial::

    python sweep.py --config configs/full_finetune.yaml --search random \
        --trials 20 \
        --lr 1e-5 1e-3 \
        --backbone-lr 1e-6 1e-4 \
        --backbone wd-vit-tagger-v3 wd-swinv2-tagger-v3 \
        --target 0.85

Grid sweep across two LRs, two backbone LRs and two backbones::

    python sweep.py --config configs/full_finetune.yaml \
        --lr 5e-4 1e-3 \
        --backbone-lr 1e-5 5e-5 \
        --backbone wd-vit-tagger-v3 wd-swinv2-tagger-v3

List the trial plan without training anything::

    python sweep.py --config configs/full_finetune.yaml --search random \
        --trials 8 --lr 1e-5 1e-3 --dry-run
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import random
import shutil
import subprocess
import sys
from pathlib import Path

from src.config import SUPPORTED_BACKBONES, load_config
from src.utils import get_logger, run_timestamp, sweep_correlations

# Default search space. The LR axes are log-uniform [low, high] ranges; the
# backbone axis is the full supported set, sampled uniformly (random) or
# expanded in turn (grid).
DEFAULT_LR_RANGE = [1e-5, 1e-3]
DEFAULT_BACKBONE_LR_RANGE = [1e-6, 1e-4]
DEFAULT_BACKBONES = sorted(SUPPORTED_BACKBONES)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Grid sweep over lr / backbone_lr / backbone.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", required=True, help="Base YAML config.")
    parser.add_argument(
        "--search", choices=("grid", "random"), default="random",
        help="grid = product of value lists; random = sampled trials.",
    )
    parser.add_argument(
        "--trials", type=int, default=200,
        help="Number of random trials (required for --search random).",
    )
    parser.add_argument(
        "--lr", nargs="+", type=float, default=DEFAULT_LR_RANGE,
        help="grid: optim.lr values. random: 'LOW HIGH' log-uniform range "
             "(or a single fixed value).",
    )
    parser.add_argument(
        "--backbone-lr", nargs="+", type=float, default=DEFAULT_BACKBONE_LR_RANGE,
        help="grid: optim.backbone_lr values. random: 'LOW HIGH' log-uniform "
             "range (or single value).",
    )
    parser.add_argument(
        "--backbone", nargs="+", default=DEFAULT_BACKBONES,
        choices=sorted(SUPPORTED_BACKBONES),
        help="grid: every backbone in turn. random: the set sampled from.",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Sampler seed for random search (default: fresh entropy each "
             "run; the seed is logged so a sweep can be replayed).",
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Sweep root (default: <base output_dir>/sweep_<timestamp>).",
    )
    parser.add_argument(
        "--target", type=float, default=None,
        help="Stop early once a trial's best metric reaches this value.",
    )
    parser.add_argument(
        "--keep-losers", action="store_true",
        help="Keep every run directory instead of pruning the heavy ones.",
    )
    parser.add_argument(
        "--python", default=sys.executable,
        help="Python interpreter used to launch each train.py subprocess.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print the trial plan and exit without training.",
    )
    return parser.parse_args()


def _round_sig(x: float, sig: int = 3) -> float:
    """Round to ``sig`` significant figures for tidy logs and filenames."""
    if x == 0:
        return 0.0
    return round(x, -int(math.floor(math.log10(abs(x)))) + (sig - 1))


def _sample_axis(rng: random.Random, spec: list[float]) -> float:
    """Sample a continuous axis: one value = fixed, two = log-uniform range."""
    if len(spec) == 1:
        return spec[0]
    lo, hi = sorted(spec)
    return _round_sig(math.exp(rng.uniform(math.log(lo), math.log(hi))))


def build_grid(args: argparse.Namespace, base) -> list[dict]:
    """Cartesian product of the three swept axes."""
    lrs = args.lr
    backbone_lrs = args.backbone_lr
    backbones = args.backbone

    # backbone_lr does nothing when the backbone is frozen; collapse it so we
    # don't train identical linear-probe runs over and over.
    if base.run.mode == "linear_probe" and len(backbone_lrs) > 1:
        backbone_lrs = backbone_lrs[:1]

    grid = []
    for backbone, lr, backbone_lr in itertools.product(backbones, lrs, backbone_lrs):
        grid.append(
            {
                "model.backbone": backbone,
                "optim.lr": lr,
                "optim.backbone_lr": backbone_lr,
            }
        )
    return grid


def build_random(args: argparse.Namespace, base, rng: random.Random) -> list[dict]:
    """Draw ``args.trials`` samples: LRs log-uniform, backbone uniform."""
    lr_spec = args.lr
    blr_spec = args.backbone_lr
    backbones = args.backbone
    # backbone_lr is unused when the backbone is frozen -> keep it fixed.
    frozen = base.run.mode == "linear_probe"

    trials = []
    for _ in range(args.trials):
        trials.append(
            {
                "model.backbone": rng.choice(backbones),
                "optim.lr": _sample_axis(rng, lr_spec),
                "optim.backbone_lr": (
                    base.optim.backbone_lr if frozen
                    else _sample_axis(rng, blr_spec)
                ),
            }
        )
    return trials


def build_trials(args: argparse.Namespace, base, rng: random.Random) -> list[dict]:
    """Dispatch to the grid or random trial generator."""
    if args.search == "random":
        return build_random(args, base, rng)
    return build_grid(args, base)


def find_run_dir(trial_base: Path) -> Path | None:
    """train.py writes into <trial_base>/<timestamp>; return that child."""
    candidates = [p.parent for p in trial_base.glob("*/history.json")]
    if not candidates:
        return None
    # Newest, in case a crashed retry left more than one.
    return max(candidates, key=lambda p: p.stat().st_mtime)


def best_metric_from_history(run_dir: Path, monitor: str) -> tuple[float, int]:
    """Return (best score, best epoch) for ``monitor`` from history.json."""
    with open(run_dir / "history.json") as fh:
        history = json.load(fh)
    best_score, best_epoch = -math.inf, -1
    for record in history:
        score = record.get(monitor)
        if score is not None and score > best_score:
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
        f"backbone={params['model.backbone']} "
        f"lr={params['optim.lr']:g} "
        f"backbone_lr={params['optim.backbone_lr']:g}"
    )


def _validate(args: argparse.Namespace) -> None:
    """Fail fast on argument combinations that can't be satisfied."""
    if args.search == "random":
        if not args.trials or args.trials < 1:
            raise SystemExit("--search random requires --trials N (N >= 1).")
        for name, spec in (("--lr", args.lr), ("--backbone-lr", args.backbone_lr)):
            if spec is not None and len(spec) > 2:
                raise SystemExit(
                    f"{name} takes one fixed value or two range bounds in "
                    f"random search, got {len(spec)}."
                )
    elif args.trials is not None:
        raise SystemExit("--trials only applies to --search random.")


def main() -> None:
    args = parse_args()
    _validate(args)
    base = load_config(args.config)
    monitor = base.run.monitor
    seed = args.seed if args.seed is not None else random.SystemRandom().randrange(2**32)
    rng = random.Random(seed)

    sweep_root = Path(
        args.output_dir
        or Path(base.run.output_dir).parent / f"sweep_{run_timestamp()}"
    )
    grid = build_trials(args, base, rng)
    plan = f"search={args.search}"
    if args.search == "random":
        plan += f" seed={seed}"

    if args.dry_run:
        print(f"Dry run | sweep root {sweep_root} | {len(grid)} trials | "
              f"monitor={monitor} | mode={base.run.mode} | {plan}")
        for i, params in enumerate(grid):
            print(f"  trial {i:03d}: {fmt_params(params)}")
        return

    sweep_root.mkdir(parents=True, exist_ok=True)
    logger = get_logger("sweep", sweep_root / "sweep.log")
    logger.info(
        "Sweep %s | %d trials | monitor=%s | mode=%s | %s",
        sweep_root, len(grid), monitor, base.run.mode, plan,
    )
    for i, params in enumerate(grid):
        logger.info("  trial %03d: %s", i, fmt_params(params))

    results: list[dict] = []
    global_best, best_run_dir, best_trial = -math.inf, None, None
    results_path = sweep_root / "sweep_results.json"

    for i, params in enumerate(grid):
        trial_base = sweep_root / f"trial_{i:03d}"
        cmd = [args.python, "train.py", "--config", args.config]
        cmd += [f"{k}={v}" for k, v in params.items()]
        cmd += [f"run.output_dir={trial_base}"]

        logger.info("[%d/%d] running: %s", i + 1, len(grid), fmt_params(params))
        proc = subprocess.run(cmd, cwd=Path(__file__).resolve().parent)

        run_dir = find_run_dir(trial_base)
        if proc.returncode != 0 or run_dir is None:
            logger.warning(
                "[%d/%d] FAILED (rc=%s); pruning %s",
                i + 1, len(grid), proc.returncode, trial_base,
            )
            shutil.rmtree(trial_base, ignore_errors=True)
            results.append({"trial": i, "params": params, "status": "failed"})
            _dump(results_path, results)
            continue

        score, epoch = best_metric_from_history(run_dir, monitor)
        with open(run_dir / "history.json") as fh:
            history = json.load(fh)
        breaks_best = score > global_best
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
                i + 1, len(grid), monitor, score, epoch, fmt_params(params),
            )
            if not args.keep_losers and best_run_dir is not None:
                logger.info("  pruning previous best %s", best_run_dir.parent)
                shutil.rmtree(best_run_dir.parent, ignore_errors=True)
            global_best, best_run_dir, best_trial = score, run_dir, i
        else:
            logger.info(
                "[%d/%d] %s=%.4f did not beat best %.4f",
                i + 1, len(grid), monitor, score, global_best,
            )
            if not args.keep_losers:
                shutil.rmtree(trial_base, ignore_errors=True)

        if args.target is not None and global_best >= args.target:
            logger.info(
                "Target %s>=%.4f reached at trial %d; stopping sweep.",
                monitor, args.target, best_trial,
            )
            break

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
