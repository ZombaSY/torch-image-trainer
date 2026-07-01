"""Training loop for alpha-matte regression: AMP, cosine warmup, MAE-based
checkpointing.

Structurally identical to the classification trainer; the differences are the
regression criterion (pixel distance, see ``losses.py``), the metric
(MAE/MSE via ``matting_metrics``), and that the monitored metric is *minimized*
(lower error is better) rather than maximized. Artifacts (config snapshot,
best/last checkpoints, per-epoch history) are written under the run's output dir
so any run can be reproduced and inspected afterwards (blueprint contract #5).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .config import Config, MINIMIZE_METRICS
from .losses import build_criterion
from .model import SegModel, count_trainable_params
from .optim import build_optimizer, build_scheduler
from .utils import get_logger, matting_metrics
from .wandb_logger import WandbLogger


class Trainer:
    def __init__(self, model: SegModel, cfg: Config):
        self.cfg = cfg
        self.device = torch.device(
            cfg.run.device if torch.cuda.is_available() else "cpu"
        )
        self.model = model.to(self.device)

        self.output_dir = Path(cfg.run.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = get_logger("trainer", self.output_dir / "train.log")

        self.criterion = build_criterion(cfg)

        self.optimizer = build_optimizer(model, cfg)
        # Scheduler is per-iteration; built in fit() once steps/epoch is known.
        self.scheduler = None
        self.use_amp = cfg.optim.amp and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

        # Monitored metric is minimized here (mae/mse), so "best" starts high.
        self.minimize = cfg.run.monitor in MINIMIZE_METRICS
        self.best_metric = math.inf if self.minimize else -math.inf
        self.best_epoch = -1
        self.best_ckpt_path: Path | None = None
        self.epochs_since_best = 0
        self.history: list[dict] = []

    def _is_better(self, score: float) -> bool:
        return score < self.best_metric if self.minimize else score > self.best_metric

    # ---- epochs -----------------------------------------------------------
    def _run_train_epoch(self, loader: DataLoader, epoch: int) -> float:
        self.model.train()
        # A frozen backbone (decoder_only) stays in eval mode for stable stats.
        if self.cfg.run.mode == "decoder_only":
            self.model.backbone.eval()

        running, seen = 0.0, 0
        for step, (images, targets) in enumerate(loader):
            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=self.use_amp):
                pred = self.model(images)
                loss = self.criterion(pred, targets)

            self.scaler.scale(loss).backward()
            if self.cfg.optim.grad_clip_norm > 0:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.optim.grad_clip_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()  # per-iteration poly/cosine + warmup

            running += loss.item() * images.size(0)
            seen += images.size(0)
            if step % self.cfg.run.log_interval == 0:
                self.logger.info(
                    "epoch %d step %d/%d loss %.4f", epoch, step, len(loader), loss.item(),
                )
        return running / max(1, seen)

    @torch.no_grad()
    def _run_val_epoch(self, loader: DataLoader) -> dict:
        self.model.eval()
        running, seen = 0.0, 0
        preds, targs = [], []
        for images, targets in loader:
            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=self.use_amp):
                pred = self.model(images)
                loss = self.criterion(pred, targets)
            running += loss.item() * images.size(0)
            seen += images.size(0)
            preds.append(pred.float().cpu().numpy())
            targs.append(targets.float().cpu().numpy())

        metrics = matting_metrics(np.concatenate(targs), np.concatenate(preds))
        metrics["loss"] = running / max(1, seen)
        return metrics

    # ---- checkpoints ------------------------------------------------------
    def _ckpt_name(self, tag: str, epoch: int, metrics: dict) -> str:
        monitor = self.cfg.run.monitor
        return f"{tag}_e{epoch}_{monitor}{metrics[monitor]:.4f}.pt"

    def _save_checkpoint(self, name: str, epoch: int, metrics: dict) -> Path:
        path = self.output_dir / name
        torch.save(
            {
                "epoch": epoch,
                "model_state": self.model.state_dict(),
                "config": self.cfg.to_dict(),
                "metrics": metrics,
            },
            path,
        )
        return path

    # ---- public API -------------------------------------------------------
    def fit(self, train_loader: DataLoader, val_loader: DataLoader) -> dict:
        trainable, total = count_trainable_params(self.model)
        self.logger.info(
            "Mode=%s backbone=%s head=%s | trainable %s / %s params (%.1f%%)",
            self.cfg.run.mode, self.cfg.model.backbone, self.cfg.head_name,
            f"{trainable:,}", f"{total:,}", 100 * trainable / total,
        )
        self.cfg.save(self.output_dir / "config.yaml")

        # Poly/cosine schedule spans the full run; warmup is in iterations.
        total_iters = self.cfg.optim.epochs * max(1, len(train_loader))
        self.scheduler = build_scheduler(self.optimizer, self.cfg, total_iters)

        # Logs config + per-epoch states only (no weights/images). No-op if off.
        self.wandb = WandbLogger(self.cfg, self.output_dir, self.logger)

        epoch = 0
        for epoch in range(self.cfg.optim.epochs):
            train_loss = self._run_train_epoch(train_loader, epoch)
            val_metrics = self._run_val_epoch(val_loader)

            monitored = val_metrics[self.cfg.run.monitor]
            record = {
                "epoch": epoch,
                "train_loss": round(train_loss, 4),
                "lr": self.optimizer.param_groups[0]["lr"],
                **{k: (round(v, 4) if isinstance(v, float) else v)
                   for k, v in val_metrics.items()},
            }
            self.history.append(record)
            self.wandb.log(record)
            self.logger.info(
                "epoch %d done | train_loss %.4f | val_loss %.4f | mae %.4f (%.2f/255) | mse %.4f",
                epoch, train_loss, val_metrics["loss"],
                val_metrics["mae"], val_metrics["mae_255"], val_metrics["mse"],
            )

            if self._is_better(monitored):
                self.best_metric = monitored
                self.best_epoch = epoch
                self.epochs_since_best = 0
                if self.best_ckpt_path is not None and self.best_ckpt_path.exists():
                    self.best_ckpt_path.unlink()
                self.best_ckpt_path = self._save_checkpoint(
                    self._ckpt_name("best", epoch, val_metrics), epoch, val_metrics
                )
                self.logger.info(
                    "new best %s=%.4f at epoch %d -> %s", self.cfg.run.monitor,
                    monitored, epoch, self.best_ckpt_path.name,
                )
            else:
                self.epochs_since_best += 1

            patience = self.cfg.run.early_stop_patience
            if patience > 0 and self.epochs_since_best >= patience:
                self.logger.info(
                    "Early stop: no %s improvement for %d epochs (best %.4f at epoch %d).",
                    self.cfg.run.monitor, patience, self.best_metric, self.best_epoch,
                )
                break

        if self.cfg.run.save_last:
            self._save_checkpoint(self._ckpt_name("last", epoch, val_metrics), epoch, val_metrics)

        with open(self.output_dir / "history.json", "w") as fh:
            json.dump(self.history, fh, indent=2)

        summary = {
            "best_epoch": self.best_epoch,
            f"best_{self.cfg.run.monitor}": round(self.best_metric, 4),
            "best_checkpoint": self.best_ckpt_path.name if self.best_ckpt_path else None,
        }
        self.logger.info("Training complete | %s", summary)
        self.wandb.finish()
        return summary
