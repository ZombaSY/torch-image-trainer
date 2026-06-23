"""Training loop: AMP, cosine warmup schedule, metric-based checkpointing.

The :class:`Trainer` owns the train/val loops and persists artifacts
(config snapshot, best/last checkpoints, per-epoch metrics) under the run's
output directory so any run can be reproduced and inspected after the fact.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .config import Config
from .dataset import class_weights
from .model import StyleClassifier, build_param_groups, count_trainable_params
from .utils import classification_metrics, get_logger


def _build_optimizer(model: StyleClassifier, cfg: Config) -> torch.optim.Optimizer:
    groups = build_param_groups(model, cfg)
    name = cfg.optim.optimizer.lower()
    if name == "adamw":
        return torch.optim.AdamW(groups, weight_decay=cfg.optim.weight_decay)
    if name == "sgd":
        return torch.optim.SGD(groups, momentum=0.9, weight_decay=cfg.optim.weight_decay)
    raise ValueError(f"Unsupported optimizer: {cfg.optim.optimizer!r}")


def _lr_lambda(cfg: Config):
    """Linear warmup then cosine decay to ``min_lr`` (as a fraction of base)."""
    total = cfg.optim.epochs
    warmup = cfg.optim.warmup_epochs
    floor = cfg.optim.min_lr / max(cfg.optim.lr, 1e-12)

    def fn(epoch: int) -> float:
        if cfg.optim.scheduler == "none":
            return 1.0
        if warmup > 0 and epoch < warmup:
            return (epoch + 1) / warmup
        progress = (epoch - warmup) / max(1, total - warmup)
        cosine = 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))
        return floor + (1 - floor) * cosine

    return fn


class Trainer:
    def __init__(self, model: StyleClassifier, cfg: Config, train_labels: list[int]):
        self.cfg = cfg
        self.device = torch.device(
            cfg.run.device if torch.cuda.is_available() else "cpu"
        )
        self.model = model.to(self.device)

        self.output_dir = Path(cfg.run.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = get_logger("trainer", self.output_dir / "train.log")

        weight = None
        if cfg.optim.class_weighted_loss:
            weight = class_weights(train_labels, cfg.data.num_classes).to(self.device)
            self.logger.info("Class weights: %s", weight.tolist())
        self.criterion = nn.CrossEntropyLoss(
            weight=weight, label_smoothing=cfg.optim.label_smoothing
        )

        self.optimizer = _build_optimizer(model, cfg)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, _lr_lambda(cfg)
        )
        self.use_amp = cfg.optim.amp and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

        self.history: list[dict] = []
        self.best_metric = -math.inf
        self.best_epoch = -1
        self.best_ckpt_path: Path | None = None
        # Epochs elapsed since the last improvement (drives early stopping).
        self.epochs_since_best = 0

    # ---- epochs -----------------------------------------------------------
    def _run_train_epoch(self, loader: DataLoader, epoch: int) -> float:
        self.model.train()
        # A frozen backbone (linear probe) stays in eval mode for stable stats.
        if self.cfg.run.mode == "linear_probe":
            self.model.backbone.eval()

        running, seen = 0.0, 0
        for step, (images, labels) in enumerate(loader):
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=self.use_amp):
                logits = self.model(images)
                loss = self.criterion(logits, labels)

            self.scaler.scale(loss).backward()
            if self.cfg.optim.grad_clip_norm > 0:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.cfg.optim.grad_clip_norm
                )
            self.scaler.step(self.optimizer)
            self.scaler.update()

            running += loss.item() * images.size(0)
            seen += images.size(0)
            if step % self.cfg.run.log_interval == 0:
                self.logger.info(
                    "epoch %d step %d/%d loss %.4f",
                    epoch, step, len(loader), loss.item(),
                )
        return running / max(1, seen)

    @torch.no_grad()
    def _run_val_epoch(self, loader: DataLoader) -> dict:
        self.model.eval()
        running, seen = 0.0, 0
        all_preds, all_targets = [], []
        for images, labels in loader:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=self.use_amp):
                logits = self.model(images)
                loss = self.criterion(logits, labels)
            running += loss.item() * images.size(0)
            seen += images.size(0)
            all_preds.append(logits.argmax(1).cpu().numpy())
            all_targets.append(labels.cpu().numpy())

        metrics = classification_metrics(
            np.concatenate(all_targets), np.concatenate(all_preds),
            self.cfg.data.num_classes,
        )
        metrics["loss"] = running / max(1, seen)
        return metrics

    # ---- checkpoints ------------------------------------------------------
    def _ckpt_name(self, tag: str, epoch: int, metrics: dict) -> str:
        """Build a filename that embeds the epoch and monitored metric score."""
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
            "Mode=%s backbone=%s | trainable %s / %s params (%.1f%%)",
            self.cfg.run.mode, self.cfg.model.backbone,
            f"{trainable:,}", f"{total:,}", 100 * trainable / total,
        )
        self.cfg.save(self.output_dir / "config.yaml")

        for epoch in range(self.cfg.optim.epochs):
            train_loss = self._run_train_epoch(train_loader, epoch)
            val_metrics = self._run_val_epoch(val_loader)
            self.scheduler.step()

            monitored = val_metrics[self.cfg.run.monitor]
            record = {
                "epoch": epoch,
                "train_loss": round(train_loss, 4),
                "lr": self.optimizer.param_groups[0]["lr"],
                **{k: (round(v, 4) if isinstance(v, float) else v)
                   for k, v in val_metrics.items()},
            }
            self.history.append(record)
            self.logger.info(
                "epoch %d done | train_loss %.4f | val_loss %.4f | "
                "acc %.4f | macro_f1 %.4f",
                epoch, train_loss, val_metrics["loss"],
                val_metrics["accuracy"], val_metrics["macro_f1"],
            )

            if monitored > self.best_metric:
                self.best_metric = monitored
                self.best_epoch = epoch
                self.epochs_since_best = 0
                # Drop the previous best (~hundreds of MB) before saving anew.
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
                    "Early stop: no %s improvement for %d epochs "
                    "(best %.4f at epoch %d).",
                    self.cfg.run.monitor, patience, self.best_metric,
                    self.best_epoch,
                )
                break

        if self.cfg.run.save_last:
            self._save_checkpoint(
                self._ckpt_name("last", epoch, val_metrics), epoch, val_metrics
            )

        with open(self.output_dir / "history.json", "w") as fh:
            json.dump(self.history, fh, indent=2)

        summary = {
            "best_epoch": self.best_epoch,
            f"best_{self.cfg.run.monitor}": round(self.best_metric, 4),
            "best_checkpoint": self.best_ckpt_path.name if self.best_ckpt_path else None,
        }
        self.logger.info("Training complete | %s", summary)
        return summary
