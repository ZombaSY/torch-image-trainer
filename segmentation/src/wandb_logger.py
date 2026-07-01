"""Thin, optional Weights & Biases logger.

Logs the run **configuration** (once, at init) and **per-epoch training states**
(train/val loss, MAE/MSE, LR). It intentionally does **not** upload model
weights or images — no ``wandb.save`` of checkpoints, no ``wandb.watch``, no
image logging — so W&B storage stays small. A no-op when ``wandb.enabled`` is
false; raises a clear error if enabled but the package is missing.
"""

from __future__ import annotations

from pathlib import Path

from .config import Config


class WandbLogger:
    def __init__(self, cfg: Config, output_dir, logger=None):
        self.run = None
        self._wandb = None
        wcfg = cfg.wandb
        if not wcfg.enabled:
            return

        try:
            import wandb
        except ImportError as exc:
            raise RuntimeError(
                "wandb.enabled=true but wandb is not installed. Run "
                "`pip install wandb` (or set wandb.enabled=false)."
            ) from exc

        name = wcfg.run_name or f"{cfg.model.backbone}_{cfg.head_name}_{Path(output_dir).name}"
        self._wandb = wandb
        # config=... records the full run config; dir keeps W&B's files inside
        # the (gitignored) run directory. No checkpoints/images are ever synced.
        self.run = wandb.init(
            project=wcfg.project,
            entity=wcfg.entity,
            name=name,
            mode=wcfg.mode,
            tags=list(wcfg.tags),
            config=cfg.to_dict(),
            dir=str(output_dir),
        )
        if logger is not None:
            logger.info("W&B logging enabled: project=%s name=%s mode=%s",
                        wcfg.project, name, wcfg.mode)

    def log(self, record: dict) -> None:
        """Log one epoch's metrics (a plain dict of scalars)."""
        if self.run is None:
            return
        self._wandb.log(record, step=record.get("epoch"))

    def finish(self) -> None:
        if self.run is not None:
            self._wandb.finish()
            self.run = None
