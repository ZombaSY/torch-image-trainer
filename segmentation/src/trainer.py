"""Training loop for segmentation. **PLACEHOLDER.**

Structurally identical to ``../../classification/src/trainer.py`` — reuse that
loop wholesale (AMP, warmup+cosine LambdaLR, per-group optimizer, metric-based
best-checkpoint with the score in the filename, early stopping, history.json,
config snapshot, logger). Only two things change for dense prediction:

  * **Criterion.** Pixel ``CrossEntropyLoss`` over ``(B, C, H, W)`` logits vs
    ``(B, H, W)`` long masks, with ``ignore_index`` set so void/pad pixels are
    skipped. Optionally add a Dice term (``optim.loss == 'ce_dice'``). Class
    weights come from pixel frequencies, same idea as classification.
  * **Validation metric.** ``segmentation_metrics`` (mean IoU / Dice) instead of
    ``classification_metrics``; select the best checkpoint by ``run.monitor``.

There is no MixUp/CutMix here (see README). Everything else — checkpoint format
``{epoch, model_state, config, metrics}`` (contract #2), timestamped output dir
(contract #5) — is unchanged.
"""

from __future__ import annotations

from .config import Config
from .model import build_param_groups, count_trainable_params  # noqa: F401  (used once ported)


class Trainer:
    """Owns the train/val loops and artifact writing. Port from classification.

    Copy the classification Trainer and change: (1) build the criterion from
    ``cfg.optim.loss`` with ``ignore_index=cfg.data.ignore_index``; (2) compute
    ``segmentation_metrics`` in the val epoch; (3) drop the MixAug usage.
    """

    def __init__(self, model, cfg: Config, train_pixel_counts=None):
        raise NotImplementedError(
            "Port Trainer.__init__ from classification: device, output dir, "
            "logger, criterion (pixel CE + ignore_index, optional Dice), "
            "optimizer from build_param_groups, LambdaLR warmup+cosine, "
            "GradScaler, best-metric/early-stop bookkeeping."
        )

    def fit(self, train_loader, val_loader) -> dict:
        raise NotImplementedError(
            "Port fit() from classification: per-epoch train/val, scheduler step, "
            "select best by cfg.run.monitor (mean_iou/dice), save best/last "
            "checkpoints embedding cfg.to_dict(), write history.json, early stop."
        )
