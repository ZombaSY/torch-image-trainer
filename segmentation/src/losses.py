"""Pixel-distance losses for alpha-matte regression.

The target is a continuous alpha map in ``[0, 1]``, so the model is trained like
an image-to-image (generation) task rather than a per-class segmentation: the
loss is a pixel distance between the predicted alpha and the ground-truth alpha,
not a cross-entropy over classes.

L1 (mean absolute error) is the default — it preserves sharper matte edges than
L2, which is why pix2pix-style image translation favors it. MSE and a weighted
L1+MSE combination are available via config.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import Config


class PixelLoss(nn.Module):
    """Pixel distance between predicted and target alpha maps.

    Both inputs are ``(B, 1, H, W)`` in ``[0, 1]``. ``kind`` is one of
    ``l1`` | ``mse`` | ``l1_mse``; for ``l1_mse`` the L1 term is weighted by
    ``l1_weight`` and MSE by ``1 - l1_weight``.
    """

    def __init__(self, kind: str = "l1", l1_weight: float = 0.5):
        super().__init__()
        self.kind = kind
        self.l1_weight = l1_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.kind == "l1":
            return F.l1_loss(pred, target)
        if self.kind == "mse":
            return F.mse_loss(pred, target)
        if self.kind == "l1_mse":
            w = self.l1_weight
            return w * F.l1_loss(pred, target) + (1.0 - w) * F.mse_loss(pred, target)
        raise ValueError(f"Unsupported loss kind: {self.kind!r}")


def build_criterion(cfg: Config) -> PixelLoss:
    return PixelLoss(kind=cfg.optim.loss, l1_weight=cfg.optim.l1_mse_weight)
