"""Batch-level MixUp and CutMix augmentation.

MixUp blends two images by a Beta-sampled ratio ``lam``; CutMix pastes a
rectangular patch from one image onto another and derives ``lam`` from the
*actual* pasted-patch area (after integer rounding/clipping), not the sampled
value. In both cases the same ``lam`` splits the ground-truth label: the
training loss is ``lam * loss(logits, y) + (1 - lam) * loss(logits, y_perm)``,
so the label weight tracks exactly how much of each source image survives.

These operate on already-batched, on-device tensors, so they live next to the
training loop rather than in the per-sample dataset pipeline. Sampling uses
``numpy``/``torch`` RNGs, both seeded by ``utils.set_seed``, so runs stay
reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .config import AugConfig


@dataclass
class MixResult:
    """A (possibly) mixed batch plus what's needed for its blended loss."""

    images: torch.Tensor
    target_a: torch.Tensor
    target_b: torch.Tensor
    lam: float  # weight of target_a; 1.0 when no mixing was applied


def _sample_lam(alpha: float, min_ratio: float) -> float:
    """Mixing ratio from ``Beta(alpha, alpha)``, clamped to keep each source's
    share at or above ``min_ratio`` (i.e. ``lam in [min_ratio, 1 - min_ratio]``).

    ``alpha<=0`` disables mixing for that op (returns 1.0, the identity ratio).
    """
    if alpha <= 0.0:
        return 1.0
    lam = float(np.random.beta(alpha, alpha))
    return float(np.clip(lam, min_ratio, 1.0 - min_ratio))


def _rand_bbox(height: int, width: int, lam: float) -> tuple[int, int, int, int]:
    """Random box covering ``(1 - lam)`` of the area, placed fully inside the
    frame.

    Keeping the box in-bounds (rather than centering then clipping) means the
    pasted area equals the target, so the floor on the mix ratio survives near
    image edges instead of shrinking the patch toward zero.
    """
    cut_ratio = float(np.sqrt(1.0 - lam))
    cut_h = min(int(round(height * cut_ratio)), height)
    cut_w = min(int(round(width * cut_ratio)), width)
    y1 = int(np.random.randint(0, height - cut_h + 1))
    x1 = int(np.random.randint(0, width - cut_w + 1))
    return y1, y1 + cut_h, x1, x1 + cut_w


class MixAug:
    """Apply MixUp and/or CutMix to a batch per the augmentation config.

    When both are enabled, each fired batch picks CutMix with probability
    ``mix_switch_prob`` and MixUp otherwise. A batch is left untouched with
    probability ``1 - mix_p`` (or always, when neither is enabled).
    """

    def __init__(self, aug: AugConfig):
        self.mixup = aug.mixup
        self.mixup_alpha = aug.mixup_alpha
        self.cutmix = aug.cutmix
        self.cutmix_alpha = aug.cutmix_alpha
        self.p = aug.mix_p
        self.switch_prob = aug.mix_switch_prob
        self.min_ratio = aug.mix_min_ratio

    @property
    def enabled(self) -> bool:
        return self.mixup or self.cutmix

    def __call__(self, images: torch.Tensor, labels: torch.Tensor) -> MixResult:
        if not self.enabled or float(np.random.rand()) >= self.p:
            return MixResult(images, labels, labels, 1.0)

        use_cutmix = self.cutmix and (
            not self.mixup or float(np.random.rand()) < self.switch_prob
        )
        perm = torch.randperm(images.size(0), device=images.device)
        if use_cutmix:
            return self._cutmix(images, labels, perm)
        return self._mixup(images, labels, perm)

    def _mixup(self, images, labels, perm) -> MixResult:
        lam = _sample_lam(self.mixup_alpha, self.min_ratio)
        mixed = lam * images + (1.0 - lam) * images[perm]
        return MixResult(mixed, labels, labels[perm], lam)

    def _cutmix(self, images, labels, perm) -> MixResult:
        lam = _sample_lam(self.cutmix_alpha, self.min_ratio)
        height, width = images.shape[-2:]
        y1, y2, x1, x2 = _rand_bbox(height, width, lam)
        mixed = images.clone()
        mixed[:, :, y1:y2, x1:x2] = images[perm, :, y1:y2, x1:x2]
        # Recompute lam from the patch's true area so the label split matches
        # the pixels actually replaced; re-clamp to absorb integer rounding at
        # the [min_ratio, 1 - min_ratio] boundary.
        area = (y2 - y1) * (x2 - x1)
        lam = 1.0 - area / float(height * width)
        lam = float(np.clip(lam, self.min_ratio, 1.0 - self.min_ratio))
        return MixResult(mixed, labels, labels[perm], lam)


def mix_loss(criterion, logits: torch.Tensor, result: MixResult) -> torch.Tensor:
    """Blend the loss against both mixed targets by the mix ratio.

    With ``lam == 1`` (no mix) this reduces exactly to ``criterion(logits, y)``.
    Splitting at the loss level keeps the class weights and label smoothing
    already configured on ``criterion`` intact.
    """
    if result.lam >= 1.0:
        return criterion(logits, result.target_a)
    return (
        result.lam * criterion(logits, result.target_a)
        + (1.0 - result.lam) * criterion(logits, result.target_b)
    )
