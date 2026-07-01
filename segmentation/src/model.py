"""Encoder + segmentation decoder. **PLACEHOLDER.**

Classification adds a ``Dropout -> Linear`` head on pooled features; segmentation
instead attaches a decoder (U-Net / FPN / DeepLabV3+) that upsamples encoder
feature maps back to an ``(B, num_classes, H, W)`` logit map.

``segmentation_models_pytorch`` builds exactly this on top of a ``timm`` encoder,
so contract #3 (the encoder dictates preprocessing) still holds: query the
encoder's expected input size / mean / std and match it in the dataloader.

Freeze logic mirrors classification: ``decoder_only`` freezes the encoder (and
keeps its norm layers in eval mode); ``full_finetune`` trains everything with a
smaller encoder LR via per-group params.
"""

from __future__ import annotations

import torch.nn as nn

from .config import Config


def build_model(cfg: Config):
    """Construct encoder+decoder for the configured backbone/mode.

    Returns (model, data_config) where data_config carries the encoder's
    input size / mean / std — same contract as classification's build_model.
    """
    raise NotImplementedError(
        "Build an smp decoder (Unet/FPN/DeepLabV3Plus) on a timm encoder. "
        "Resolve preprocessing from the encoder and return it alongside the model. "
        "Freeze the encoder when cfg.run.mode == 'decoder_only'."
    )


def build_param_groups(model: nn.Module, cfg: Config) -> list[dict]:
    """Per-group LRs: decoder at optim.lr; encoder at backbone_lr in full_finetune.

    Directly analogous to classification's build_param_groups (head vs backbone).
    """
    raise NotImplementedError("Port build_param_groups from classification (decoder vs encoder).")


def count_trainable_params(model: nn.Module) -> tuple[int, int]:
    """(trainable, total) — identical to classification. Copy verbatim."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable, total
