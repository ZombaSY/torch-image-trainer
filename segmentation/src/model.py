"""Backbone + decoder head + shared matte conv.

The backbone (see ``backbones.py``) produces features; the head (see
``heads.py``) fuses them into a dense ``decoder_channels`` map; a shared
``Dropout -> 1x1 Conv`` projects that to a single alpha channel, which is
upsampled to the input resolution and squashed to ``[0, 1]`` with a sigmoid.

Freeze logic mirrors the classification trainer: ``decoder_only`` freezes the
backbone (and keeps its norm layers in eval mode); ``full_finetune`` trains
everything with a smaller backbone LR via per-group params.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbones import build_backbone
from .config import Config
from .heads import DPTHead, UPerHead


class SegModel(nn.Module):
    def __init__(self, backbone: nn.Module, head: nn.Module, decoder_channels: int, dropout: float):
        super().__init__()
        self.backbone = backbone
        self.decoder = head
        self.matte = nn.Sequential(
            nn.Dropout2d(p=dropout),
            nn.Conv2d(decoder_channels, 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)                 # list of feature maps
        dec = self.decoder(feats)                # (B, decoder_channels, H', W')
        logit = self.matte(dec)                  # (B, 1, H', W')
        logit = F.interpolate(logit, size=x.shape[2:], mode="bilinear", align_corners=False)
        return torch.sigmoid(logit)              # alpha in [0, 1]

    # ---- parameter groups -------------------------------------------------
    def set_backbone_trainable(self, trainable: bool) -> None:
        for p in self.backbone.parameters():
            p.requires_grad = trainable
        # Frozen backbone: keep norm layers from updating running stats.
        self.backbone.train(trainable)

    def backbone_parameters(self):
        return self.backbone.parameters()

    def head_parameters(self):
        yield from self.decoder.parameters()
        yield from self.matte.parameters()


def build_model(cfg: Config) -> tuple[SegModel, dict]:
    """Build backbone + head for the configured model and mode.

    Returns the model and the resolved data config (input size, mean, std) so
    the dataloader can match the backbone's expected preprocessing.
    """
    backbone, data_config = build_backbone(cfg)

    head_name = cfg.head_name
    if head_name == "uper":
        head: nn.Module = UPerHead(backbone.feature_channels, cfg.model.decoder_channels)
    elif head_name == "dpt":
        head = DPTHead(backbone.embed_dim, cfg.model.decoder_channels)
    else:
        raise ValueError(f"Unknown head {head_name!r}")

    model = SegModel(
        backbone=backbone,
        head=head,
        decoder_channels=cfg.model.decoder_channels,
        dropout=cfg.model.head_dropout,
    )
    model.set_backbone_trainable(cfg.run.mode == "full_finetune")
    return model, data_config


def count_trainable_params(model: nn.Module) -> tuple[int, int]:
    """Return (trainable, total) parameter counts."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable, total
