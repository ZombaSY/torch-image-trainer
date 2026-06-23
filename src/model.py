"""WD tagger backbone + a simple classifier head.

The backbone is a WD v3 tagger loaded through timm's HuggingFace Hub
integration and turned into a pooled feature extractor. A lightweight
``Dropout -> Linear`` head is added on top to produce the 4 style logits.
"""

from __future__ import annotations

import timm
import torch
import torch.nn as nn

from .config import Config


class StyleClassifier(nn.Module):
    """Pretrained WD tagger backbone with a small linear classifier head."""

    def __init__(self, backbone: nn.Module, num_features: int, num_classes: int, dropout: float):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(num_features, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)            # pooled features (B, num_features)
        return self.head(feats)

    # ---- parameter groups -------------------------------------------------
    def set_backbone_trainable(self, trainable: bool) -> None:
        for p in self.backbone.parameters():
            p.requires_grad = trainable
        # BatchNorm/other norm layers must also stop updating running stats
        # when the backbone is frozen.
        self.backbone.train(trainable)

    def backbone_parameters(self):
        return self.backbone.parameters()

    def head_parameters(self):
        return self.head.parameters()


def build_model(cfg: Config) -> tuple[StyleClassifier, dict]:
    """Construct the model for the configured backbone and training mode.

    Returns the model and the resolved timm data config (input size, mean,
    std) so the dataloader can match the backbone's expected preprocessing.
    """
    backbone = timm.create_model(
        f"hf-hub:{cfg.hf_hub_id}",
        pretrained=cfg.model.pretrained,
    )
    data_config = timm.data.resolve_model_data_config(backbone)

    num_features = backbone.num_features
    backbone.reset_classifier(0)  # strip native tag head -> pooled features

    model = StyleClassifier(
        backbone=backbone,
        num_features=num_features,
        num_classes=cfg.data.num_classes,
        dropout=cfg.model.head_dropout,
    )

    # Linear probing freezes the backbone; full fine-tuning trains everything.
    model.set_backbone_trainable(cfg.run.mode == "full_finetune")
    return model, data_config


def build_param_groups(model: StyleClassifier, cfg: Config) -> list[dict]:
    """Per-group LRs: head always trains; backbone only in full_finetune."""
    groups = [{"params": list(model.head_parameters()), "lr": cfg.optim.lr}]
    if cfg.run.mode == "full_finetune":
        groups.append(
            {"params": list(model.backbone_parameters()), "lr": cfg.optim.backbone_lr}
        )
    return groups


def count_trainable_params(model: nn.Module) -> tuple[int, int]:
    """Return (trainable, total) parameter counts."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable, total
