"""Typed, YAML-backed configuration for the segmentation trainer. **PLACEHOLDER.**

Mirrors ``../../classification/src/config.py`` (contract #1 of the blueprint).
The loader/validator/override *machinery* is identical to the classification
trainer and should be copied verbatim — only the section dataclasses below
differ for dense prediction. See ``../README.md`` for the contract summary.

TODO: copy ``_build_section`` / ``from_dict`` / ``load_config`` /
``apply_overrides`` / ``_coerce`` from the classification config module. They are
task-agnostic and must not be re-derived. The section dataclasses are sketched
here so the shape is clear; adjust fields to your dataset before implementing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# Encoder backbones -> their source ids (loaded through timm / smp). Populate
# with the encoders you actually support (e.g. timm-efficientnet, resnet50).
SUPPORTED_ENCODERS: dict[str, str] = {
    # "resnet50": "resnet50",
    # "timm-efficientnet-b3": "timm-efficientnet-b3",
}

# Decoder architectures (segmentation_models_pytorch): Unet | FPN | DeepLabV3Plus
SUPPORTED_DECODERS = ("unet", "fpn", "deeplabv3plus")

# Training regimes, mirroring classification's linear_probe / full_finetune.
TRAIN_MODES = ("decoder_only", "full_finetune")


@dataclass
class DataConfig:
    """Where images and masks live and how they are read.

    Unlike classification (one label per row), segmentation needs a per-image
    *mask* path. Keep the shared image-read knobs (pad/rgb) identical so the
    geometry matches classification.
    """

    root: str = "/path/to/segmentation-dataset"
    train_csv: str = "train.csv"          # columns: image path, mask path
    val_csv: str = "val.csv"
    image_column: str = "image"
    mask_column: str = "mask"             # replaces classification's label_column
    num_classes: int = 2                  # number of segmentation classes
    class_names: list[str] = field(default_factory=lambda: ["background", "foreground"])
    ignore_index: int = 255               # mask value to exclude from loss/metrics (void)
    pad_to_square: bool = True
    pad_value: int = 255                  # image pad; mask is padded with ignore_index
    to_rgb: bool = True


@dataclass
class AugConfig:
    """Geometric + photometric augmentation switches (train split only).

    Geometric ops (flip, rotate, crop) MUST be applied to image AND mask
    together; photometric ops (blur, brightness) apply to the image only.
    albumentations handles this automatically when the mask is passed to the
    same Compose call. Note: MixUp/CutMix are omitted — mixing masks is
    mask-aware and rarely worth it for segmentation.
    """

    horizontal_flip: bool = True
    horizontal_flip_p: float = 0.5
    rotate: bool = True
    rotate_p: float = 0.5
    rotate_limit: int = 20
    blur: bool = False                    # photometric: image only
    blur_p: float = 0.2
    blur_limit: int = 5


@dataclass
class ModelConfig:
    """Encoder + decoder selection."""

    encoder: str = "resnet50"
    decoder: str = "unet"
    pretrained: bool = True
    image_size: int | None = None         # None keeps the encoder default


@dataclass
class OptimConfig:
    """Optimizer, schedule, regularization. Same shape as classification."""

    epochs: int = 100
    batch_size: int = 8
    num_workers: int = 8
    optimizer: str = "adamw"
    lr: float = 1e-3                      # decoder LR
    backbone_lr: float = 1e-5             # encoder LR (full_finetune only)
    weight_decay: float = 1e-4
    scheduler: str = "cosine"
    warmup_epochs: int = 2
    min_lr: float = 1e-6
    loss: str = "ce"                      # ce | dice | ce_dice
    class_weighted_loss: bool = True
    amp: bool = True
    grad_clip_norm: float = 1.0


@dataclass
class RunConfig:
    """Bookkeeping: seeds, devices, output locations."""

    mode: str = "decoder_only"
    seed: int = 42
    deterministic: bool = True
    device: str = "cuda"
    output_dir: str = "runs/exp"
    monitor: str = "mean_iou"             # mean_iou | dice (replaces macro_f1)
    early_stop_patience: int = 50
    save_last: bool = True
    log_interval: int = 10


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    aug: AugConfig = field(default_factory=AugConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    run: RunConfig = field(default_factory=RunConfig)

    def validate(self) -> "Config":
        """Fail fast on impossible combos — mirror classification's validate()."""
        raise NotImplementedError(
            "Port validate() from ../../classification/src/config.py: check "
            "encoder/decoder/mode are supported, len(class_names)==num_classes, "
            "and run.monitor in ('mean_iou', 'dice')."
        )

    def to_dict(self) -> dict[str, Any]:
        raise NotImplementedError("Copy asdict-based to_dict from classification.")


def load_config(path) -> Config:
    """Load + validate a Config from YAML. Copy verbatim from classification."""
    raise NotImplementedError("Port load_config/from_dict/_build_section from classification.")


def apply_overrides(cfg: Config, overrides: dict[str, str]) -> Config:
    """Apply dotted section.field=value CLI overrides. Copy from classification."""
    raise NotImplementedError("Port apply_overrides/_coerce from classification.")
