"""Typed, YAML-backed configuration for the style-classifier trainer.

A single :class:`Config` object fully describes a run so experiments are
reproducible from the YAML file alone. CLI flags may override individual
leaf fields via dotted keys (see ``train.py``).
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import yaml

# Supported foundation backbones -> their HuggingFace Hub ids (loaded through timm).
SUPPORTED_BACKBONES: dict[str, str] = {
    "wd-swinv2-tagger-v3": "SmilingWolf/wd-swinv2-tagger-v3",
    "wd-vit-large-tagger-v3": "SmilingWolf/wd-vit-large-tagger-v3",
    "wd-vit-tagger-v3": "SmilingWolf/wd-vit-tagger-v3",
}

# The two supported training regimes.
TRAIN_MODES = ("linear_probe", "full_finetune")


@dataclass
class DataConfig:
    """Where the data lives and how images are read."""

    root: str = "/data/seo_sunyong/workspace/data/cocone/UI/P1/style-classifier-dataset"
    train_csv: str = "train.csv"
    val_csv: str = "val.csv"
    image_column: str = "input"
    label_column: str = "class"
    num_classes: int = 4
    class_names: list[str] = field(
        default_factory=lambda: ["chic", "feminine", "lovely", "other"]
    )
    # WD taggers expect images padded to a square on a white background.
    pad_to_square: bool = True
    pad_value: int = 255
    # cv2 reads BGR; convert to RGB to match conventional pretrained pipelines.
    to_rgb: bool = True


@dataclass
class AugConfig:
    """Augmentation switches (train split only).

    Per-sample albumentations ops (horizontal flip, coarse dropout, blur,
    rotate) plus the batch-level mixing augmentations (MixUp, CutMix) applied
    in the training loop.
    """

    horizontal_flip: bool = True
    horizontal_flip_p: float = 0.5

    coarse_dropout: bool = True
    coarse_dropout_p: float = 0.5
    coarse_dropout_max_holes: int = 8
    coarse_dropout_max_height_frac: float = 0.15
    coarse_dropout_max_width_frac: float = 0.15

    blur: bool = True
    blur_p: float = 0.2
    blur_limit: int = 5

    rotate: bool = True
    rotate_p: float = 0.5
    rotate_limit: int = 20

    # Batch-level mixing (applied in the trainer, not albumentations). The
    # ground-truth label is split by the same ratio used to mix the pixels:
    # CutMix uses the true pasted-patch area, MixUp uses its blend ratio.
    mixup: bool = False
    mixup_alpha: float = 0.2          # Beta(alpha, alpha); higher -> stronger

    cutmix: bool = False
    cutmix_alpha: float = 1.0

    # Probability of mixing a given batch, and (when both are on) the chance
    # of picking CutMix over MixUp.
    mix_p: float = 0.5
    mix_switch_prob: float = 0.5
    # Floor on each source's share, so the ratio stays in
    # [mix_min_ratio, 1 - mix_min_ratio] (default [0.3, 0.7]).
    mix_min_ratio: float = 0.3


@dataclass
class ModelConfig:
    """Backbone selection and the classifier head added on top."""

    backbone: str = "wd-vit-tagger-v3"
    pretrained: bool = True
    # Extra dropout applied before the final linear layer (the "simple
    # classifier layer" added on each backbone).
    head_dropout: float = 0.1
    # Override the input resolution; ``None`` keeps the backbone default (448).
    image_size: int | None = None


@dataclass
class OptimConfig:
    """Optimizer, schedule, and regularization."""

    epochs: int = 30
    batch_size: int = 16
    num_workers: int = 8

    optimizer: str = "adamw"
    lr: float = 1e-3            # head LR (linear probe) / base LR (full finetune)
    backbone_lr: float = 1e-5  # backbone LR used only in full_finetune
    weight_decay: float = 1e-4

    scheduler: str = "cosine"   # one of: cosine, none
    warmup_epochs: int = 2
    min_lr: float = 1e-6

    label_smoothing: float = 0.0
    # Counteract class imbalance (class 3 is under-represented).
    class_weighted_loss: bool = True
    weighted_sampler: bool = False

    amp: bool = True
    grad_clip_norm: float = 1.0


@dataclass
class RunConfig:
    """Bookkeeping: seeds, devices, and output locations."""

    mode: str = "linear_probe"
    seed: int = 42
    deterministic: bool = True
    device: str = "cuda"
    output_dir: str = "runs/exp"
    # Metric used to select the best checkpoint: accuracy | macro_f1.
    monitor: str = "macro_f1"
    # Stop after this many consecutive epochs without a new best metric.
    # Set to 0 to disable early stopping (train the full `optim.epochs`).
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

    # ---- validation -------------------------------------------------------
    def validate(self) -> "Config":
        if self.model.backbone not in SUPPORTED_BACKBONES:
            raise ValueError(
                f"Unknown backbone {self.model.backbone!r}; "
                f"choose one of {sorted(SUPPORTED_BACKBONES)}"
            )
        if self.run.mode not in TRAIN_MODES:
            raise ValueError(
                f"Unknown mode {self.run.mode!r}; choose one of {TRAIN_MODES}"
            )
        if len(self.data.class_names) != self.data.num_classes:
            raise ValueError(
                f"class_names ({len(self.data.class_names)}) must match "
                f"num_classes ({self.data.num_classes})"
            )
        if self.run.monitor not in ("accuracy", "macro_f1"):
            raise ValueError("run.monitor must be 'accuracy' or 'macro_f1'")
        return self

    @property
    def hf_hub_id(self) -> str:
        return SUPPORTED_BACKBONES[self.model.backbone]

    # ---- (de)serialization -----------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as fh:
            yaml.safe_dump(self.to_dict(), fh, sort_keys=False)


_SECTION_TYPES = {
    "data": DataConfig,
    "aug": AugConfig,
    "model": ModelConfig,
    "optim": OptimConfig,
    "run": RunConfig,
}


def _build_section(section_cls, values: dict[str, Any] | None):
    """Instantiate one config section, rejecting unknown keys early."""
    values = values or {}
    known = {f.name for f in dataclasses.fields(section_cls)}
    unknown = set(values) - known
    if unknown:
        raise ValueError(
            f"Unknown keys for {section_cls.__name__}: {sorted(unknown)}"
        )
    return section_cls(**values)


def from_dict(raw: dict[str, Any]) -> Config:
    """Build and validate a :class:`Config` from a plain dict.

    Used to reconstruct the exact config stored inside a checkpoint.
    """
    sections = {
        name: _build_section(cls, raw.get(name))
        for name, cls in _SECTION_TYPES.items()
    }
    return Config(**sections).validate()


def load_config(path: str | Path) -> Config:
    """Load and validate a :class:`Config` from a YAML file."""
    with open(path) as fh:
        raw = yaml.safe_load(fh) or {}
    return from_dict(raw)


def apply_overrides(cfg: Config, overrides: dict[str, str]) -> Config:
    """Apply ``section.field=value`` overrides parsed from the CLI.

    Returns a new validated Config (the input is left untouched).
    """
    data = cfg.to_dict()
    for dotted, raw_value in overrides.items():
        section, _, leaf = dotted.partition(".")
        if not leaf or section not in data or leaf not in data[section]:
            raise ValueError(f"Unknown override key: {dotted!r}")
        current = data[section][leaf]
        data[section][leaf] = _coerce(raw_value, current)
    sections = {name: cls(**data[name]) for name, cls in _SECTION_TYPES.items()}
    return Config(**sections).validate()


def _coerce(raw: str, like: Any) -> Any:
    """Coerce a string override to the type of the existing value."""
    if isinstance(like, bool):
        return raw.lower() in ("1", "true", "yes", "y")
    if isinstance(like, int):
        return int(raw)
    if isinstance(like, float):
        return float(raw)
    if like is None:
        # Best effort: int, then float, then leave as string ("none" -> None).
        if raw.lower() in ("none", "null"):
            return None
        for caster in (int, float):
            try:
                return caster(raw)
            except ValueError:
                continue
    return raw
