"""Typed, YAML-backed configuration for the alpha-matte segmentation trainer.

A single :class:`Config` fully describes a run so experiments are reproducible
from the YAML alone (blueprint contract #1). CLI flags override individual leaf
fields via dotted keys (see ``train.py``). The loader/override/validate
machinery is intentionally identical to the classification trainer — only the
sections and the backbone registry differ.

The task is alpha-matte *regression*: the model takes an RGB composite and
predicts a single-channel alpha map in ``[0, 1]``. There is no class axis, so
the loss is a pixel distance (see ``losses.py``) and the monitored metric is an
error that is *minimized* (MAE).
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import yaml

# Backbone registry. Pure metadata (no heavy imports) so config stays light;
# ``backbones.py`` reads this to actually build each model.
#   family:  'vit' (non-hierarchical, single-stride tokens) -> DPT head
#            'hierarchical' (multi-stage feature pyramid)    -> UPerHead
#   source:  who loads it -> 'timm' | 'dinov2' (transformers) | 'internimage'
#   image_size: backbone-native default input (overridable via model.image_size)
BACKBONES: dict[str, dict[str, Any]] = {
    "eva02-l": {
        "family": "vit", "head": "dpt", "source": "timm",
        "model_id": "eva02_large_patch14_448", "image_size": 448,
    },
    # DINOv3 stands in as DINOv2-large (no DINOv3 access); ViT-L, patch 14.
    "dinov2-l": {
        "family": "vit", "head": "dpt", "source": "dinov2",
        "model_id": "facebook/dinov2-large", "image_size": 448,
    },
    "swinv2-l": {
        "family": "hierarchical", "head": "uper", "source": "timm",
        "model_id": "swinv2_large_window12to24_192to384", "image_size": 384,
    },
    "internimage-l": {
        "family": "hierarchical", "head": "uper", "source": "internimage",
        "model_id": "OpenGVLab/internimage_l_22k_384", "image_size": 384,
    },
}

# Decoder heads and the backbone family each one accepts. The plain heads
# ('dpt'/'uper') predict the matte from backbone-stride features (final logit
# bilinearly upsampled -> blurry alpha boundaries); the '*matte' detail-capture
# heads (ViTMatte-style ConvStream + DeepLab ASPP context) fuse raw-image
# detail up to stride 1 for sharp boundaries. See heads.py.
HEAD_FAMILIES: dict[str, str] = {
    "dpt": "vit",
    "vitmatte": "vit",
    "uper": "hierarchical",
    "upermatte": "hierarchical",
}
HEADS = tuple(HEAD_FAMILIES)
# Heads whose fusion widths are decoder_channels / {2,4,8}.
DETAIL_HEADS = ("vitmatte", "upermatte")
TRAIN_MODES = ("decoder_only", "full_finetune")
LOSSES = ("l1", "mse", "l1_mse")
# Metrics where a *lower* value is better (drives best-checkpoint selection).
MINIMIZE_METRICS = ("mae", "mse", "loss")


@dataclass
class DataConfig:
    """Where the data lives and how images/masks are read.

    ``input`` is an RGBA logo; the shared reader composites it onto a solid
    ``pad_value`` background and returns RGB (the model never sees the alpha it
    must predict). ``label`` is that alpha channel saved as a grayscale matte —
    the regression target, normalized to ``[0, 1]``.
    """

    root: str = "/data/seo_sunyong/workspace/data/cocone/background-segmentation/P1-logo"
    train_csv: str = "train-seg.csv"
    val_csv: str = "val-seg.csv"
    image_column: str = "input"
    mask_column: str = "label"
    # WD-style square padding for the RGB input. The composite/pad background
    # is white (``dataset.WHITE_BG``) unless the random-background aug fires;
    # ``pad_value`` no longer drives it and is kept only so configs embedded in
    # existing checkpoints/run snapshots still load.
    pad_to_square: bool = True
    pad_value: int = 255
    # Padded regions are background, so the alpha there is fully transparent.
    mask_pad_value: int = 0
    to_rgb: bool = True
    # Recover straight color (rgb / alpha) before compositing, for premultiplied
    # source assets. These logos look mostly straight-alpha, so set false if
    # matte edges look over-bright. See dataset.composite_rgba.
    unpremultiply_alpha: bool = True
    # Preload every raw decoded image+matte into RAM at dataset construction,
    # removing the per-item disk read + PNG decode from the training loop (the
    # full dataset decodes to ~3 GB). Disable if the dataset outgrows memory.
    cache_in_memory: bool = True


@dataclass
class AugConfig:
    """Train-split augmentation (albumentations). Strong on purpose — the
    dataset is tiny (~90 images), so aggressive aug fights overfitting.

    Geometric ops (flips, rotate) are applied to the image AND the mask
    together; photometric ops (color jitter, blur, coarse dropout) touch only
    the image. CutMix (below) runs first, before any of these. MixUp is absent:
    blending two mattes is ill-defined for this regression target, whereas
    CutMix cut-pastes a patch on both image and mask, keeping the matte exact.
    """

    horizontal_flip: bool = True
    horizontal_flip_p: float = 0.5

    # Vertical flip — added for this task; logos are not strongly up/down biased.
    vertical_flip: bool = True
    vertical_flip_p: float = 0.5

    rotate: bool = True
    rotate_p: float = 0.5
    rotate_limit: int = 20

    # Random crop at the model input size (train only, image AND mask). When it
    # fires, the padded square is cropped at native scale instead of resized
    # down — translation/scale augmentation that also presents matte edges at
    # full detail. Samples smaller than the input size are padded (white image
    # bg, transparent mask) at a random position instead. Defaults off so
    # config snapshots from older runs reproduce unchanged.
    random_crop: bool = False
    random_crop_p: float = 0.5

    blur: bool = True
    blur_p: float = 0.2
    blur_limit: int = 5

    coarse_dropout: bool = True
    coarse_dropout_p: float = 0.5
    coarse_dropout_max_holes: int = 8
    coarse_dropout_max_height_frac: float = 0.15
    coarse_dropout_max_width_frac: float = 0.15

    # Color jitter — added for this task (photometric, image only).
    color_jitter: bool = True
    color_jitter_p: float = 0.5
    color_jitter_brightness: float = 0.2
    color_jitter_contrast: float = 0.2
    color_jitter_saturation: float = 0.2
    color_jitter_hue: float = 0.1

    # Random solid-background compositing — the logo (RGBA) is composited onto a
    # random solid color instead of white, so the model learns alpha regardless
    # of background. Applied in the dataset reader (needs the raw RGBA), not
    # albumentations. The target matte is unchanged.
    random_background: bool = True
    random_background_p: float = 0.5

    # CutMix — paste a rectangular patch from another sample onto this one, on
    # both the image AND its matte (dense target -> no label mixing needed).
    # Applied FIRST, before every other augmentation, so the mixed image+mask is
    # then flipped/rotated/jittered as one coherent sample. Box size ~ (1 - lam),
    # lam ~ Beta(alpha, alpha).
    cutmix: bool = True
    cutmix_alpha: float = 1.0
    cutmix_p: float = 0.5


@dataclass
class ModelConfig:
    """Backbone selection; the head is derived from the backbone family."""

    backbone: str = "eva02-l"
    # None -> use the family default from BACKBONES (dpt for vit, uper for
    # hierarchical). Set explicitly to force a specific head — e.g. the
    # detail-capture heads ('vitmatte' / 'upermatte') for sharp matte edges.
    head: str | None = None
    pretrained: bool = True
    # Channel width of the decoder's fused features.
    decoder_channels: int = 256
    # Dropout before the final 1-channel matte conv.
    head_dropout: float = 0.1
    # Stochastic depth on the backbone (passed to timm; seg recipes use ~0.2).
    drop_path_rate: float = 0.0
    # Override the input resolution; None keeps the backbone-native size.
    image_size: int | None = None


@dataclass
class OptimConfig:
    """Optimizer, schedule, regularization."""

    epochs: int = 300
    batch_size: int = 8
    num_workers: int = 8

    optimizer: str = "adamw"
    lr: float = 1e-4            # base/peak LR (decode head; backbone under LLRD)
    backbone_lr: float = 1e-5  # backbone LR when layer_decay == 1 and full_finetune
    weight_decay: float = 1e-4

    # Layer-wise LR decay on the backbone (BEiT/EVA-style). <1 enables it: layer
    # i gets lr * layer_decay**(num_layers - layer_id). 1.0 disables it (then the
    # 2-group head-lr / backbone_lr scheme is used).
    layer_decay: float = 1.0

    scheduler: str = "poly"     # poly | cosine | none
    power: float = 1.0          # poly power (1.0 = linear decay to min_lr)
    # Warmup + schedule run per optimizer step (matching the seg papers), so
    # warmup is measured in iterations, not epochs.
    warmup_iters: int = 1500
    warmup_ratio: float = 1e-6
    min_lr: float = 1e-6

    loss: str = "l1"            # l1 | mse | l1_mse
    l1_mse_weight: float = 0.5  # weight on the L1 term when loss == l1_mse

    amp: bool = True
    grad_clip_norm: float = 1.0


@dataclass
class RunConfig:
    """Bookkeeping: seeds, devices, output locations."""

    mode: str = "decoder_only"  # decoder_only (backbone frozen) | full_finetune
    seed: int = 42
    deterministic: bool = True
    device: str = "cuda"
    output_dir: str = "runs/exp"
    # Metric that selects the best checkpoint. mae/mse are minimized.
    monitor: str = "mae"
    early_stop_patience: int = 50
    save_last: bool = True
    log_interval: int = 10


@dataclass
class WandbConfig:
    """Weights & Biases logging.

    Logs the run **configuration** and per-epoch **training states** (losses,
    metrics, LR) only. Model weights and images are deliberately excluded — the
    trainer never uploads checkpoints or image artifacts, keeping W&B storage
    light. Disabled by default so runs don't require a W&B account.
    """

    enabled: bool = False
    project: str = "torch-image-trainer-seg"
    entity: str | None = None
    run_name: str | None = None      # None -> "<backbone>_<head>_<run timestamp>"
    mode: str = "online"             # online | offline | disabled
    tags: list[str] = field(default_factory=list)


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    aug: AugConfig = field(default_factory=AugConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    run: RunConfig = field(default_factory=RunConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)

    # ---- validation -------------------------------------------------------
    def validate(self) -> "Config":
        if self.model.backbone not in BACKBONES:
            raise ValueError(
                f"Unknown backbone {self.model.backbone!r}; "
                f"choose one of {sorted(BACKBONES)}"
            )
        head = self.model.head
        if head is not None and head not in HEADS:
            raise ValueError(f"Unknown head {head!r}; choose one of {HEADS}")
        if head is not None and HEAD_FAMILIES[head] != self.backbone_meta["family"]:
            raise ValueError(
                f"Head {head!r} expects a {HEAD_FAMILIES[head]!r} backbone, but "
                f"{self.model.backbone!r} is {self.backbone_meta['family']!r}"
            )
        if self.head_name in DETAIL_HEADS and self.model.decoder_channels % 8:
            raise ValueError(
                f"{self.head_name!r} needs model.decoder_channels divisible by 8 "
                f"(got {self.model.decoder_channels})"
            )
        if self.run.mode not in TRAIN_MODES:
            raise ValueError(f"Unknown mode {self.run.mode!r}; choose one of {TRAIN_MODES}")
        if self.optim.loss not in LOSSES:
            raise ValueError(f"Unknown loss {self.optim.loss!r}; choose one of {LOSSES}")
        if self.optim.scheduler not in ("poly", "cosine", "none"):
            raise ValueError("optim.scheduler must be 'poly', 'cosine', or 'none'")
        if self.run.monitor not in ("mae", "mse"):
            raise ValueError("run.monitor must be 'mae' or 'mse'")
        if self.wandb.mode not in ("online", "offline", "disabled"):
            raise ValueError("wandb.mode must be 'online', 'offline', or 'disabled'")
        return self

    # ---- derived ----------------------------------------------------------
    @property
    def backbone_meta(self) -> dict[str, Any]:
        return BACKBONES[self.model.backbone]

    @property
    def head_name(self) -> str:
        """The head to use: explicit override, else the backbone-family default."""
        return self.model.head or self.backbone_meta["head"]

    @property
    def image_size(self) -> int:
        return self.model.image_size or self.backbone_meta["image_size"]

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
    "wandb": WandbConfig,
}


def _build_section(section_cls, values: dict[str, Any] | None):
    """Instantiate one config section, rejecting unknown keys early."""
    values = values or {}
    known = {f.name for f in dataclasses.fields(section_cls)}
    unknown = set(values) - known
    if unknown:
        raise ValueError(f"Unknown keys for {section_cls.__name__}: {sorted(unknown)}")
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
        if raw.lower() in ("none", "null"):
            return None
        for caster in (int, float):
            try:
                return caster(raw)
            except ValueError:
                continue
    return raw
