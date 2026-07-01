"""Optimizer + LR schedule construction faithful to the seg papers' recipes.

Two pieces the papers rely on that a plain 2-group optimizer lacks:

* **Layer-wise LR decay (LLRD).** EVA-02 (0.9) and InternImage (0.94) scale each
  backbone layer's LR by ``layer_decay ** (num_layers - layer_id)`` so shallow
  layers move less than deep ones. Each backbone adapter reports its own
  ``num_layers`` / ``layer_id(name)`` (see ``backbones.py``). ``layer_decay == 1``
  disables it and falls back to the simple head-lr / backbone_lr split (Swin
  uses a single global LR; DINOv2 freezes the backbone entirely).
* **No-decay param groups.** Norms, biases, and backbone-specific tokens
  (position embeddings, relative-position-bias tables, LayerScale gammas) get
  ``weight_decay = 0`` — every recipe does this.

The schedule is stepped **per optimizer iteration** (not per epoch): a linear
warmup over ``warmup_iters`` then poly (or cosine) decay to ``min_lr``, matching
the papers' iteration-based `poly` + 1500-iter warmup.
"""

from __future__ import annotations

import math

import torch
from torch.optim.lr_scheduler import LambdaLR

from .config import Config

_BACKBONE_PREFIX = "backbone."


def _is_no_decay(param, raw_name: str, extra_keys: tuple) -> bool:
    # 1D params are norms/biases; skip weight decay on them plus any adapter keys.
    return param.ndim <= 1 or raw_name.endswith(".bias") or any(k in raw_name for k in extra_keys)


def build_param_groups(model, cfg: Config) -> list[dict]:
    """Group parameters by (lr, weight_decay), applying LLRD + no-decay rules."""
    wd = cfg.optim.weight_decay
    ld = cfg.optim.layer_decay
    base_lr = cfg.optim.lr
    backbone = model.backbone
    no_wd_keys = tuple(getattr(backbone, "no_wd", ()))
    num_layers = getattr(backbone, "num_layers", 1)
    use_llrd = ld < 1.0 and num_layers > 1

    groups: dict[tuple, dict] = {}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        is_backbone = name.startswith(_BACKBONE_PREFIX)
        raw = name[len(_BACKBONE_PREFIX):] if is_backbone else name

        if not is_backbone:                       # decode head + matte conv
            lr = base_lr
            no_decay = p.ndim <= 1 or name.endswith(".bias")
        elif use_llrd:                            # backbone with layer-wise decay
            layer_id = backbone.layer_id(raw)
            lr = base_lr * (ld ** (num_layers - layer_id))
            no_decay = _is_no_decay(p, raw, no_wd_keys)
        else:                                     # backbone, single/other LR
            lr = cfg.optim.backbone_lr if cfg.run.mode == "full_finetune" else base_lr
            no_decay = _is_no_decay(p, raw, no_wd_keys)

        this_wd = 0.0 if no_decay else wd
        key = (round(lr, 12), this_wd)
        groups.setdefault(key, {"params": [], "lr": lr, "weight_decay": this_wd})["params"].append(p)

    return list(groups.values())


def build_optimizer(model, cfg: Config) -> torch.optim.Optimizer:
    groups = build_param_groups(model, cfg)
    name = cfg.optim.optimizer.lower()
    if name == "adamw":
        return torch.optim.AdamW(groups, betas=(0.9, 0.999), weight_decay=cfg.optim.weight_decay)
    if name == "sgd":
        return torch.optim.SGD(groups, momentum=0.9, weight_decay=cfg.optim.weight_decay)
    raise ValueError(f"Unsupported optimizer: {cfg.optim.optimizer!r}")


def build_scheduler(optimizer, cfg: Config, total_iters: int) -> LambdaLR:
    """Per-iteration linear warmup then poly/cosine decay to ``min_lr``.

    Returned as a multiplier on each param group's own base LR, so LLRD's
    per-group LRs all scale together correctly.
    """
    warmup = cfg.optim.warmup_iters
    power = cfg.optim.power
    ratio = cfg.optim.warmup_ratio
    floor = cfg.optim.min_lr / max(cfg.optim.lr, 1e-12)
    sched = cfg.optim.scheduler

    def fn(it: int) -> float:
        if sched == "none":
            return 1.0
        if warmup > 0 and it < warmup:
            return ratio + (1.0 - ratio) * it / warmup
        progress = min(1.0, (it - warmup) / max(1, total_iters - warmup))
        if sched == "poly":
            base = (1.0 - progress) ** power
        else:  # cosine
            base = 0.5 * (1.0 + math.cos(math.pi * progress))
        return floor + (1.0 - floor) * base

    return LambdaLR(optimizer, fn)
