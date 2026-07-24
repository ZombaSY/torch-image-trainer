#!/usr/bin/env python
"""Dump augmented + preprocessed training samples for visual inspection.

For each sample it runs the *exact* train pipeline (RGBA read -> random-background
composite / unpremultiply -> geometric + photometric aug -> resize -> normalize),
then de-normalizes the tensor back to a viewable RGB and saves it side-by-side
with its aligned alpha target. Use this to confirm the augmentation and
preprocessing are what you expect before a long training run.

    python preview_aug.py --config configs/eva02_dpt.yaml --n 30
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from src.config import load_config
from src.dataset import MatteDataset, build_transforms
from src.utils import set_seed


def _resolve_mean_std(cfg) -> tuple[tuple, tuple]:
    """Match backbones.py: timm nets use their pretrained cfg, others ImageNet."""
    meta = cfg.backbone_meta
    if meta["source"] == "timm":
        import timm

        pc = timm.get_pretrained_cfg(meta["model_id"])
        return tuple(pc.mean), tuple(pc.std)
    return (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def _denormalize(img: torch.Tensor, mean, std) -> np.ndarray:
    """(3,H,W) normalized tensor -> HxWx3 BGR uint8 for cv2."""
    mean_t = torch.tensor(mean).view(3, 1, 1)
    std_t = torch.tensor(std).view(3, 1, 1)
    rgb = (img * std_t + mean_t).clamp(0, 1).mul(255).round().byte()
    rgb = rgb.permute(1, 2, 0).numpy()  # HWC, RGB
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preview train augmentation/preprocessing.")
    parser.add_argument("--config", default="configs/eva02_dpt.yaml", help="Config to mirror.")
    parser.add_argument("--n", type=int, default=30, help="Number of samples to dump.")
    parser.add_argument("--output-dir", default="outputs/aug_preview", help="Where to write panels.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(cfg.run.seed, cfg.run.deterministic)  # reproducible aug sampling

    mean, std = _resolve_mean_std(cfg)
    size = cfg.image_size
    tf = build_transforms(cfg.aug, size, mean, std, train=True)
    ds = MatteDataset(f"{cfg.data.root}/{cfg.data.train_csv}", cfg.data, cfg.aug, tf, train=True, image_size=size)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    n = min(args.n, len(ds))

    for i in range(n):
        image, alpha = ds[i]
        rgb = _denormalize(image, mean, std)                          # HxWx3 BGR
        matte = (alpha[0] * 255).round().byte().numpy()              # HxW
        matte3 = cv2.cvtColor(matte, cv2.COLOR_GRAY2BGR)
        sep = np.full((size, 4, 3), 255, np.uint8)                   # white divider
        panel = cv2.hconcat([rgb, sep, matte3])                      # [input | alpha]
        stem = Path(ds.images[i]).stem
        cv2.imwrite(str(out_dir / f"{i:02d}_{stem}.png"), panel)

    print(f"Wrote {n} preview panels (input | alpha) to {out_dir}")
    print(f"Config: {args.config} | input {size}x{size} | mean {mean} | std {std}")
    print(f"random_background={cfg.aug.random_background} (p={cfg.aug.random_background_p}) | "
          f"unpremultiply_alpha={cfg.data.unpremultiply_alpha}")


if __name__ == "__main__":
    main()
