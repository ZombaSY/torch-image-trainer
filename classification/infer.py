#!/usr/bin/env python
"""Predict class-wise probabilities for a single image.

Loads a ``.pt`` checkpoint (which embeds its own config) and prints the
per-class probabilities for one ``.png`` (or any readable) image.

Example
-------
    python infer.py --checkpoint runs/linear_probe_vit/best.pt --image foo.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from src.dataset import build_transforms, load_image
from test import load_checkpoint  # reuse the checkpoint loader from test.py


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Single-image inference.")
    parser.add_argument("--checkpoint", required=True, help="Path to a .pt checkpoint.")
    parser.add_argument("--image", required=True, help="Path to one image (.png).")
    parser.add_argument("--device", default="cuda", help="cuda or cpu.")
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    device = torch.device(
        args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu"
    )

    model, cfg, data_config, _ = load_checkpoint(args.checkpoint, device)

    image_size = cfg.model.image_size or data_config["input_size"][1]
    transform = build_transforms(
        cfg.aug, image_size, tuple(data_config["mean"]), tuple(data_config["std"]),
        train=False,
    )

    image = load_image(
        Path(args.image), cfg.data.to_rgb, cfg.data.pad_to_square, cfg.data.pad_value
    )
    x = transform(image=image)["image"].unsqueeze(0).to(device)
    probs = torch.softmax(model(x), dim=1)[0].cpu().tolist()

    for name, p in zip(cfg.data.class_names, probs):
        print(f"{name}: {p:.4f}")


if __name__ == "__main__":
    main()
