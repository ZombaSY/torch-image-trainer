#!/usr/bin/env python
"""Entrypoint for training the alpha-matte segmentation model.

Examples
--------
Decoder-only (backbone frozen) with the EVA-02 + DPT config::

    python train.py --config configs/eva02_dpt.yaml

Full fine-tuning of Swin-L + UPerHead, overriding a couple of fields::

    python train.py --config configs/swin_uper.yaml \
        run.mode=full_finetune optim.epochs=200
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.config import load_config, apply_overrides
from src.dataset import build_dataloaders
from src.model import build_model
from src.trainer import Trainer
from src.utils import run_timestamp, set_seed


def parse_args() -> tuple[argparse.Namespace, dict]:
    parser = argparse.ArgumentParser(description="Train the alpha-matte segmentation model.")
    parser.add_argument("--config", required=True, help="Path to a YAML config file.")
    parser.add_argument(
        "overrides", nargs="*",
        help="Optional section.field=value overrides, e.g. optim.lr=5e-4",
    )
    args = parser.parse_args()
    overrides: dict[str, str] = {}
    for item in args.overrides:
        if "=" not in item:
            parser.error(f"Override must be key=value, got {item!r}")
        key, value = item.split("=", 1)
        overrides[key] = value
    return args, overrides


def main() -> None:
    args, overrides = parse_args()
    cfg = load_config(args.config)
    if overrides:
        cfg = apply_overrides(cfg, overrides)

    # Save each run into a timestamped subdir: <output_dir>/<yymmdd-hhmmss>.
    run_dir = str(Path(cfg.run.output_dir) / run_timestamp())
    cfg = apply_overrides(cfg, {"run.output_dir": run_dir})

    set_seed(cfg.run.seed, cfg.run.deterministic)

    # Backbone first: it dictates the input size and normalization the
    # dataloader must match.
    model, data_config = build_model(cfg)
    image_size = data_config["input_size"]
    mean = tuple(data_config["mean"])
    std = tuple(data_config["std"])

    train_loader, val_loader, train_ds = build_dataloaders(cfg, mean, std, image_size)

    trainer = Trainer(model, cfg)
    trainer.logger.info(
        "Input %dx%d | mean %s | std %s | train %d / val %d images",
        image_size, image_size, mean, std, len(train_ds), len(val_loader.dataset),
    )
    trainer.fit(train_loader, val_loader)


if __name__ == "__main__":
    main()
