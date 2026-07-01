#!/usr/bin/env python
"""Training entrypoint for the segmentation trainer. **PLACEHOLDER.**

Declarative, exactly like ``../classification/train.py``: parse args, load +
override config, timestamp the run dir, seed, build model (which dictates
preprocessing), build dataloaders to match, then ``Trainer(...).fit(...)``.

    python train.py --config configs/unet.yaml
    python train.py --config configs/unet.yaml optim.lr=5e-4 model.encoder=resnet50
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
    parser = argparse.ArgumentParser(description="Train the segmentation model.")
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
    # Structure mirrors classification/train.py — see it for the reference flow.
    raise NotImplementedError(
        "Wire up like classification/train.py: load_config -> apply_overrides -> "
        "timestamped run.output_dir -> set_seed -> build_model (gives data_config) "
        "-> build_dataloaders(mean/std/image_size from data_config) -> Trainer.fit."
    )


if __name__ == "__main__":
    main()
