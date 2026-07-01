#!/usr/bin/env python
"""Inference entrypoint for the segmentation trainer. **PLACEHOLDER.**

Mirrors ``../classification/test.py``: load a ``.pt`` checkpoint (which embeds
its own config — contract #2), rebuild the exact model + preprocessing, predict
a mask per image in a CSV, and write outputs. For segmentation the outputs are
predicted mask images (and, with labels, per-image / aggregate IoU-Dice).

    python test.py --checkpoint runs/unet/best.pt --save-masks
    python test.py --checkpoint runs/unet/best.pt --csv /path/to/test.csv
"""

from __future__ import annotations

import argparse

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained segmentation checkpoint.")
    parser.add_argument("--checkpoint", required=True, help="Path to a .pt checkpoint.")
    parser.add_argument("--csv", default=None, help="CSV of images (defaults to config's val.csv).")
    parser.add_argument("--data-root", default=None, help="Override dataset root.")
    parser.add_argument("--output-dir", default=None, help="Defaults to <checkpoint_dir>/test.")
    parser.add_argument("--batch-size", type=int, default=8, help="Inference batch size.")
    parser.add_argument("--save-masks", action="store_true", help="Write predicted mask images.")
    parser.add_argument("--device", default="cuda", help="cuda or cpu.")
    return parser.parse_args()


def load_checkpoint(path: str, device: torch.device):
    """Load ckpt and rebuild the model from its embedded config (contract #2).

    Identical in spirit to classification/test.py: torch.load(weights_only=True),
    flip model.pretrained=False, from_dict -> build_model -> load_state_dict.
    """
    raise NotImplementedError("Port load_checkpoint from classification/test.py.")


def main() -> None:
    raise NotImplementedError(
        "Wire up like classification/test.py: load_checkpoint -> build val "
        "transform from the embedded config -> predict masks -> write mask PNGs "
        "and (if the CSV has masks) segmentation_metrics."
    )


if __name__ == "__main__":
    main()
