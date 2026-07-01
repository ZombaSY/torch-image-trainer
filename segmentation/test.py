#!/usr/bin/env python
"""Run a trained checkpoint over a CSV of images and save predicted mattes.

Loads a ``.pt`` checkpoint (which embeds its own config), predicts the alpha
matte for every image listed in a CSV, and writes:

  * ``<stem>.png`` mattes under ``masks/`` (predicted alpha, 0-255).
  * ``metrics.json`` with MAE/MSE when the CSV has a ground-truth ``label``
    column.

The checkpoint embeds its full config, so the exact model + preprocessing are
rebuilt automatically — no need to pass the original YAML.

Examples
--------
    python test.py --checkpoint runs/eva02_dpt/best.pt
    python test.py --checkpoint runs/swin_uper/best.pt \
        --csv /path/to/test.csv --output-dir runs/swin_uper/test
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

from src.config import from_dict
from src.dataset import build_transforms, load_image, load_mask  # shared preprocessing
from src.model import build_model
from src.utils import get_logger, matting_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained matte checkpoint.")
    parser.add_argument("--checkpoint", required=True, help="Path to a .pt checkpoint.")
    parser.add_argument("--csv", default=None, help="CSV of images (defaults to config's val_csv).")
    parser.add_argument("--data-root", default=None, help="Override the dataset root.")
    parser.add_argument("--output-dir", default=None, help="Defaults to <checkpoint_dir>/test.")
    parser.add_argument("--batch-size", type=int, default=8, help="Inference batch size.")
    parser.add_argument("--device", default="cuda", help="cuda or cpu.")
    return parser.parse_args()


def load_checkpoint(path: str, device: torch.device):
    """Load a checkpoint and rebuild the model with its embedded config."""
    ckpt = torch.load(path, map_location=device, weights_only=True)
    raw_cfg = dict(ckpt["config"])
    # Skip re-downloading backbone weights; they are restored from the ckpt.
    raw_cfg["model"] = {**raw_cfg["model"], "pretrained": False}
    cfg = from_dict(raw_cfg)

    model, data_config = build_model(cfg)
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    return model, cfg, data_config


@torch.no_grad()
def predict(model, cfg, transform, device, root, rel_paths, batch_size):
    """Return a list of predicted alpha maps (HxW float in [0,1]), full-size per image."""
    # NOTE: images are resized to the network input; the returned matte is at
    # that resolution. Batch by identical shape (all square, same size here).
    out_alphas = []
    batch, sizes, idx, n = [], [], 0, len(rel_paths)

    def flush():
        x = torch.stack(batch).to(device)
        alpha = model(x)[:, 0].float().cpu().numpy()  # (B, H, W)
        out_alphas.extend(list(alpha))

    white = [cfg.data.pad_value] * 3
    for rel in rel_paths:
        image = load_image(
            root / rel, white, cfg.data.to_rgb, cfg.data.pad_to_square,
            cfg.data.unpremultiply_alpha,
        )
        batch.append(transform(image=image)["image"])
        idx += 1
        if len(batch) == batch_size or idx == n:
            flush()
            batch = []
    return out_alphas


def main() -> None:
    args = parse_args()
    device = torch.device(
        args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu"
    )

    model, cfg, data_config = load_checkpoint(args.checkpoint, device)

    root = Path(args.data_root or cfg.data.root)
    csv_path = Path(args.csv) if args.csv else root / cfg.data.val_csv
    output_dir = Path(args.output_dir or (Path(args.checkpoint).parent / "test"))
    (output_dir / "masks").mkdir(parents=True, exist_ok=True)
    logger = get_logger("test", output_dir / "test.log")

    image_size = data_config["input_size"]
    transform = build_transforms(
        cfg.aug, image_size, tuple(data_config["mean"]), tuple(data_config["std"]), train=False,
    )

    frame = pd.read_csv(csv_path)
    img_col, lbl_col = cfg.data.image_column, cfg.data.mask_column
    if img_col not in frame.columns:
        raise ValueError(f"Column {img_col!r} missing from {csv_path}")
    rel_paths = frame[img_col].astype(str).tolist()
    has_labels = lbl_col in frame.columns

    logger.info(
        "Checkpoint %s | backbone %s + %s | %d images from %s",
        args.checkpoint, cfg.model.backbone, cfg.head_name, len(rel_paths), csv_path,
    )

    alphas = predict(model, cfg, transform, device, root, rel_paths, args.batch_size)

    # Save predicted mattes (flatten nested paths so they don't collide).
    for rel, alpha in zip(rel_paths, alphas):
        flat = rel.replace("/", "__")
        stem = Path(flat).with_suffix(".png").name
        cv2.imwrite(str(output_dir / "masks" / stem), np.round(alpha * 255).astype(np.uint8))
    logger.info("Wrote %d predicted mattes: %s", len(alphas), output_dir / "masks")

    if has_labels:
        all_pred, all_true = [], []
        for rel_mask, alpha in zip(frame[lbl_col].astype(str).tolist(), alphas):
            gt = load_mask(root / rel_mask, cfg.data.pad_to_square, cfg.data.mask_pad_value)
            gt = cv2.resize(gt, (alpha.shape[1], alpha.shape[0]), interpolation=cv2.INTER_NEAREST)
            all_pred.append(alpha.ravel())
            all_true.append(gt.ravel().astype(np.float32) / 255.0)
        metrics = matting_metrics(np.concatenate(all_true), np.concatenate(all_pred))
        logger.info("Eval | mae %.4f (%.2f/255) | mse %.4f",
                    metrics["mae"], metrics["mae_255"], metrics["mse"])
        with open(output_dir / "metrics.json", "w") as fh:
            json.dump(metrics, fh, indent=2)


if __name__ == "__main__":
    main()
