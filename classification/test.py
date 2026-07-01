#!/usr/bin/env python
"""Run a trained checkpoint over a CSV of images and save predictions.

Loads a ``.pt`` checkpoint (which embeds its own config), predicts the style
class for every image listed in a CSV, and writes:

  * ``predictions.csv`` — one row per image with predicted class, confidence,
    per-class probabilities, and (if the CSV has labels) correctness.
  * ``images/`` — copies of each image annotated with the predicted class
    text (only with ``--save-images``).

Examples
--------
    python test.py --checkpoint runs/linear_probe_vit/best.pt --save-images
    python test.py --checkpoint runs/full_finetune_vit/best.pt \
        --csv /path/to/test.csv --output-dir runs/full_finetune_vit/test
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

from src.config import from_dict
from src.dataset import build_transforms, load_image  # load_image: shared read/composite
from src.model import build_model
from src.utils import classification_metrics, get_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained checkpoint.")
    parser.add_argument("--checkpoint", required=True, help="Path to a .pt checkpoint.")
    parser.add_argument(
        "--csv", default=None,
        help="CSV of images to score. Defaults to the val.csv from the config.",
    )
    parser.add_argument(
        "--data-root", default=None,
        help="Override the dataset root the CSV paths are relative to.",
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Where to write outputs. Defaults to <checkpoint_dir>/test.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=16, help="Inference batch size."
    )
    parser.add_argument(
        "--save-images", action="store_true",
        help="Also save each image annotated with the predicted class text.",
    )
    parser.add_argument("--device", default="cuda", help="cuda or cpu.")
    return parser.parse_args()


def load_checkpoint(path: str, device: torch.device):
    """Load a checkpoint and rebuild the model with its embedded config."""
    # Our checkpoints hold only tensors + primitive config/metrics, so the
    # safe loader is sufficient (avoids arbitrary-code-execution on unpickle).
    ckpt = torch.load(path, map_location=device, weights_only=True)
    raw_cfg = dict(ckpt["config"])
    # Skip re-downloading backbone weights; they are restored from the ckpt.
    raw_cfg["model"] = {**raw_cfg["model"], "pretrained": False}
    cfg = from_dict(raw_cfg)

    model, data_config = build_model(cfg)
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    return model, cfg, data_config, ckpt.get("metrics", {})


@torch.no_grad()
def predict(model, cfg, transform, device, root, rel_paths, batch_size):
    """Return (pred_indices, probabilities) for every image path."""
    preds, probs = [], []
    batch, idx = [], 0
    n = len(rel_paths)

    def flush(tensors):
        x = torch.stack(tensors).to(device)
        logits = model(x)
        p = torch.softmax(logits, dim=1).cpu().numpy()
        probs.extend(p)
        preds.extend(p.argmax(1).tolist())

    for rel in rel_paths:
        image = load_image(
            root / rel, cfg.data.to_rgb, cfg.data.pad_to_square, cfg.data.pad_value
        )
        batch.append(transform(image=image)["image"])
        idx += 1
        if len(batch) == batch_size or idx == n:
            flush(batch)
            batch = []
    return preds, np.asarray(probs)


def annotate(src_path: Path, dst_path: Path, text: str, cfg) -> None:
    """Draw a labeled banner with the predicted class onto a copy of the image.

    Reuses ``load_image`` so the saved preview reflects exactly what the model
    ingested (alpha composited onto the pad background, padded to square). Reads
    as BGR (``to_rgb=False``) so cv2's drawing and ``imwrite`` colors stay correct.
    """
    try:
        image = load_image(
            src_path, to_rgb=False,
            pad_square=cfg.data.pad_to_square, pad_value=cfg.data.pad_value,
        )
    except FileNotFoundError:
        return
    h, w = image.shape[:2]
    scale = max(0.6, w / 900)
    thickness = max(1, int(scale * 2))
    (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    pad = int(8 * scale)
    cv2.rectangle(image, (0, 0), (tw + 2 * pad, th + base + 2 * pad), (0, 0, 0), -1)
    cv2.putText(
        image, text, (pad, th + pad), cv2.FONT_HERSHEY_SIMPLEX,
        scale, (0, 255, 0), thickness, cv2.LINE_AA,
    )
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dst_path), image)


def main() -> None:
    args = parse_args()
    device = torch.device(
        args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu"
    )

    model, cfg, data_config, train_metrics = load_checkpoint(args.checkpoint, device)

    root = Path(args.data_root or cfg.data.root)
    csv_path = Path(args.csv) if args.csv else root / cfg.data.val_csv
    output_dir = Path(args.output_dir or (Path(args.checkpoint).parent / "test"))
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = get_logger("test", output_dir / "test.log")

    image_size = cfg.model.image_size or data_config["input_size"][1]
    transform = build_transforms(
        cfg.aug, image_size, tuple(data_config["mean"]), tuple(data_config["std"]),
        train=False,
    )

    frame = pd.read_csv(csv_path)
    img_col, lbl_col = cfg.data.image_column, cfg.data.label_column
    if img_col not in frame.columns:
        raise ValueError(f"Column {img_col!r} missing from {csv_path}")
    rel_paths = frame[img_col].astype(str).tolist()
    has_labels = lbl_col in frame.columns

    logger.info(
        "Checkpoint %s | backbone %s | %d images from %s",
        args.checkpoint, cfg.model.backbone, len(rel_paths), csv_path,
    )

    preds, probs = predict(
        model, cfg, transform, device, root, rel_paths, args.batch_size
    )
    class_names = cfg.data.class_names

    out = pd.DataFrame({img_col: rel_paths})
    out["pred_class"] = preds
    out["pred_name"] = [class_names[p] for p in preds]
    out["confidence"] = [round(float(probs[i, p]), 4) for i, p in enumerate(preds)]
    for c, name in enumerate(class_names):
        out[f"prob_{name}"] = np.round(probs[:, c], 4)

    if has_labels:
        targets = frame[lbl_col].astype(int).tolist()
        out["true_class"] = targets
        out["true_name"] = [class_names[t] for t in targets]
        out["correct"] = [int(p == t) for p, t in zip(preds, targets)]
        metrics = classification_metrics(
            np.asarray(targets), np.asarray(preds), cfg.data.num_classes
        )
        logger.info(
            "Eval | acc %.4f | macro_f1 %.4f | per_class_f1 %s",
            metrics["accuracy"], metrics["macro_f1"], metrics["per_class_f1"],
        )

    csv_out = output_dir / "predictions.csv"
    out.to_csv(csv_out, index=False)
    logger.info("Wrote predictions: %s", csv_out)

    if args.save_images:
        img_dir = output_dir / "images"
        for i, rel in enumerate(rel_paths):
            label = f"{out['pred_name'][i]} {out['confidence'][i]:.2f}"
            if has_labels and not out["correct"][i]:
                label += f" (gt:{out['true_name'][i]})"
            # Flatten relative path so nested dirs do not collide.
            flat = rel.replace("/", "__")
            annotate(root / rel, img_dir / flat, label, cfg)
        logger.info("Wrote %d annotated images: %s", len(rel_paths), img_dir)


if __name__ == "__main__":
    main()
