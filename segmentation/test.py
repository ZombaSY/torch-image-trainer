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
    python test.py --folders today            # only images under .../today/
    python test.py --folders today logo foreground-png
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
from src.dataset import WHITE_BG, build_transforms, load_image, load_mask  # shared preprocessing
from src.model import build_model
from src.utils import get_logger, matting_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained matte checkpoint.")
    parser.add_argument("--checkpoint", default="runs/sweep_260709-101240/trial_018/260710-145027/best_e259_mae0.0238.pt", help="Path to a .pt checkpoint.")
    parser.add_argument("--csv", default=None, help="CSV of images (defaults to config's val_csv).")
    parser.add_argument("--data-root", default=None, help="Override the dataset root.")
    parser.add_argument(
        "--folders", nargs="+", default=None,
        help="Only test images whose parent folder matches (e.g. today foreground-png logo).",
    )
    parser.add_argument("--output-dir", default=None, help="Defaults to <checkpoint_dir>/test.")
    parser.add_argument("--batch-size", type=int, default=8, help="Inference batch size.")
    parser.add_argument(
        "--save-comparison", action="store_true",
        help="Save per-image [input | prediction | ground truth] panels (needs labels).",
    )
    parser.add_argument(
        "--precision", default="fp32", choices=["fp32", "fp16"],
        help="Inference precision; fp16 uses CUDA autocast (same as training AMP).",
    )
    parser.add_argument("--device", default="cuda", help="cuda or cpu.")
    return parser.parse_args()


def _label(img: np.ndarray, text: str) -> np.ndarray:
    """Draw a small caption in the top-left corner (in place) and return it."""
    cv2.putText(img, text, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1, cv2.LINE_AA)
    return img


def comparison_panel(input_rgb: np.ndarray, pred_alpha: np.ndarray, gt_gray: np.ndarray) -> np.ndarray:
    """[input | prediction | ground truth] side-by-side, all at the pred size."""
    size = pred_alpha.shape[0]
    inp = cv2.cvtColor(cv2.resize(input_rgb, (size, size)), cv2.COLOR_RGB2BGR)
    pred3 = cv2.cvtColor(np.round(pred_alpha * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    gt3 = cv2.cvtColor(cv2.resize(gt_gray, (size, size), interpolation=cv2.INTER_NEAREST), cv2.COLOR_GRAY2BGR)
    sep = np.full((size, 4, 3), 255, np.uint8)
    return cv2.hconcat([_label(inp, "input"), sep, _label(pred3, "pred"), sep, _label(gt3, "gt")])


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
def predict(model, cfg, transform, device, root, rel_paths, batch_size, fp16=False):
    """Return a list of predicted alpha maps (HxW float in [0,1]), full-size per image."""
    # NOTE: images are resized to the network input; the returned matte is at
    # that resolution. Batch by identical shape (all square, same size here).
    out_alphas = []
    batch, sizes, idx, n = [], [], 0, len(rel_paths)

    def flush():
        x = torch.stack(batch).to(device)
        with torch.amp.autocast("cuda", enabled=fp16 and device.type == "cuda"):
            alpha = model(x)[:, 0].float().cpu().numpy()  # (B, H, W)
        out_alphas.extend(list(alpha))

    white = list(WHITE_BG)  # same default background the trainer uses
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
    # Folder-filtered runs get their own directory so full-set results survive.
    default_name = "test" if not args.folders else "test_" + "-".join(args.folders)
    output_dir = Path(args.output_dir or (Path(args.checkpoint).parent / default_name))
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
    if args.folders:
        parents = frame[img_col].astype(str).map(lambda p: Path(p).parent.name)
        keep = parents.isin(args.folders)
        if not keep.any():
            raise ValueError(
                f"No rows in {csv_path} under folders {args.folders}; "
                f"available: {sorted(parents.unique())}"
            )
        frame = frame[keep].reset_index(drop=True)
    rel_paths = frame[img_col].astype(str).tolist()
    has_labels = lbl_col in frame.columns

    logger.info(
        "Checkpoint %s | backbone %s + %s | %s | %d images from %s%s",
        args.checkpoint, cfg.model.backbone, cfg.head_name, args.precision,
        len(rel_paths), csv_path,
        f" (folders: {', '.join(args.folders)})" if args.folders else "",
    )

    alphas = predict(
        model, cfg, transform, device, root, rel_paths, args.batch_size,
        fp16=args.precision == "fp16",
    )

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

        if args.save_comparison:
            cmp_dir = output_dir / "comparison"
            cmp_dir.mkdir(parents=True, exist_ok=True)
            white = list(WHITE_BG)
            mask_rels = frame[lbl_col].astype(str).tolist()
            for rel, rel_mask, alpha in zip(rel_paths, mask_rels, alphas):
                inp = load_image(
                    root / rel, white, cfg.data.to_rgb, cfg.data.pad_to_square,
                    cfg.data.unpremultiply_alpha,
                )
                gt = load_mask(root / rel_mask, cfg.data.pad_to_square, cfg.data.mask_pad_value)
                panel = comparison_panel(inp, alpha, gt)
                stem = Path(rel.replace("/", "__")).with_suffix(".png").name
                cv2.imwrite(str(cmp_dir / stem), panel)
            logger.info("Wrote %d comparison panels: %s", len(alphas), cmp_dir)


if __name__ == "__main__":
    main()
