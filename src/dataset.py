"""Dataset and dataloader built on cv2 + albumentations.

Images are read with OpenCV, optionally padded to a square (matching the WD
tagger preprocessing), then run through an albumentations pipeline. Only the
four requested train-time augmentations are enabled: horizontal flip, coarse
dropout, blur, and rotate.
"""

from __future__ import annotations

from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from .config import AugConfig, Config, DataConfig

# Keep OpenCV from spawning its own threads inside dataloader workers.
cv2.setNumThreads(0)


def pad_to_square(image: np.ndarray, value: int) -> np.ndarray:
    """Pad an HxWxC image to a centered square with a constant border."""
    h, w = image.shape[:2]
    if h == w:
        return image
    size = max(h, w)
    top = (size - h) // 2
    bottom = size - h - top
    left = (size - w) // 2
    right = size - w - left
    return cv2.copyMakeBorder(
        image, top, bottom, left, right,
        borderType=cv2.BORDER_CONSTANT, value=(value, value, value),
    )


def flatten_alpha(image: np.ndarray, value: int) -> np.ndarray:
    """Flatten a raw cv2 read onto a constant background, returning BGR uint8.

    Accepts the ``cv2.IMREAD_UNCHANGED`` result (grayscale, BGR, or BGRA) and
    alpha-composites transparent pixels onto a solid ``value`` background. This
    matters because the WD taggers expect a white background: dropping the
    alpha channel (as ``IMREAD_COLOR`` does) leaks whatever arbitrary RGB sits
    under transparent pixels, which shows up as a "splatted" background.
    """
    if image.ndim == 2:  # grayscale
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    channels = image.shape[2]
    if channels == 3:
        return image
    if channels == 4:
        bgr = image[:, :, :3].astype(np.float32)
        alpha = (image[:, :, 3].astype(np.float32) / 255.0)[:, :, None]
        composited = bgr * alpha + value * (1.0 - alpha)
        return composited.round().astype(np.uint8)
    raise ValueError(f"Unsupported channel count {channels} for image")


def load_image(
    path: str | Path, to_rgb: bool, pad_square: bool, pad_value: int
) -> np.ndarray:
    """Read an image with cv2 and apply the shared preprocessing.

    Returns an HxWx3 uint8 array (RGB if ``to_rgb`` else BGR). Centralized so
    training and inference use identical preprocessing. Reads unchanged so the
    alpha channel can be composited onto ``pad_value`` rather than discarded.
    """
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    image = flatten_alpha(image, pad_value)  # BGR uint8, transparency on bg
    if to_rgb:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if pad_square:
        image = pad_to_square(image, pad_value)
    return image


def build_transforms(
    aug: AugConfig, image_size: int, mean: tuple, std: tuple, train: bool
) -> A.Compose:
    """Compose the albumentations pipeline for a split.

    Augmentations are applied only when ``train`` is True; both splits share
    the same resize + normalize tail so train/val stay consistent.
    """
    ops: list[A.BasicTransform] = []

    if train:
        if aug.horizontal_flip:
            ops.append(A.HorizontalFlip(p=aug.horizontal_flip_p))
        if aug.rotate:
            ops.append(
                A.Rotate(
                    limit=aug.rotate_limit,
                    border_mode=cv2.BORDER_CONSTANT,
                    p=aug.rotate_p,
                )
            )
        if aug.blur:
            ops.append(A.Blur(blur_limit=aug.blur_limit, p=aug.blur_p))
        if aug.coarse_dropout:
            ops.append(
                A.CoarseDropout(
                    num_holes_range=(1, aug.coarse_dropout_max_holes),
                    hole_height_range=(0.02, aug.coarse_dropout_max_height_frac),
                    hole_width_range=(0.02, aug.coarse_dropout_max_width_frac),
                    p=aug.coarse_dropout_p,
                )
            )

    ops.append(A.Resize(image_size, image_size))
    ops.append(A.Normalize(mean=mean, std=std))
    ops.append(ToTensorV2())
    return A.Compose(ops)


class StyleDataset(Dataset):
    """CSV-driven image classification dataset.

    The CSV holds an image path (relative to ``data.root``) and an integer
    class label per row.
    """

    def __init__(self, csv_path: str | Path, data_cfg: DataConfig, transform: A.Compose):
        self.root = Path(data_cfg.root)
        self.cfg = data_cfg
        self.transform = transform

        frame = pd.read_csv(csv_path)
        for col in (data_cfg.image_column, data_cfg.label_column):
            if col not in frame.columns:
                raise ValueError(f"Column {col!r} missing from {csv_path}")
        self.paths = frame[data_cfg.image_column].astype(str).tolist()
        self.labels = frame[data_cfg.label_column].astype(int).tolist()

        bad = [l for l in self.labels if not 0 <= l < data_cfg.num_classes]
        if bad:
            raise ValueError(f"Labels out of range [0,{data_cfg.num_classes}): {set(bad)}")

    def __len__(self) -> int:
        return len(self.paths)

    @property
    def targets(self) -> list[int]:
        return self.labels

    def _read_image(self, rel_path: str) -> np.ndarray:
        return load_image(
            self.root / rel_path,
            self.cfg.to_rgb, self.cfg.pad_to_square, self.cfg.pad_value,
        )

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        image = self._read_image(self.paths[idx])
        image = self.transform(image=image)["image"]
        return image, self.labels[idx]


def class_weights(labels: list[int], num_classes: int) -> torch.Tensor:
    """Inverse-frequency class weights, normalized to mean 1."""
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    counts = np.clip(counts, 1.0, None)
    weights = counts.sum() / (num_classes * counts)
    return torch.tensor(weights, dtype=torch.float32)


def _make_sampler(labels: list[int], num_classes: int) -> WeightedRandomSampler:
    per_class = class_weights(labels, num_classes)
    sample_weights = [per_class[l].item() for l in labels]
    return WeightedRandomSampler(sample_weights, num_samples=len(labels), replacement=True)


def build_dataloaders(
    cfg: Config, mean: tuple, std: tuple, image_size: int
) -> tuple[DataLoader, DataLoader, StyleDataset]:
    """Build the train and val dataloaders. Returns (train, val, train_ds)."""
    d = cfg.data
    train_tf = build_transforms(cfg.aug, image_size, mean, std, train=True)
    val_tf = build_transforms(cfg.aug, image_size, mean, std, train=False)

    train_ds = StyleDataset(Path(d.root) / d.train_csv, d, train_tf)
    val_ds = StyleDataset(Path(d.root) / d.val_csv, d, val_tf)

    sampler = None
    shuffle = True
    if cfg.optim.weighted_sampler:
        sampler = _make_sampler(train_ds.targets, d.num_classes)
        shuffle = False

    common = dict(
        num_workers=cfg.optim.num_workers,
        pin_memory=True,
        persistent_workers=cfg.optim.num_workers > 0,
    )
    train_loader = DataLoader(
        train_ds, batch_size=cfg.optim.batch_size,
        shuffle=shuffle, sampler=sampler, drop_last=False, **common,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.optim.batch_size, shuffle=False, **common,
    )
    return train_loader, val_loader, train_ds
