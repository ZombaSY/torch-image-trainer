"""Dataset/dataloader for alpha-matte regression (cv2 + albumentations).

Input images are RGBA logos; the target is the logo's alpha channel saved as a
grayscale matte. The model must predict alpha from an *RGB* composite, so the
reader composites the logo onto a background and drops the alpha it has to
learn.

Two matte-specific pieces beyond the classification reader:

* **Random-background augmentation** (train only). Instead of always compositing
  onto white, composite onto a random solid color so the model learns alpha
  independent of background — the key robustness aug for matting. The target
  matte is unchanged.
* **Unpremultiply** (optional). If the source RGBA is premultiplied
  (``rgb = color * alpha``), recover the straight color (``color = rgb / alpha``)
  before compositing so the result looks natural on any background. Toggle via
  ``data.unpremultiply_alpha``. NOTE: these logos look mostly *straight*-alpha,
  where unpremultiply over-brightens edges — leave it off unless you know the
  assets are premultiplied.

Geometric aug (flips, rotate) applies to image AND mask; photometric aug
(color jitter, blur, coarse dropout) applies to the image only. albumentations
routes this automatically when the mask is passed to the same Compose call.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Dataset

from .config import AugConfig, Config, DataConfig

# Keep OpenCV from spawning its own threads inside dataloader workers.
cv2.setNumThreads(0)

_ALPHA_EPS = 1.0 / 255.0


def pad_to_square(image: np.ndarray, value) -> np.ndarray:
    """Pad an HxWxC (or HxW) image to a centered square with a constant border."""
    h, w = image.shape[:2]
    if h == w:
        return image
    size = max(h, w)
    top = (size - h) // 2
    bottom = size - h - top
    left = (size - w) // 2
    right = size - w - left
    return cv2.copyMakeBorder(
        image, top, bottom, left, right, borderType=cv2.BORDER_CONSTANT, value=value,
    )


def composite_rgba(image: np.ndarray, bg_bgr, unpremultiply: bool) -> np.ndarray:
    """Composite a raw cv2 read onto a solid ``bg_bgr`` background -> BGR uint8.

    Accepts the ``cv2.IMREAD_UNCHANGED`` result (grayscale, BGR, or BGRA). For
    BGRA, alpha-composites onto ``bg_bgr``; when ``unpremultiply`` is set, the
    straight color is recovered first (``rgb / alpha``) so premultiplied assets
    composite naturally.
    """
    if image.ndim == 2:  # grayscale -> opaque BGR
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    channels = image.shape[2]
    if channels == 3:
        return image
    if channels != 4:
        raise ValueError(f"Unsupported channel count {channels} for image")

    bgr = image[:, :, :3].astype(np.float32)
    alpha = (image[:, :, 3].astype(np.float32) / 255.0)[:, :, None]
    if unpremultiply:
        bgr = np.clip(bgr / np.clip(alpha, _ALPHA_EPS, None), 0.0, 255.0)
    bg = np.asarray(bg_bgr, dtype=np.float32).reshape(1, 1, 3)
    composited = bgr * alpha + bg * (1.0 - alpha)
    return composited.round().astype(np.uint8)


def read_raw(path, flags: int = cv2.IMREAD_UNCHANGED) -> np.ndarray:
    """Read a file with cv2, failing loudly on unreadable paths."""
    image = cv2.imread(str(path), flags)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return image


def process_image(
    raw: np.ndarray, bg_bgr, to_rgb: bool, pad_square: bool, unpremultiply: bool
) -> np.ndarray:
    """Composite/pad a raw cv2 read into an HxWx3 uint8 frame (RGB if ``to_rgb``).

    Compositing and square-padding both use ``bg_bgr`` so the whole frame shares
    one background. Split from :func:`load_image` so cached raw reads take the
    same path as fresh disk reads.
    """
    image = composite_rgba(raw, bg_bgr, unpremultiply)  # BGR uint8
    if pad_square:
        image = pad_to_square(image, [float(c) for c in bg_bgr])
    if to_rgb:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def load_image(
    path, bg_bgr, to_rgb: bool, pad_square: bool, unpremultiply: bool
) -> np.ndarray:
    """Read an RGBA logo and return an HxWx3 uint8 composite (RGB if ``to_rgb``).

    Centralized so training and inference preprocess identically.
    """
    return process_image(read_raw(path), bg_bgr, to_rgb, pad_square, unpremultiply)


def load_mask(path, pad_square: bool, pad_value: int) -> np.ndarray:
    """Read a grayscale alpha matte as HxW uint8; pad with ``pad_value`` (0)."""
    mask = read_raw(path, cv2.IMREAD_GRAYSCALE)
    if pad_square:
        mask = pad_to_square(mask, pad_value)
    return mask


def build_transforms(
    aug: AugConfig, image_size: int, mean: tuple, std: tuple, train: bool
) -> A.Compose:
    """Compose the albumentations pipeline; geometry is shared with the mask.

    Only the image is normalized; the mask stays a raw label map (converted to a
    float alpha tensor in the dataset). Both splits share the resize + normalize
    tail so train/val geometry stays consistent.
    """
    ops: list[A.BasicTransform] = []

    if train:
        # Geometric — applied to image AND mask.
        if aug.horizontal_flip:
            ops.append(A.HorizontalFlip(p=aug.horizontal_flip_p))
        if aug.vertical_flip:
            ops.append(A.VerticalFlip(p=aug.vertical_flip_p))
        if aug.rotate:
            ops.append(
                A.Rotate(
                    limit=aug.rotate_limit, border_mode=cv2.BORDER_CONSTANT,
                    fill=0, fill_mask=0, p=aug.rotate_p,
                )
            )
        # Photometric — image only (mask untouched).
        if aug.color_jitter:
            ops.append(
                A.ColorJitter(
                    brightness=aug.color_jitter_brightness,
                    contrast=aug.color_jitter_contrast,
                    saturation=aug.color_jitter_saturation,
                    hue=aug.color_jitter_hue,
                    p=aug.color_jitter_p,
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


def cutmix_pair(
    image_a: np.ndarray, mask_a: np.ndarray,
    image_b: np.ndarray, mask_b: np.ndarray, alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Paste a random patch from sample B onto A, on the image AND the matte.

    For a dense regression target there is no label to mix — cutting the same
    box on the matte keeps it pixel-exact. B is resized to A's size first so the
    patch lines up. The box covers ~``(1 - lam)`` of the area with
    ``lam ~ Beta(alpha, alpha)`` and is placed fully in-frame. Returns modified
    copies (the inputs are not mutated).
    """
    h, w = image_a.shape[:2]
    if image_b.shape[:2] != (h, w):
        image_b = cv2.resize(image_b, (w, h), interpolation=cv2.INTER_LINEAR)
        mask_b = cv2.resize(mask_b, (w, h), interpolation=cv2.INTER_NEAREST)

    lam = float(np.random.beta(alpha, alpha)) if alpha > 0 else 0.5
    cut_h = min(int(round(h * np.sqrt(1.0 - lam))), h)
    cut_w = min(int(round(w * np.sqrt(1.0 - lam))), w)
    if cut_h == 0 or cut_w == 0:
        return image_a, mask_a
    y1 = int(np.random.randint(0, h - cut_h + 1))
    x1 = int(np.random.randint(0, w - cut_w + 1))

    image = image_a.copy()
    mask = mask_a.copy()
    image[y1:y1 + cut_h, x1:x1 + cut_w] = image_b[y1:y1 + cut_h, x1:x1 + cut_w]
    mask[y1:y1 + cut_h, x1:x1 + cut_w] = mask_b[y1:y1 + cut_h, x1:x1 + cut_w]
    return image, mask


class MatteDataset(Dataset):
    """CSV-driven alpha-matte dataset.

    Each row holds an RGBA image path and a grayscale matte path (both relative
    to ``data.root``). ``__getitem__`` returns ``(image, alpha)`` where ``image``
    is a normalized ``(3, H, W)`` float tensor and ``alpha`` is a ``(1, H, W)``
    float tensor in ``[0, 1]``.
    """

    def __init__(self, csv_path, data_cfg: DataConfig, aug_cfg: AugConfig, transform: A.Compose, train: bool):
        self.root = Path(data_cfg.root)
        self.cfg = data_cfg
        self.aug = aug_cfg
        self.transform = transform
        self.train = train

        frame = pd.read_csv(csv_path)
        for col in (data_cfg.image_column, data_cfg.mask_column):
            if col not in frame.columns:
                raise ValueError(f"Column {col!r} missing from {csv_path}")
        self.images = frame[data_cfg.image_column].astype(str).tolist()
        self.masks = frame[data_cfg.mask_column].astype(str).tolist()

        # In-memory cache of the *raw decoded* reads (RGBA image + matte). The
        # cache sits before compositing because the background color is sampled
        # per access (random_background aug). Built eagerly in the main process
        # so forked dataloader workers share the arrays copy-on-write.
        self.cache: list[tuple[np.ndarray, np.ndarray]] | None = None
        if data_cfg.cache_in_memory:
            with ThreadPoolExecutor() as pool:
                self.cache = list(pool.map(self._read_pair, range(len(self.images))))

    def __len__(self) -> int:
        return len(self.images)

    def _read_pair(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        """Raw decoded (image, matte) pair straight from disk, pre-composite."""
        return (
            read_raw(self.root / self.images[idx]),
            read_raw(self.root / self.masks[idx], cv2.IMREAD_GRAYSCALE),
        )

    def _sample_background(self) -> list[int]:
        """White by default; a random solid color when the aug fires (train)."""
        if (
            self.train and self.aug.random_background
            and float(np.random.rand()) < self.aug.random_background_p
        ):
            return [int(c) for c in np.random.randint(0, 256, size=3)]
        return [self.cfg.pad_value] * 3

    def _load_raw(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        """Composited RGB input + alpha matte at padded-square size (pre-aug)."""
        bg = self._sample_background()
        if self.cache is not None:
            # Copies so no downstream op can ever mutate the cached arrays.
            raw_image = self.cache[idx][0].copy()
            raw_mask = self.cache[idx][1].copy()
        else:
            raw_image, raw_mask = self._read_pair(idx)
        image = process_image(
            raw_image, bg,
            self.cfg.to_rgb, self.cfg.pad_to_square, self.cfg.unpremultiply_alpha,
        )
        if self.cfg.pad_to_square:
            raw_mask = pad_to_square(raw_mask, self.cfg.mask_pad_value)
        return image, raw_mask

    def __getitem__(self, idx: int):
        image, mask = self._load_raw(idx)

        # CutMix FIRST, before the albumentations pipeline, so the mixed
        # image+mask is then augmented as one coherent sample.
        if (
            self.train and self.aug.cutmix
            and float(np.random.rand()) < self.aug.cutmix_p
        ):
            j = int(np.random.randint(len(self.images)))
            image_b, mask_b = self._load_raw(j)
            image, mask = cutmix_pair(image, mask, image_b, mask_b, self.aug.cutmix_alpha)

        out = self.transform(image=image, mask=mask)
        image_t = out["image"]
        mask_t = out["mask"]
        if mask_t.ndim == 2:  # ToTensorV2 leaves masks as (H, W)
            mask_t = mask_t.unsqueeze(0)
        alpha = mask_t.float() / 255.0
        return image_t, alpha


def build_dataloaders(
    cfg: Config, mean: tuple, std: tuple, image_size: int
) -> tuple[DataLoader, DataLoader, MatteDataset]:
    """Build the train and val dataloaders. Returns (train, val, train_ds)."""
    d = cfg.data
    train_tf = build_transforms(cfg.aug, image_size, mean, std, train=True)
    val_tf = build_transforms(cfg.aug, image_size, mean, std, train=False)

    train_ds = MatteDataset(Path(d.root) / d.train_csv, d, cfg.aug, train_tf, train=True)
    val_ds = MatteDataset(Path(d.root) / d.val_csv, d, cfg.aug, val_tf, train=False)

    common = dict(
        num_workers=cfg.optim.num_workers,
        pin_memory=True,
        persistent_workers=cfg.optim.num_workers > 0,
    )
    # drop_last=True so a trailing size-1 batch can't hit BatchNorm's
    # "1 value per channel" error in the UPerHead PPM (1x1 pooled features).
    train_loader = DataLoader(
        train_ds, batch_size=cfg.optim.batch_size, shuffle=True, drop_last=True, **common,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.optim.batch_size, shuffle=False, **common,
    )
    return train_loader, val_loader, train_ds
