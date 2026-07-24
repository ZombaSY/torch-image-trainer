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

import math
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

# Default composite/pad background for the RGB input: white. Applied whenever
# the random-background aug does not fire (and always at val/inference), so
# train and test see the same background unless the aug deliberately varies it.
WHITE_BG = (255, 255, 255)


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
        # Random crop at the model input size: keeps native scale (the resize
        # below becomes a no-op); smaller samples are padded at a random
        # position (white bg, transparent mask) for a translation aug instead.
        # When the crop doesn't fire, the resize handles the full square.
        if aug.random_crop:
            ops.append(
                A.RandomCrop(
                    image_size, image_size, pad_if_needed=True,
                    pad_position="random", border_mode=cv2.BORDER_CONSTANT,
                    fill=WHITE_BG, fill_mask=0, p=aug.random_crop_p,
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


def _tile_positions(dim: int, tile: int, overlap: int) -> list[int]:
    """Evenly spaced tile starts covering ``[0, dim)`` with at least *overlap*
    pixels shared between neighbours. Ported from ``logo_matter._tile_positions``
    (the inference tiler) so train/val decompose an image exactly as inference
    does."""
    if dim <= tile:
        return [0]
    n = math.ceil((dim - overlap) / (tile - overlap))
    return [round(i * (dim - tile) / (n - 1)) for i in range(n)]


class MatteDataset(Dataset):
    """CSV-driven alpha-matte dataset.

    Each row holds an RGBA image path and a grayscale matte path (both relative
    to ``data.root``). ``__getitem__`` returns ``(image, alpha)`` where ``image``
    is a normalized ``(3, H, W)`` float tensor and ``alpha`` is a ``(1, H, W)``
    float tensor in ``[0, 1]``.

    With ``data.tiling`` on, each image is expanded into a grid of native-scale
    ``image_size`` square tiles (see :func:`_tile_positions`); the sample list
    becomes one entry per ``(image, tile)`` so both splits get deterministic
    full coverage at source resolution instead of a downscaled square. Train
    tiles additionally get isotropic scale/position jitter (``tile_scale_*``);
    val tiles are exact native crops.
    """

    def __init__(
        self, csv_path, data_cfg: DataConfig, aug_cfg: AugConfig,
        transform: A.Compose, train: bool, image_size: int,
    ):
        self.root = Path(data_cfg.root)
        self.cfg = data_cfg
        self.aug = aug_cfg
        self.transform = transform
        self.train = train
        self.image_size = image_size
        self.tiling = data_cfg.tiling

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

        # Flat (image_idx, tile_y, tile_x) index for tiling; None otherwise (one
        # sample per image). Built from each image's native size after the cache
        # so tile counts come from the real resolution.
        self.tiles: list[tuple[int, int, int]] | None = None
        if self.tiling:
            self.tiles = self._build_tile_index()

    def __len__(self) -> int:
        return len(self.tiles) if self.tiling else len(self.images)

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
        return list(WHITE_BG)

    def _load_raw(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        """Composited RGB input + alpha matte (pre-aug).

        Square-padded to ``pad_to_square`` normally; under tiling the padding is
        bypassed so tiles are cropped from the native-resolution composite (the
        tile grid handles edges via per-tile padding instead)."""
        bg = self._sample_background()
        if self.cache is not None:
            # Copies so no downstream op can ever mutate the cached arrays.
            raw_image = self.cache[idx][0].copy()
            raw_mask = self.cache[idx][1].copy()
        else:
            raw_image, raw_mask = self._read_pair(idx)
        pad_square = self.cfg.pad_to_square and not self.tiling
        image = process_image(
            raw_image, bg,
            self.cfg.to_rgb, pad_square, self.cfg.unpremultiply_alpha,
        )
        if pad_square:
            raw_mask = pad_to_square(raw_mask, self.cfg.mask_pad_value)
        return image, raw_mask

    def _image_hw(self, idx: int) -> tuple[int, int]:
        """Native (H, W) of image *idx* — from the cache when present, else a
        one-time raw read. Compositing/tiling never change these dims."""
        if self.cache is not None:
            return self.cache[idx][0].shape[:2]
        return read_raw(self.root / self.images[idx]).shape[:2]

    def _build_tile_index(self) -> list[tuple[int, int, int]]:
        """Expand every image into its ``(image_idx, tile_y, tile_x)`` grid."""
        tile, overlap = self.image_size, self.cfg.tile_overlap
        index: list[tuple[int, int, int]] = []
        for i in range(len(self.images)):
            h, w = self._image_hw(i)
            for y in _tile_positions(h, tile, overlap):
                for x in _tile_positions(w, tile, overlap):
                    index.append((i, y, x))
        return index

    def _make_tile(
        self, image: np.ndarray, mask: np.ndarray, y: int, x: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Crop one square tile from a native-resolution composite.

        Val (no jitter): an exact ``image_size`` window at ``(y, x)``, edges
        center-padded — matches ``logo_matter._predict_alpha_tiled``. Train:
        isotropic enlarge (a smaller ``image_size / s`` window, ``s`` in
        ``[1, tile_scale_max]``, upscaled by the resize tail) plus random
        window placement; sub-window content is padded (never stretched, never
        downscaled). Returns a square ``win x win`` image+mask; the transform
        tail's ``A.Resize`` scales it to ``image_size``.
        """
        tile = self.image_size

        # Isotropic enlarge: shrink the sampled window so the resize tail
        # upscales it (object appears larger). win <= tile, so never a downscale.
        win = tile
        if (
            self.train and self.aug.tile_scale_jitter
            and float(np.random.rand()) < self.aug.tile_scale_jitter_p
        ):
            s = float(np.random.uniform(1.0, self.aug.tile_scale_max))
            win = max(1, min(tile, int(round(tile / s))))

        # Window top-left in native coords. Train: random placement within the
        # tile extent (translation). Val / no jitter: aligned to (y, x).
        if self.train and win < tile:
            y = y + int(np.random.randint(0, tile - win + 1))
            x = x + int(np.random.randint(0, tile - win + 1))

        return self._crop_window(image, mask, y, x, win)

    def _crop_window(
        self, image: np.ndarray, mask: np.ndarray, y: int, x: int, win: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Crop a ``win x win`` window at native scale, padding out-of-bounds
        regions (image -> white, mask -> ``mask_pad_value``). Sub-``win`` content
        is placed at a random position when ``tile_random_pad`` (train), else
        centered."""
        h, w = image.shape[:2]
        sy0, sx0 = max(0, y), max(0, x)
        sy1, sx1 = min(h, y + win), min(w, x + win)
        ch, cw = sy1 - sy0, sx1 - sx0  # in-bounds content size

        if self.train and self.aug.tile_random_pad and (ch < win or cw < win):
            top = int(np.random.randint(0, win - ch + 1))
            left = int(np.random.randint(0, win - cw + 1))
        else:  # center-pad (val, and train when tile_random_pad is off)
            top, left = (win - ch) // 2, (win - cw) // 2

        img_tile = np.full((win, win, 3), WHITE_BG, dtype=image.dtype)
        img_tile[top:top + ch, left:left + cw] = image[sy0:sy1, sx0:sx1]
        mask_tile = np.full((win, win), self.cfg.mask_pad_value, dtype=mask.dtype)
        mask_tile[top:top + ch, left:left + cw] = mask[sy0:sy1, sx0:sx1]
        return img_tile, mask_tile

    def _load_sample(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        """One augmentable ``(image, mask)``: a native-scale tile crop under
        tiling (resolving the flat tile index), else the whole composite."""
        if self.tiling:
            img_idx, y, x = self.tiles[idx]
            image, mask = self._load_raw(img_idx)
            return self._make_tile(image, mask, y, x)
        return self._load_raw(idx)

    def __getitem__(self, idx: int):
        image, mask = self._load_sample(idx)

        # CutMix FIRST, before the albumentations pipeline, so the mixed
        # image+mask is then augmented as one coherent sample.
        if (
            self.train and self.aug.cutmix
            and float(np.random.rand()) < self.aug.cutmix_p
        ):
            j = int(np.random.randint(len(self)))
            image_b, mask_b = self._load_sample(j)
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

    train_ds = MatteDataset(Path(d.root) / d.train_csv, d, cfg.aug, train_tf, train=True, image_size=image_size)
    val_ds = MatteDataset(Path(d.root) / d.val_csv, d, cfg.aug, val_tf, train=False, image_size=image_size)

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
