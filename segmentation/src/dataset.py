"""Dataset/dataloader for segmentation (image + mask). **PLACEHOLDER.**

This is the module that differs most from classification (contract #4). The
image and its mask must go through the SAME geometric transforms but the mask
must NOT be normalized — it stays an integer label map. albumentations does
this automatically when the mask is passed to the same Compose call:

    out = transform(image=image, mask=mask)
    image, mask = out["image"], out["mask"]

Reuse the shared image read (``load_image``: cv2 IMREAD_UNCHANGED, alpha
composite onto pad background, optional pad-to-square) exactly as classification
does — copy it from ``../../classification/src/dataset.py``. The mask is read as
a single-channel label map and padded with ``ignore_index`` (not the image pad
value) so padded regions are excluded from loss and metrics.
"""

from __future__ import annotations

import albumentations as A

from .config import AugConfig, Config, DataConfig


def load_image(path, to_rgb, pad_square, pad_value):
    """Shared image read — identical to classification. Copy it verbatim."""
    raise NotImplementedError("Copy load_image from ../../classification/src/dataset.py")


def load_mask(path, pad_square, ignore_index):
    """Read an HxW integer label map; pad with ``ignore_index`` if squaring.

    Read the mask as-is (no RGB conversion, no normalization). Pad with
    ``ignore_index`` so padded pixels don't count toward loss/metrics.
    """
    raise NotImplementedError("Read single-channel mask; pad with ignore_index.")


def build_transforms(aug: AugConfig, image_size: int, mean, std, train: bool) -> A.Compose:
    """Compose the albumentations pipeline; apply the same geometry to the mask.

    Geometric ops (flip/rotate/crop) apply to image AND mask (albumentations
    routes them when the Compose is called with mask=...). Photometric ops
    (blur/brightness) and Normalize apply to the image only. End with
    Normalize(image) + ToTensorV2 — the mask comes out as a long tensor of
    class indices. See classification's build_transforms for the resize/normalize
    tail shared by both splits.
    """
    raise NotImplementedError(
        "Build image+mask transforms. Reuse classification's train/val split "
        "and resize/normalize tail; add mask handling per the docstring."
    )


def build_dataloaders(cfg: Config, mean, std, image_size):
    """Build (train_loader, val_loader, train_ds). Mirror classification.

    The Dataset __getitem__ returns (image_tensor, mask_long_tensor). Otherwise
    the dataloader plumbing (num_workers, pin_memory, persistent_workers,
    cv2.setNumThreads(0)) is identical to classification.
    """
    raise NotImplementedError("Port build_dataloaders from classification; yield (image, mask).")
