"""Dense-prediction decoder heads: UPerHead, DPTHead, and detail-capture variants.

All heads consume features from a backbone and emit a ``(B, out_channels, H, W)``
feature map (upsampled toward the input). The final 1-channel matte conv +
sigmoid lives in :class:`~src.model.SegModel`, so all heads share it.

* **UPerHead** — for *hierarchical* backbones (Swin-L, InternImage-L). Classic
  UPerNet: a Pyramid Pooling Module on the deepest stage + an FPN top-down
  fusion over all stages. Consumes the 4 native feature-pyramid maps.
* **DPTHead** — for *non-hierarchical* ViT backbones (DINOv2-L, EVA-02-L). Takes
  tokens reshaped to 2D grids from 4 transformer depths, "reassembles" them to
  4 resolutions (×4, ×2, ×1, ×0.5 of the token grid), then fuses them
  coarse-to-fine with RefineNet-style residual blocks (Ranftl et al., 2021).

The two heads above predict the matte from backbone-stride features and rely on
a final bilinear upsample of the 1-channel logit, which blurs alpha boundaries:
no true image-resolution detail ever reaches the matte. The *detail-capture*
heads fix that with ViTMatte's decoder recipe (Yao et al., Information Fusion
2024): a lightweight ``ConvStream`` over the raw input image supplies genuine
stride-2/4/8 high-frequency features, fused in during progressive upsampling so
the head's output is already at full input resolution (``needs_image = True``
tells :class:`SegModel` to pass the image alongside the backbone features).

* **ViTMatteHead** — for *non-hierarchical* ViT backbones. Merges the 4 token
  grids, adds DeepLabv3-style ASPP context (Chen et al., 2017), then fuses
  ConvStream details up to stride 1.
* **UPerMatteHead** — for *hierarchical* backbones. Reuses :class:`UPerHead`
  (its PPM already provides the global context) as the coarse path, then fuses
  ConvStream details from stride 4 up to stride 1.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _up(x: torch.Tensor, size, mode: str = "bilinear") -> torch.Tensor:
    return F.interpolate(x, size=size, mode=mode, align_corners=False)


class ConvModule(nn.Sequential):
    """Conv -> BN -> ReLU, the standard mmseg building block."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int = 3,
        padding: int | None = None,
        stride: int = 1,
        dilation: int = 1,
    ):
        if padding is None:
            padding = dilation * (kernel_size // 2)
        super().__init__(
            nn.Conv2d(
                in_ch, out_ch, kernel_size,
                stride=stride, padding=padding, dilation=dilation, bias=False,
            ),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )


# ---------------------------------------------------------------------------
# UPerHead (PPM + FPN) for hierarchical backbones
# ---------------------------------------------------------------------------
class PPM(nn.Module):
    """Pyramid Pooling Module: pool the deepest feature at several scales."""

    def __init__(self, in_ch: int, channels: int, pool_scales=(1, 2, 3, 6)):
        super().__init__()
        self.stages = nn.ModuleList(
            nn.Sequential(nn.AdaptiveAvgPool2d(s), ConvModule(in_ch, channels, 1))
            for s in pool_scales
        )

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        size = x.shape[2:]
        return [_up(stage(x), size) for stage in self.stages]


class UPerHead(nn.Module):
    def __init__(self, in_channels: list[int], channels: int, pool_scales=(1, 2, 3, 6)):
        super().__init__()
        self.out_channels = channels
        # PSP on the deepest stage.
        self.ppm = PPM(in_channels[-1], channels, pool_scales)
        self.ppm_bottleneck = ConvModule(
            in_channels[-1] + len(pool_scales) * channels, channels, 3
        )
        # FPN laterals + smoothing convs for the shallower stages.
        self.lateral_convs = nn.ModuleList(
            ConvModule(ch, channels, 1) for ch in in_channels[:-1]
        )
        self.fpn_convs = nn.ModuleList(
            ConvModule(channels, channels, 3) for _ in in_channels[:-1]
        )
        self.fpn_bottleneck = ConvModule(len(in_channels) * channels, channels, 3)

    def _psp_forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ppm_bottleneck(torch.cat([x, *self.ppm(x)], dim=1))

    def forward(self, feats: list[torch.Tensor]) -> torch.Tensor:
        laterals = [conv(feats[i]) for i, conv in enumerate(self.lateral_convs)]
        laterals.append(self._psp_forward(feats[-1]))

        # Top-down: add coarser levels into finer ones.
        for i in range(len(laterals) - 1, 0, -1):
            laterals[i - 1] = laterals[i - 1] + _up(laterals[i], laterals[i - 1].shape[2:])

        # Smooth all but the coarsest, then fuse at the finest resolution.
        fpn_outs = [self.fpn_convs[i](laterals[i]) for i in range(len(self.fpn_convs))]
        fpn_outs.append(laterals[-1])
        target = fpn_outs[0].shape[2:]
        fpn_outs = [_up(x, target) for x in fpn_outs]
        return self.fpn_bottleneck(torch.cat(fpn_outs, dim=1))


# ---------------------------------------------------------------------------
# DPTHead for non-hierarchical (ViT) backbones
# ---------------------------------------------------------------------------
class ResidualConvUnit(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv2(F.relu(self.conv1(F.relu(x))))
        return out + x


class FeatureFusionBlock(nn.Module):
    """RefineNet fusion: optionally add a skip feature, refine, upsample ×2."""

    def __init__(self, channels: int):
        super().__init__()
        self.rcu_skip = ResidualConvUnit(channels)
        self.rcu_out = ResidualConvUnit(channels)
        self.out_conv = nn.Conv2d(channels, channels, 1)

    def forward(self, x: torch.Tensor, skip: torch.Tensor | None = None) -> torch.Tensor:
        if skip is not None:
            x = x + self.rcu_skip(skip)
        x = self.rcu_out(x)
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        return self.out_conv(x)


class DPTHead(nn.Module):
    """Reassemble ViT tokens to 4 scales, then fuse coarse-to-fine.

    ``in_channels`` is the ViT embedding dim; inputs are 4 token grids
    ``(B, C, Hp, Wp)`` taken from increasing depth (shallow -> deep). The
    shallowest is upsampled most (×4), the deepest downsampled (×0.5), matching
    DPT's resolution assignment.
    """

    RESAMPLE = (4, 2, 1, 0.5)

    def __init__(self, in_channels: int, channels: int, n_inputs: int = 4):
        super().__init__()
        assert n_inputs == len(self.RESAMPLE), "DPTHead expects 4 token grids"
        self.out_channels = channels
        self.projects = nn.ModuleList(
            nn.Conv2d(in_channels, channels, 1) for _ in range(n_inputs)
        )
        self.resamples = nn.ModuleList(self._make_resample(channels, f) for f in self.RESAMPLE)
        self.fusions = nn.ModuleList(FeatureFusionBlock(channels) for _ in range(n_inputs))
        self.out_conv = ConvModule(channels, channels, 3)

    @staticmethod
    def _make_resample(channels: int, factor: float) -> nn.Module:
        if factor == 4:
            return nn.ConvTranspose2d(channels, channels, kernel_size=4, stride=4)
        if factor == 2:
            return nn.ConvTranspose2d(channels, channels, kernel_size=2, stride=2)
        if factor == 1:
            return nn.Identity()
        if factor == 0.5:
            return nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)
        raise ValueError(f"Unsupported resample factor {factor}")

    def forward(self, feats: list[torch.Tensor]) -> torch.Tensor:
        # Reassemble: project to common width, then bring to 4 scales.
        z = [self.resamples[i](self.projects[i](feats[i])) for i in range(len(feats))]
        # Fuse deepest -> shallowest (each block upsamples ×2).
        path = self.fusions[-1](z[-1])
        for i in range(len(z) - 2, -1, -1):
            path = self.fusions[i](path, z[i])
        return self.out_conv(path)


# ---------------------------------------------------------------------------
# Detail-capture matte heads (ViTMatte, Yao et al. 2024 + DeepLabv3 ASPP)
# ---------------------------------------------------------------------------
class ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling (DeepLabv3): parallel dilated convs +
    global image pooling, fused by a bottleneck conv."""

    def __init__(self, in_ch: int, channels: int, rates=(6, 12, 18)):
        super().__init__()
        self.branches = nn.ModuleList(
            [ConvModule(in_ch, channels, 1)]
            + [ConvModule(in_ch, channels, 3, dilation=r) for r in rates]
        )
        self.image_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), ConvModule(in_ch, channels, 1)
        )
        self.bottleneck = ConvModule((len(rates) + 2) * channels, channels, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outs = [branch(x) for branch in self.branches]
        outs.append(_up(self.image_pool(x), x.shape[2:]))
        return self.bottleneck(torch.cat(outs, dim=1))


class ConvStream(nn.Module):
    """ViTMatte detail-capture stream: strided 3x3 convs over the raw input
    image yield genuine high-frequency features at strides 2, 4, ... — detail
    the backbone (stride >= 4) never sees and bilinear upsampling can't invent.
    """

    def __init__(self, in_ch: int = 3, channels=(48, 96, 192)):
        super().__init__()
        self.convs = nn.ModuleList()
        ch = in_ch
        for out_ch in channels:
            self.convs.append(ConvModule(ch, out_ch, 3, stride=2))
            ch = out_ch

    def forward(self, image: torch.Tensor) -> list[torch.Tensor]:
        """Return detail maps at [1/2, 1/4, ...] of the image resolution."""
        details = []
        x = image
        for conv in self.convs:
            x = conv(x)
            details.append(x)
        return details


class FusionBlock(nn.Module):
    """ViTMatte fusion: upsample the coarse path to the detail map's size,
    concatenate, refine with a 3x3 conv."""

    def __init__(self, in_ch: int, detail_ch: int, out_ch: int):
        super().__init__()
        self.conv = ConvModule(in_ch + detail_ch, out_ch, 3)

    def forward(self, x: torch.Tensor, detail: torch.Tensor) -> torch.Tensor:
        x = _up(x, detail.shape[2:])
        return self.conv(torch.cat([x, detail], dim=1))


DETAIL_CHANNELS = (48, 96, 192)  # ConvStream widths at strides 2/4/8 (ViTMatte)


class ViTMatteHead(nn.Module):
    """Detail-capture head for non-hierarchical ViT backbones.

    Merges the 4 token grids (all at the patch stride) into one context map,
    enriches it with ASPP, then walks back to full resolution fusing ConvStream
    details at strides 8/4/2 and finally the raw image at stride 1. Output is
    ``channels // 8`` wide at the full input resolution, so the shared matte
    conv predicts alpha per-pixel instead of upsampling a coarse logit.
    """

    needs_image = True

    def __init__(self, in_channels: int, channels: int, n_inputs: int = 4):
        super().__init__()
        self.out_channels = channels // 8
        self.merge = ConvModule(n_inputs * in_channels, channels, 1)
        self.aspp = ASPP(channels, channels)
        self.convstream = ConvStream(3, DETAIL_CHANNELS)
        d1, d2, d3 = DETAIL_CHANNELS
        self.fusions = nn.ModuleList([
            FusionBlock(channels, d3, channels),            # -> 1/8
            FusionBlock(channels, d2, channels // 2),       # -> 1/4
            FusionBlock(channels // 2, d1, channels // 4),  # -> 1/2
            FusionBlock(channels // 4, 3, channels // 8),   # -> 1/1 (raw image)
        ])

    def forward(self, feats: list[torch.Tensor], image: torch.Tensor) -> torch.Tensor:
        x = self.aspp(self.merge(torch.cat(feats, dim=1)))
        details = self.convstream(image)  # [1/2, 1/4, 1/8]
        for fusion, detail in zip(self.fusions, [*reversed(details), image]):
            x = fusion(x, detail)
        return x


class UPerMatteHead(nn.Module):
    """Detail-capture head for hierarchical backbones.

    :class:`UPerHead` (PPM context + FPN fusion) provides the coarse stride-4
    semantic path; ConvStream details at strides 4/2 and the raw image restore
    the boundary sharpness the stride-4 map lacks. Output is ``channels // 8``
    wide at the full input resolution.
    """

    needs_image = True

    def __init__(self, in_channels: list[int], channels: int, pool_scales=(1, 2, 3, 6)):
        super().__init__()
        self.out_channels = channels // 8
        self.context = UPerHead(in_channels, channels, pool_scales)
        d1, d2 = DETAIL_CHANNELS[:2]
        self.convstream = ConvStream(3, (d1, d2))
        self.fusions = nn.ModuleList([
            FusionBlock(channels, d2, channels // 2),       # 1/4 (same scale)
            FusionBlock(channels // 2, d1, channels // 4),  # -> 1/2
            FusionBlock(channels // 4, 3, channels // 8),   # -> 1/1 (raw image)
        ])

    def forward(self, feats: list[torch.Tensor], image: torch.Tensor) -> torch.Tensor:
        x = self.context(feats)          # stride-4 fused semantics
        details = self.convstream(image)  # [1/2, 1/4]
        for fusion, detail in zip(self.fusions, [*reversed(details), image]):
            x = fusion(x, detail)
        return x
