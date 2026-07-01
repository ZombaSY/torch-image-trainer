"""Dense-prediction decoder heads: UPerHead and DPTHead.

Both consume features from a backbone and emit a ``(B, decoder_channels, H, W)``
feature map (upsampled toward the input). The final 1-channel matte conv +
sigmoid lives in :class:`~src.model.SegModel`, so both heads share it.

* **UPerHead** — for *hierarchical* backbones (Swin-L, InternImage-L). Classic
  UPerNet: a Pyramid Pooling Module on the deepest stage + an FPN top-down
  fusion over all stages. Consumes the 4 native feature-pyramid maps.
* **DPTHead** — for *non-hierarchical* ViT backbones (DINOv2-L, EVA-02-L). Takes
  tokens reshaped to 2D grids from 4 transformer depths, "reassembles" them to
  4 resolutions (×4, ×2, ×1, ×0.5 of the token grid), then fuses them
  coarse-to-fine with RefineNet-style residual blocks (Ranftl et al., 2021).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _up(x: torch.Tensor, size, mode: str = "bilinear") -> torch.Tensor:
    return F.interpolate(x, size=size, mode=mode, align_corners=False)


class ConvModule(nn.Sequential):
    """Conv -> BN -> ReLU, the standard mmseg building block."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, padding: int | None = None):
        if padding is None:
            padding = kernel_size // 2
        super().__init__(
            nn.Conv2d(in_ch, out_ch, kernel_size, padding=padding, bias=False),
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
