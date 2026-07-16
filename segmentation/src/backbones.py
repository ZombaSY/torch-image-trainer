"""Backbone adapters + registry.

Four heterogeneous backbones are unified behind one interface so the decoder
heads don't care where features come from (blueprint contract #3: the backbone
still dictates preprocessing). Each adapter is an ``nn.Module`` whose
``.parameters()`` are the backbone weights, and whose ``forward`` returns a list
of feature maps for the head:

* **hierarchical** (Swin-L-v2, InternImage-L) -> 4 native pyramid maps
  ``(B, C_i, H_i, W_i)`` at strides 4/8/16/32, for UPerHead.
* **vit** (DINOv2-L, EVA-02-L) -> 4 token grids ``(B, C, Hp, Wp)`` from
  increasing depth, for DPTHead.

``build_backbone`` also returns a ``data_config`` (input size, mean, std) so the
dataloader matches the backbone's expected normalization.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .config import Config

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _even_block_indices(depth: int, n: int = 4) -> list[int]:
    """``n`` roughly evenly spaced block outputs, ending at the last block."""
    return [max(0, round(depth * (i + 1) / n) - 1) for i in range(n)]


def _to_nchw(feat: torch.Tensor, channels: int) -> torch.Tensor:
    """Normalize a 4D feature map to NCHW (timm hierarchical nets vary)."""
    if feat.shape[1] == channels:
        return feat
    if feat.shape[-1] == channels:
        return feat.permute(0, 3, 1, 2).contiguous()
    return feat


# ---------------------------------------------------------------------------
# Adapters
# ---------------------------------------------------------------------------
class TimmViTAdapter(nn.Module):
    """timm ViT (EVA-02) -> 4 token grids via ``forward_intermediates``."""

    is_hierarchical = False
    # No weight decay on embeddings/rotary and LayerScale gammas.
    no_wd = ("pos_embed", "cls_token", "rope", "gamma")

    def __init__(self, model: nn.Module, indices: list[int], embed_dim: int, depth: int):
        super().__init__()
        self.model = model
        self.indices = indices
        self.embed_dim = embed_dim
        self.feature_channels = [embed_dim] * len(indices)
        self.num_layers = depth + 1  # embeddings=0, blocks 1..depth, final norm=depth+1

    def layer_id(self, name: str) -> int:
        if "blocks." in name:
            return int(name.split("blocks.")[1].split(".")[0]) + 1
        if any(k in name for k in ("patch_embed", "cls_token", "pos_embed", "rope")):
            return 0
        return self.num_layers

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        return self.model.forward_intermediates(
            x, indices=self.indices, norm=True,
            output_fmt="NCHW", intermediates_only=True,
        )


class TimmHierAdapter(nn.Module):
    """timm hierarchical net (Swin-L-v2) via ``features_only``."""

    is_hierarchical = True
    # Swin's recipe uses no layer-wise decay (layer_decay=1) but excludes these
    # from weight decay.
    no_wd = ("relative_position_bias_table", "absolute_pos_embed")
    num_layers = 1  # LLRD unused for Swin; layer_id is a no-op

    def __init__(self, model: nn.Module, channels: list[int]):
        super().__init__()
        self.model = model
        self.feature_channels = channels

    def layer_id(self, name: str) -> int:
        return 0

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        feats = self.model(x)
        return [_to_nchw(f, c) for f, c in zip(feats, self.feature_channels)]


class Dinov2Adapter(nn.Module):
    """HF DINOv2 ViT -> 4 token grids from selected hidden states.

    Stands in for DINOv3 (no access). Non-hierarchical -> DPTHead.
    """

    is_hierarchical = False
    no_wd = ("cls_token", "position_embeddings", "mask_token", "register_tokens")

    def __init__(self, model: nn.Module, layer_ids: list[int], patch: int, embed_dim: int, depth: int):
        super().__init__()
        self.model = model
        self.layer_ids = layer_ids
        self.patch = patch
        self.embed_dim = embed_dim
        self.feature_channels = [embed_dim] * len(layer_ids)
        self.num_layers = depth + 1  # embeddings=0, encoder.layer.i -> i+1

    def layer_id(self, name: str) -> int:
        if "encoder.layer." in name:
            return int(name.split("encoder.layer.")[1].split(".")[0]) + 1
        if "embeddings" in name:
            return 0
        return self.num_layers

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        hp, wp = x.shape[2] // self.patch, x.shape[3] // self.patch
        out = self.model(pixel_values=x, output_hidden_states=True)
        grids = []
        for li in self.layer_ids:
            tokens = out.hidden_states[li][:, 1:, :]  # drop CLS
            b, n, c = tokens.shape
            grids.append(tokens.transpose(1, 2).reshape(b, c, hp, wp).contiguous())
        return grids


class InternImageAdapter(nn.Module):
    """OpenGVLab InternImage -> 4 pyramid maps. Hierarchical -> UPerHead.

    InternImage needs OpenGVLab's custom DCNv3 CUDA op, which is not part of
    timm/transformers. This adapter loads the published weights
    (``OpenGVLab/internimage_l_22k_384``) via ``trust_remote_code`` and raises a
    clear, actionable error if the op/package is missing rather than failing
    obscurely deep in the loop.
    """

    is_hierarchical = True
    no_wd = ()  # norms/biases handled by the 1D-param rule

    def __init__(self, model: nn.Module, channels: list[int], depths: list[int]):
        super().__init__()
        self.model = model
        self.feature_channels = channels
        self.depths = depths
        # Cumulative block offset per stage; matches InternImage's layer-wise
        # decay constructor (num_layers = total blocks + 1).
        self.offsets = [sum(depths[:i]) for i in range(len(depths))]
        self.num_layers = sum(depths) + 1

    def layer_id(self, name: str) -> int:
        if "patch_embed" in name:
            return 0
        if "levels." in name and ".blocks." in name:
            lvl = int(name.split("levels.")[1].split(".")[0])
            blk = int(name.split("blocks.")[1].split(".")[0])
            return self.offsets[lvl] + blk + 1
        if "levels." in name and "downsample" in name:
            lvl = int(name.split("levels.")[1].split(".")[0])
            return self.offsets[lvl] + self.depths[lvl]
        return self.num_layers

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        # The HF InternImage returns 4 stage maps (already NHWC->NCHW inside
        # forward_features) under 'hidden_states'.
        out = self.model(x)
        feats = out["hidden_states"] if isinstance(out, dict) else out.hidden_states
        return [_to_nchw(f, c) for f, c in zip(feats, self.feature_channels)]


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------
def _timm_data_config(model_id: str, image_size: int) -> dict:
    import timm

    pc = timm.get_pretrained_cfg(model_id)
    return {"input_size": image_size, "mean": tuple(pc.mean), "std": tuple(pc.std)}


def _build_timm_vit(cfg: Config, meta: dict):
    import timm

    model = timm.create_model(
        meta["model_id"], pretrained=cfg.model.pretrained,
        num_classes=0, img_size=cfg.image_size,
        drop_path_rate=cfg.model.drop_path_rate,
    )
    depth = len(model.blocks)
    embed_dim = model.embed_dim
    adapter = TimmViTAdapter(model, _even_block_indices(depth), embed_dim, depth)
    return adapter, _timm_data_config(meta["model_id"], cfg.image_size)


def _build_timm_hier(cfg: Config, meta: dict):
    import timm

    model = timm.create_model(
        meta["model_id"], pretrained=cfg.model.pretrained,
        features_only=True, out_indices=(0, 1, 2, 3), img_size=cfg.image_size,
        drop_path_rate=cfg.model.drop_path_rate,
    )
    channels = list(model.feature_info.channels())
    adapter = TimmHierAdapter(model, channels)
    return adapter, _timm_data_config(meta["model_id"], cfg.image_size)


def _build_dinov2(cfg: Config, meta: dict):
    from transformers import AutoModel

    # drop_path_rate is forwarded into the Dinov2 config (stochastic depth for
    # full fine-tuning; 0.0 is a no-op for the frozen/linear setup).
    model = AutoModel.from_pretrained(
        meta["model_id"], drop_path_rate=cfg.model.drop_path_rate
    )
    depth = model.config.num_hidden_layers
    embed_dim = model.config.hidden_size
    patch = model.config.patch_size
    # hidden_states has depth+1 entries ([0]=embeddings); take 4 spaced blocks.
    layer_ids = [round(depth * (i + 1) / 4) for i in range(4)]
    adapter = Dinov2Adapter(model, layer_ids, patch, embed_dim, depth)
    data_config = {"input_size": cfg.image_size, "mean": IMAGENET_MEAN, "std": IMAGENET_STD}
    return adapter, data_config


def _ensure_pkg_resources():
    """setuptools>=81 no longer ships ``pkg_resources``, but the InternImage
    remote code still imports it at module level (it only uses it to
    version-check the optional DCNv3 CUDA op). Register a minimal stand-in so
    transformers' import check and the remote module both resolve it.
    """
    try:
        import pkg_resources  # noqa: F401
        return
    except ImportError:
        pass
    import sys
    import types
    from importlib import metadata

    stub = types.ModuleType("pkg_resources")
    stub.get_distribution = lambda name: types.SimpleNamespace(
        version=metadata.version(name)
    )
    stub.DistributionNotFound = metadata.PackageNotFoundError
    sys.modules["pkg_resources"] = stub


def _build_internimage(cfg: Config, meta: dict):
    try:
        from transformers import AutoConfig
        from transformers.dynamic_module_utils import get_class_from_dynamic_module

        _ensure_pkg_resources()

        # The HF repo ships a pure-PyTorch DCNv3 fallback, so no custom CUDA op
        # build is required; trust_remote_code loads the model definition.
        conf = AutoConfig.from_pretrained(meta["model_id"], trust_remote_code=True)
        model_cls = get_class_from_dynamic_module(
            conf.auto_map["AutoModel"], meta["model_id"]
        )
        # The remote code predates transformers 5 and never calls post_init(),
        # which from_pretrained now requires (all_tied_weights_keys etc.).
        if not getattr(model_cls, "_post_init_patched", False):
            orig_init = model_cls.__init__

            def _init(self, config, *args, **kwargs):
                orig_init(self, config, *args, **kwargs)
                self.post_init()

            model_cls.__init__ = _init
            model_cls._post_init_patched = True
        model = model_cls.from_pretrained(meta["model_id"])
    except Exception as exc:  # missing remote code, weights, or (rare) op path
        raise RuntimeError(
            f"Could not load InternImage backbone {meta['model_id']!r}: {exc}\n"
            "It loads via transformers trust_remote_code (with a PyTorch DCNv3 "
            "fallback). See https://huggingface.co/OpenGVLab/internimage_l_22k_384 . "
            "The other three backbones (eva02-l, dinov2-l, swinv2-l) run without it."
        ) from exc
    # InternImage doubles channels each stage: channels -> [c, 2c, 4c, 8c].
    conf = getattr(model, "config", None)
    c0 = getattr(conf, "channels", 160)
    channels = [c0 * (2 ** i) for i in range(4)]
    depths = list(getattr(conf, "depths", [5, 5, 22, 5]))
    adapter = InternImageAdapter(model, channels, depths)
    data_config = {"input_size": cfg.image_size, "mean": IMAGENET_MEAN, "std": IMAGENET_STD}
    return adapter, data_config


def build_backbone(cfg: Config):
    """Construct the backbone adapter + its data_config for the configured model."""
    meta = cfg.backbone_meta
    source = meta["source"]
    if source == "timm":
        if meta["family"] == "vit":
            return _build_timm_vit(cfg, meta)
        return _build_timm_hier(cfg, meta)
    if source == "dinov2":
        return _build_dinov2(cfg, meta)
    if source == "internimage":
        return _build_internimage(cfg, meta)
    raise ValueError(f"Unknown backbone source: {source!r}")
