# Alpha-Matte Segmentation Trainer

Fine-tunes a pretrained vision backbone into a **dense alpha-matte regressor**:
given an RGB composite of a logo, predict the logo's alpha channel. It follows
the same design as the sibling [`../classification`](../classification) trainer
(typed config, config-embedded checkpoints, backbone-driven preprocessing,
timestamped runs) — see the blueprint skill at
`.claude/skills/pytorch-image-trainer-blueprint/`.

## Why regression, not class segmentation

The dataset's masks are the logos' **alpha channels** (a continuous 0–255
matte), so the target has no class axis. The model outputs a single-channel
alpha in `[0, 1]` (sigmoid) and is trained with a **pixel distance** (L1 by
default), exactly like an image-to-image generation task — not a per-class
cross-entropy.

## Backbones × heads

| Backbone (`model.backbone`) | Family | Head | Source |
|-----------------------------|--------|------|--------|
| `eva02-l` (EVA-02-L) | non-hierarchical ViT | **DPT** | timm |
| `dinov2-l` (DINOv2-L, stands in for DINOv3) | non-hierarchical ViT | **DPT** | transformers |
| `swinv2-l` (Swin-L-v2) | hierarchical | **UPerHead** | timm |
| `internimage-l` (InternImage-L) | hierarchical | **UPerHead** | OpenGVLab (HF `trust_remote_code`) |

The head is auto-selected from the backbone family (DPT for ViTs, UPerHead for
hierarchical pyramids); override with `model.head` if needed. **All four run out
of the box** (verified on GPU 0): `eva02-l`/`swinv2-l` via timm, `dinov2-l` via
`transformers`, and `internimage-l` via `transformers` `trust_remote_code` —
its remote code ships a pure-PyTorch DCNv3 fallback, so no custom CUDA op build
is needed (compiling OpenGVLab's op is optional, just faster).

> DINOv3 was requested but isn't accessible here, so `dinov2-l`
> (`facebook/dinov2-large`) stands in — same ViT-L + DPT wiring; swap the
> `model_id` in `src/config.py` when DINOv3 becomes available.

## Layout

```
configs/         eva02_dpt.yaml  dinov2_dpt.yaml  swin_uper.yaml  internimage_uper.yaml
src/
  config.py      typed config + backbone registry + YAML/CLI overrides
  dataset.py     RGBA read/composite, random-bg aug, image+mask transforms
  backbones.py   4 backbone adapters -> uniform multi-scale features
  heads.py       DPTHead (ViT) + UPerHead (hierarchical)
  model.py       backbone + head + shared 1-ch matte conv (sigmoid)
  losses.py      pixel distance (l1 | mse | l1_mse)
  trainer.py     train/val loop, AMP, cosine warmup, MAE-based checkpointing
  utils.py       seeding, logging, matting metrics (MAE/MSE)
train.py         training entrypoint
test.py          inference: predict mattes + MAE/MSE
```

## Usage

Pick GPU 0 (others may be busy) and run from this directory:

```bash
CUDA_VISIBLE_DEVICES=0 python train.py --config configs/eva02_dpt.yaml

# Override any leaf field as section.field=value (e.g. shorter warmup for the
# tiny dataset, smaller batch to fit one GPU)
CUDA_VISIBLE_DEVICES=0 python train.py --config configs/swin_uper.yaml \
    optim.warmup_iters=200 optim.batch_size=4

# Score the val split from a checkpoint (embeds its own config)
CUDA_VISIBLE_DEVICES=0 python test.py --checkpoint runs/eva02_dpt/<ts>/best.pt
```

## Learning-rate recipes (paper-faithful)

Each config mirrors that backbone's **official semantic-segmentation** recipe
(UPerNet/DPT on ADE20K), not a generic default — LR, weight decay, layer-wise LR
decay (LLRD), stochastic depth, grad-clip, and freeze mode all follow the source:

| Backbone | LR | Weight decay | Layer-wise LR decay | drop_path | Mode | Source |
|---|---|---|---|---|---|---|
| `swinv2-l` | 6e-5 (single) | 0.01 | none (no-decay on norm/pos-embed/rel-pos-bias) | 0.3 | full_finetune | Swin / mmseg UPerNet |
| `eva02-l` | 4e-5 | 0.05 | **0.9** (24 blocks) | 0.2 | full_finetune | baaivision/EVA UPerNet |
| `internimage-l` | 2e-5 | 0.05 | **0.94** (37 layers) + grad-clip 0.1 | (HF default) | full_finetune | OpenGVLab UPerNet |
| `dinov2-l` | 4e-5 | 0.05 | **0.9** | 0.2 | full_finetune | *adapted* from EVA-02-L — DINOv2 has no official seg fine-tune recipe (its released one freezes the backbone) |

All use AdamW, **poly** decay (`power=1.0`), and a **1500-iter linear warmup**
(`warmup_ratio=1e-6`). LLRD is implemented in `src/optim.py`: backbone layer `i`
gets `lr · layer_decay**(num_layers − layer_id)`; norms/biases/embeddings get
`weight_decay=0`. DINOv2 is the exception: it has no official end-to-end seg
recipe (the released one freezes the backbone), so its config full-finetunes
with a recipe adapted from the EVA-02-L ViT-L analog. To reproduce DINOv2's
official frozen setup instead, run with `run.mode=decoder_only optim.lr=1e-3
optim.weight_decay=1e-4 optim.layer_decay=1.0`.

> **Caveats for this dataset.** These are ADE20K recipes (20k images, batch 16,
> 40k–160k iters). Your set is ~90 images, so: (1) `warmup_iters=1500` is a large
> fraction of a short run — consider lowering it; (2) the poly schedule spans the
> *actual* run length (`epochs × iters/epoch`), so the LR *shape* is faithful even
> though absolute iteration counts differ; (3) batch 8 on one GPU is smaller than
> the papers' effective 16 — lower `optim.batch_size` if you hit OOM on
> full-finetune. The LR *values* follow the papers; treat them as a strong
> starting point, not a guarantee for a tiny regression task.

## Augmentation

Strong on purpose — the dataset is tiny (~90 train / 22 val). Matches the
classification per-sample ops (horizontal flip, rotate, blur, coarse dropout)
**minus** MixUp/CutMix, **plus**:

- **Vertical flip** and **color jitter** (photometric, image only).
- **Random-background compositing** — composite the logo onto a random solid
  color (else white) so the model learns alpha independent of background. The
  target matte is unchanged; this is the key matting robustness aug.
- **CutMix** — paste a rectangular patch from another sample onto both the
  image and its matte. Because the target is dense, the matte is cut identically
  (no label-mixing ratio, unlike classification CutMix). It runs **first**, at
  the raw-sample level **before every other augmentation**, so the mixed
  image+mask is then flipped/rotated/jittered as one coherent sample.

Geometric ops apply to image **and** mask; photometric ops to the image only.

### Unpremultiply

`data.unpremultiply_alpha` recovers straight color (`rgb / alpha`) before
compositing, for *premultiplied* source assets. These logos look mostly
**straight-alpha** in spot checks, where unpremultiply over-brightens edges — set
it to `false` if predicted/observed mattes look wrong around edges.

## Logging (Weights & Biases)

Off by default. Enable per config or on the CLI:

```bash
wandb login   # once
CUDA_VISIBLE_DEVICES=0 python train.py --config configs/eva02_dpt.yaml \
    wandb.enabled=true wandb.project=my-matting wandb.mode=online
```

It logs the **run configuration** (once) and **per-epoch training states**
(train/val loss, MAE/MSE, LR). It deliberately **excludes large data** — no model
weights and no images are ever uploaded (no `wandb.save` of checkpoints, no
image logging). Configure under the `wandb:` block (`enabled`, `project`,
`entity`, `run_name`, `mode` = online/offline/disabled, `tags`). Requires
`pip install wandb`; if `enabled=true` without it installed, the run errors
clearly. W&B's local files are written inside the run directory (gitignored).

## Outputs

Each run writes `runs/<name>/<yymmdd-hhmmss>/`: `config.yaml` (snapshot),
`best_e<epoch>_mae<score>.pt` / `last_...pt` (checkpoints embed model+config+metrics),
`history.json`, `train.log`. Best checkpoint = lowest val MAE (`run.monitor`).
