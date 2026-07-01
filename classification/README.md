# UI Style Classifier — WD v3 Tagger Trainer

PyTorch training pipeline that fine-tunes a [WD v3 tagger](https://huggingface.co/SmilingWolf)
backbone into a 4-class UI style classifier (`chic`, `feminine`, `lovely`, `other`).

Images are read with **OpenCV (cv2)** and augmented with **albumentations**.
Trainer, dataloader, and configuration are separated for reproducibility.

## Supported backbones

| key | HuggingFace Hub id |
|-----|--------------------|
| `wd-swinv2-tagger-v3` | `SmilingWolf/wd-swinv2-tagger-v3` |
| `wd-vit-large-tagger-v3` | `SmilingWolf/wd-vit-large-tagger-v3` |
| `wd-vit-tagger-v3` | `SmilingWolf/wd-vit-tagger-v3` |

Each backbone is loaded through `timm`, turned into a pooled feature extractor,
and topped with a simple `Dropout → Linear` classifier head.

## Training modes

- **`linear_probe`** — backbone frozen, only the classifier head trains.
- **`full_finetune`** — backbone + head train, with a smaller LR on the backbone.

## Layout

```
configs/
  linear_probe.yaml      # frozen backbone
  full_finetune.yaml     # end-to-end fine-tuning
src/
  config.py              # typed config + YAML loader + CLI overrides
  dataset.py             # cv2 loading, albumentations, dataloaders
  model.py               # backbone + classifier head, freeze logic
  trainer.py             # train/val loop, AMP, scheduler, checkpoints
  utils.py               # seeding, logging, metrics
train.py                 # training entrypoint
test.py                  # inference entrypoint
```

## Setup

```bash
pip install -r requirements.txt
```

The dataset is read from `data.root` in the config:
`/data/seo_sunyong/workspace/data/cocone/UI/P1/style-classifier-dataset`
with `train.csv` / `val.csv` (columns: `input`, `class`).

## Usage

```bash
# Linear probing (default ViT tagger)
python train.py --config configs/linear_probe.yaml

# Full fine-tuning on the SwinV2 tagger
python train.py --config configs/full_finetune.yaml model.backbone=wd-swinv2-tagger-v3

# Override any leaf field as section.field=value
python train.py --config configs/linear_probe.yaml \
    model.backbone=wd-vit-large-tagger-v3 optim.lr=5e-4 optim.epochs=40
```

Pick the GPU with `CUDA_VISIBLE_DEVICES=0` before the command.

## Early stopping

Training stops once the monitored metric (`run.monitor`, macro-F1 by default)
has not set a new best for `run.early_stop_patience` consecutive epochs (50 by
default; set to `0` to disable). `optim.epochs` (500) is just a hard upper cap.

## Inference / testing

Load a checkpoint and score a CSV of images:

```bash
# Score the val split (from the checkpoint's own config) + annotate images
python test.py --checkpoint runs/linear_probe_vit/best.pt --save-images

# Score an arbitrary CSV
python test.py --checkpoint runs/full_finetune_vit/best.pt \
    --csv /path/to/test.csv --output-dir runs/full_finetune_vit/test
```

Flags:

| flag | default | meaning |
|------|---------|---------|
| `--checkpoint` | _(required)_ | Path to a `.pt` checkpoint (embeds its own config). |
| `--csv` | config's `val.csv` | CSV of images to score. |
| `--data-root` | config's `data.root` | Root the CSV paths are relative to. |
| `--output-dir` | `<checkpoint_dir>/test` | Where outputs are written. |
| `--batch-size` | `16` | Inference batch size. |
| `--save-images` | off | Also save each image annotated with the prediction. |
| `--device` | `cuda` | `cuda` or `cpu` (falls back to `cpu` if no GPU). |

Writes to `--output-dir` (defaults to `<checkpoint_dir>/test`):

- `predictions.csv` — per image: predicted class/name, confidence, per-class
  probabilities, and (when the CSV has a `class` column) true label + `correct`.
- `images/` — each image annotated with the predicted class text (with
  `--save-images`). The flattened filename mirrors the source path.

The checkpoint embeds its full config, so `test.py` rebuilds the exact model and
preprocessing automatically — no need to pass the original YAML.

## Augmentations (train split only)

All toggleable in the `aug:` config block.

Per-sample (albumentations):

- Horizontal flip (`A.HorizontalFlip`)
- Coarse dropout (`A.CoarseDropout`)
- Blur (`A.Blur`)
- Rotate (`A.Rotate`)

Batch-level (`src/mixaug.py`, applied in the training loop):

- **MixUp** — blends two images by a `Beta(mixup_alpha, mixup_alpha)` ratio.
- **CutMix** — pastes a random patch from one image onto another.

The ground-truth label is split by the same ratio used to mix the pixels:
MixUp uses its blend ratio, CutMix uses the **actual pasted-patch area**
(recomputed after edge clipping). The split is folded into the loss as
`lam·loss(·, y) + (1−lam)·loss(·, y_perm)`, preserving class weights and label
smoothing. `mix_p` sets the per-batch chance of mixing; with both enabled,
`mix_switch_prob` picks CutMix over MixUp. The ratio is clamped to
`[mix_min_ratio, 1 − mix_min_ratio]` (default `[0.3, 0.7]`) so neither source
ever dominates; CutMix places its patch fully in-frame so this floor survives
near the edges.

Both splits share the resize → normalize tail. Input is **448×448**, normalized
with mean/std `0.5` (the WD tagger preprocessing); images are padded to a square
on a white background before resizing.

## Outputs

Each run writes to a timestamped subdirectory `run.output_dir/<yymmdd-hhmmss>`
(e.g. `runs/linear_probe_vit/250623-121738/`):

- `config.yaml` — exact resolved config (reproducibility snapshot)
- `best_e<epoch>_<metric><score>.pt` / `last_e<epoch>_<metric><score>.pt` —
  checkpoints (`model_state`, `config`, `metrics`) with the score in the
  filename, e.g. `best_e7_macro_f10.6445.pt`. Only the current best is kept.
- `history.json` — per-epoch metrics
- `train.log` — full training log

The best checkpoint is selected by `run.monitor` (`macro_f1` by default), which
matters here because the `other` class is under-represented — class-weighted
loss is enabled by default to compensate.
