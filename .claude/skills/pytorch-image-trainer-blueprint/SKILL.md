---
name: pytorch-image-trainer-blueprint
description: >-
  Architecture blueprint for building or extending a PyTorch image-classification
  training pipeline (backbone + classifier head, config-driven, reproducible).
  Use this whenever you are scaffolding a new image trainer, adding a training
  mode / backbone / augmentation / metric, wiring up an inference or
  hyper-parameter-sweep script, or touching config, dataset, model, trainer, or
  checkpoint code in a project shaped like this. Apply it even when the request
  is phrased as "just add X" — the point is that X should slot into the existing
  structure (typed config, config-embedded checkpoints, backbone-driven
  preprocessing) instead of growing a second architecture beside it.
---

# PyTorch Image Trainer Blueprint

This skill captures a proven layout for fine-tuning a pretrained vision backbone
into a small image classifier, and running / sweeping / serving it
reproducibly. The reference implementation fine-tunes WD v3 taggers (via `timm`)
into a UI-style classifier, but the structure is backbone- and task-agnostic.

The goal is not to copy files verbatim — it is to preserve the **contracts**
that make the pipeline reproducible and easy to extend. When you add something,
make it flow through these contracts rather than around them.

## Module layout

Keep thin entrypoints and a `src/` package of focused modules. Each module owns
one concern so a change (new backbone, new metric, new aug) touches one place.

```
configs/          # one YAML per training regime (linear_probe.yaml, full_finetune.yaml)
src/
  config.py       # typed dataclass config + YAML loader + CLI overrides + validate()
  dataset.py      # image read/preprocess, transforms, Dataset, dataloaders, class weights
  model.py        # build backbone + head, freeze logic, per-group params, param counts
  trainer.py      # train/val loop, AMP, scheduler, metric-based checkpointing, early stop
  mixaug.py       # batch-level augmentation (MixUp/CutMix) applied inside the loop
  utils.py        # seeding, logging, metrics — small shared helpers, no project logic
train.py          # training entrypoint (parse args -> build -> fit)
test.py           # inference entrypoint (load ckpt -> predict CSV -> write outputs)
sweep.py          # hyper-parameter sweep that shells out to train.py per trial
```

Entrypoints stay declarative: parse args, build the pieces, call `.fit()` or
`predict()`. All real logic lives in `src/`. Prefer many small files (200–400
lines) over few large ones.

## The five contracts

These are the load-bearing decisions. Everything else is detail.

### 1. One typed Config object fully describes a run

A run must be reproducible from its config alone. Model this as a top-level
`Config` composed of section dataclasses (`data`, `aug`, `model`, `optim`,
`run`), each with defaults and docstrings explaining *why* each knob exists.

- `load_config(path)` reads YAML → validates → returns `Config`.
- `Config.validate()` fails fast on impossible combos (unknown backbone, mode,
  `len(class_names) != num_classes`, unsupported monitor metric). Validation
  lives on the config, not scattered through the trainer.
- `_build_section` rejects **unknown YAML keys** so a typo becomes a loud error
  instead of a silently-ignored setting.
- CLI overrides are dotted `section.field=value` and type-coerced to match the
  existing field's type (`_coerce`). This lets `python train.py --config c.yaml
  optim.lr=5e-4 model.backbone=...` work without a bespoke flag per field.
- `to_dict` / `from_dict` / `save` round-trip the config through plain dicts —
  this is what makes contract #2 possible.

See `references/config-module.md` for the full pattern. When you add a knob, add
a typed field with a default and a comment on its purpose — never read a loose
key from a dict deep in the trainer.

### 2. Checkpoints embed their own config

A `.pt` checkpoint stores `{"epoch", "model_state", "config", "metrics"}`. The
embedded `config` is the config dict from contract #1. This means inference and
evaluation **rebuild the exact model and preprocessing from the checkpoint** —
no need to remember or pass the original YAML.

```python
ckpt = torch.load(path, map_location=device, weights_only=True)  # tensors + primitives only
raw_cfg = dict(ckpt["config"])
raw_cfg["model"] = {**raw_cfg["model"], "pretrained": False}  # weights come from the ckpt, not the hub
cfg = from_dict(raw_cfg)
model, data_config = build_model(cfg)
model.load_state_dict(ckpt["model_state"])
```

Use `weights_only=True` — these checkpoints hold only tensors and primitive
config, so the safe loader is sufficient and avoids arbitrary-code execution on
unpickle. Flip `pretrained=False` on reload so you don't re-download backbone
weights you're about to overwrite.

### 3. The backbone dictates preprocessing

Build the model **first**, then let it tell the dataloader how to preprocess.
`timm.data.resolve_model_data_config(backbone)` yields the input size, mean, and
std the backbone was trained with; the dataloader's resize/normalize tail must
match exactly. Order in the entrypoint:

```python
model, data_config = build_model(cfg)
image_size = cfg.model.image_size or data_config["input_size"][1]
mean, std  = tuple(data_config["mean"]), tuple(data_config["std"])
train_loader, val_loader, train_ds = build_dataloaders(cfg, mean, std, image_size)
```

Never hardcode 224/0.5/ImageNet-stats in the dataset. Getting train/inference
preprocessing to diverge is the most common silent accuracy bug; deriving it
from the backbone in one place prevents it.

### 4. One shared image-read function for train AND inference

Preprocessing that differs between training and inference silently wrecks
accuracy. Centralize the raw read + geometric normalization in a single
`load_image(path, to_rgb, pad_square, pad_value)` used by the `Dataset`, the
inference loop, and the image-annotation path alike. Read with
`cv2.IMREAD_UNCHANGED` and alpha-composite onto the pad background rather than
dropping the alpha channel (dropping it leaks arbitrary RGB from under
transparent pixels). Split the pipeline as: shared read/pad (both splits) →
train-only augmentations → shared resize/normalize/ToTensor tail (both splits).

### 5. Every run is a timestamped, self-documenting directory

`train.py` writes into `<output_dir>/<yymmdd-hhmmss>/` containing:

- `config.yaml` — the exact resolved config (reproducibility snapshot)
- `best_e<epoch>_<metric><score>.pt` / `last_...pt` — score in the filename so
  runs are comparable at a glance; only the current best is kept on disk
- `history.json` — per-epoch metrics
- `train.log` — full log (console + file via `get_logger`)

Seed everything (`set_seed`: python/numpy/torch/cuda + cudnn deterministic)
before building anything. The config snapshot plus the seed is what lets a run
be reproduced or resumed later.

## The training loop (Trainer)

`Trainer` owns the train/val loops and all artifact writing. Keep these
behaviors — each exists for a reason:

- **AMP** via `torch.amp.autocast` + `GradScaler`, enabled only on CUDA.
- **Param groups**: the head always trains at `optim.lr`; the backbone trains
  only in `full_finetune`, at a smaller `optim.backbone_lr`. `linear_probe`
  freezes the backbone (`requires_grad=False` *and* keeps it in `.eval()` so
  norm-layer running stats stop updating).
- **Schedule**: linear warmup then cosine decay to `min_lr`, implemented as a
  `LambdaLR` fraction-of-base function so `scheduler == "none"` is a clean
  no-op.
- **Class-weighted loss** (inverse-frequency, normalized to mean 1) to counter
  class imbalance; label smoothing configurable on the same criterion.
- **Metric-based checkpointing**: select best by `run.monitor` (e.g.
  `macro_f1`), delete the previous best before saving the new one (checkpoints
  are hundreds of MB).
- **Early stopping**: stop after `early_stop_patience` epochs with no new best;
  `epochs` is a hard upper cap, not the expected stopping point.

See `references/trainer-and-mixaug.md` for the loop skeleton and the batch-aug
contract.

## Batch augmentation splits the label at the loss, not the tensor

MixUp/CutMix operate on already-batched, on-device tensors, so they live next to
the loop (`mixaug.py`), not in the per-sample dataset. Return a small dataclass
carrying `(images, target_a, target_b, lam)` and blend at the loss level:

```python
lam * criterion(logits, target_a) + (1 - lam) * criterion(logits, target_b)
```

This keeps the class weights and label smoothing already configured on
`criterion` intact (a one-hot soft-label approach would discard them). For
CutMix, recompute `lam` from the **actual pasted-patch area** after rounding, so
the label weight matches the pixels that actually changed. With `lam == 1` (no
mix fired) it must reduce exactly to `criterion(logits, y)`.

## Sweeps shell out to the training entrypoint

`sweep.py` runs one `subprocess` per trial invoking `train.py` with dotted
overrides. One process per trial frees CUDA memory between runs and reuses the
exact training path (no duplicated loop). Score each trial from its
`history.json`. Because checkpoints are large, **archive each trial's result
(params, score, full history) to `sweep_results.json` before deleting** the
heavy run directory, and keep only the single best run on disk (`--keep-losers`
to override). Support both `grid` (product of value lists) and `random`
(log-uniform LR ranges, uniform categorical) search, and collapse axes that do
nothing in the current mode (e.g. `backbone_lr` under `linear_probe`).

## Dependency-light metrics

Compute accuracy + macro precision/recall/F1 from confusion counts in plain
numpy (`classification_metrics`). Keeping the metric path free of sklearn makes
it trivial to audit and guarantees train/eval use identical math. Return
per-class F1 too — a macro metric hides which class is failing, which matters
under imbalance.

## When extending — checklist

- [ ] New knob → typed field on the right section dataclass, with a default and
      a comment on its purpose. Not a loose dict read.
- [ ] New backbone → add to the supported map; confirm preprocessing flows from
      `resolve_model_data_config`, not hardcoded values.
- [ ] New augmentation → per-sample goes in `build_transforms` (train-only,
      behind an `aug` toggle); batch-level goes in `mixaug.py`.
- [ ] New metric → add to `classification_metrics` and allow it in
      `run.monitor` validation.
- [ ] Anything touching the model → does the checkpoint still round-trip
      (train → save → `from_dict` → `build_model` → `load_state_dict`)?
- [ ] Preprocessing change → made in the **shared** `load_image` / transform
      tail so train and inference stay identical.
- [ ] Run still self-documents (config snapshot + seed + history + log)?
