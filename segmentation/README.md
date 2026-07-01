# Semantic Segmentation Trainer — placeholder

> **Status: scaffold only.** Every module here is a stub that raises
> `NotImplementedError`. It mirrors the structure of the sibling
> [`../classification`](../classification) trainer so the two share the same
> design contracts. Fill the stubs in following the pattern described below.

This trainer fine-tunes a pretrained encoder into a dense (per-pixel) semantic
segmentation model. It deliberately reuses the architecture of the
classification trainer — see the repo's blueprint skill at
`.claude/skills/pytorch-image-trainer-blueprint/` for the full rationale.

## Layout (same shape as classification)

```
configs/         # one YAML per regime (unet.yaml, ...)
src/
  config.py      # typed dataclass config + YAML loader + CLI overrides + validate()
  dataset.py     # image+mask read/preprocess, transforms, Dataset, dataloaders
  model.py       # build encoder+decoder (e.g. U-Net/DeepLab), freeze logic, param groups
  trainer.py     # train/val loop, AMP, scheduler, IoU/Dice-based checkpointing, early stop
  utils.py       # seeding, logging, segmentation metrics (IoU/Dice)
train.py         # training entrypoint
test.py          # inference entrypoint (load ckpt -> predict masks -> write outputs)
```

## The five contracts (carried over from classification)

The load-bearing decisions are identical; only the task-specific pieces change.
Read `../classification/src/*.py` for the concrete reference implementation.

1. **One typed `Config` fully describes a run.** Same machinery
   (`load_config` / `validate` / `from_dict` / dotted CLI overrides / unknown-key
   rejection). The `_build_section` / `from_dict` / `apply_overrides` / `_coerce`
   helpers are **identical** to `../classification/src/config.py` — copy them
   verbatim. Only the section dataclasses differ (mask paths instead of a label
   column; `monitor` becomes `mean_iou` / `dice`; no MixUp/CutMix by default).

2. **Checkpoints embed their own config** — `{epoch, model_state, config,
   metrics}`, rebuilt at inference via `from_dict`. Unchanged from classification.

3. **The encoder dictates preprocessing** — build the model first, derive input
   size / mean / std from the encoder (`timm.data.resolve_model_data_config`, or
   `smp.encoders.get_preprocessing_params`), and match it in the dataloader.

4. **One shared image+mask read for train AND inference.** *This is the main
   segmentation-specific change.* The mask must receive the **same geometric
   transforms** as the image (resize, flip, rotate, crop) but must **not** be
   normalized — it stays an integer label map. Pass both through one
   albumentations `Compose` call: `transform(image=img, mask=msk)`. Normalize +
   `ToTensorV2` affect only the image; the mask becomes a `long` tensor of class
   indices. Keep this in one place so train/inference geometry can't diverge.

5. **Every run is a timestamped, self-documenting directory** — `config.yaml`
   snapshot, `best_e<epoch>_<metric><score>.pt`, `history.json`, `train.log`.
   Unchanged. `monitor` is `mean_iou` (or `dice`) instead of `macro_f1`.

## What differs from classification

| Concern | Classification | Segmentation |
|---------|----------------|--------------|
| Target | one class index per image | an `H×W` integer mask per image |
| Head | `Dropout → Linear` on pooled features | decoder (U-Net/FPN/DeepLabV3+) on feature maps |
| Loss | `CrossEntropyLoss` over logits | pixel `CrossEntropyLoss` (optionally + Dice); `ignore_index` for void |
| Metric | accuracy, macro-F1 | per-class IoU + mean IoU, Dice |
| Batch aug | MixUp/CutMix (label-split at loss) | usually omitted — mixing masks is mask-aware and tricky; leave out unless justified |
| Preprocessing | resize/normalize image only | same geometric aug applied to **image and mask**; normalize image only |

## Suggested extra dependency

`segmentation-models-pytorch` gives U-Net / FPN / DeepLabV3+ decoders on top of
`timm` encoders, so contract #3 (encoder-driven preprocessing) still holds. See
`requirements.txt`.

## Getting started

1. Copy the config machinery from `../classification/src/config.py`; keep the
   section dataclasses defined here.
2. Implement `src/dataset.py` first (contract #4 — the image+mask transform is
   the part most different from classification).
3. Implement `src/model.py` (encoder + decoder), then `src/trainer.py` (reuse
   the classification loop; swap the criterion and metric).
4. Wire `train.py` / `test.py` exactly like their classification counterparts.
