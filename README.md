# torch-image-trainer

PyTorch training pipelines that fine-tune pretrained vision backbones, split by
task. Both trainers share one design (typed config, config-embedded checkpoints,
backbone-driven preprocessing, timestamped self-documenting runs) — captured in
the blueprint skill under `.claude/skills/pytorch-image-trainer-blueprint/`.

| Directory | Task | Status |
|-----------|------|--------|
| [`classification/`](classification) | Image classification (backbone + linear head; WD v3 tagger → UI-style classifier) | Implemented |
| [`segmentation/`](segmentation) | Semantic segmentation (encoder + decoder) | Placeholder — mirrors the classification design |

Each trainer is self-contained (its own `configs/`, `src/`, entrypoints, and
`requirements.txt`); run commands from inside its directory. See each
subdirectory's `README.md` for details.
