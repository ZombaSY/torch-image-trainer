# Config module pattern (contract #1)

The whole reproducibility story rests on one idea: a single typed `Config`
object fully describes a run, round-trips through plain dicts, and validates
itself. Adapt the section names/fields to your task; keep the machinery.

## Sectioned dataclasses

Group knobs into section dataclasses so a run reads as `cfg.optim.lr`,
`cfg.data.root`, etc. Give every field a default and a comment on *why* it
exists — the config doubles as documentation.

```python
from __future__ import annotations
import dataclasses
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any
import yaml

SUPPORTED_BACKBONES: dict[str, str] = {"key": "hub/id", ...}
TRAIN_MODES = ("linear_probe", "full_finetune")

@dataclass
class OptimConfig:
    epochs: int = 30
    batch_size: int = 16
    optimizer: str = "adamw"
    lr: float = 1e-3            # head LR / base LR
    backbone_lr: float = 1e-5  # used only in full_finetune
    scheduler: str = "cosine"  # cosine | none
    warmup_epochs: int = 2
    min_lr: float = 1e-6
    class_weighted_loss: bool = True
    amp: bool = True
    grad_clip_norm: float = 1.0

# ... DataConfig, AugConfig, ModelConfig, RunConfig likewise ...

@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    aug: AugConfig = field(default_factory=AugConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    run: RunConfig = field(default_factory=RunConfig)
```

## Validation lives on the config

Fail fast on impossible combinations, in one place, before any expensive build.

```python
    def validate(self) -> "Config":
        if self.model.backbone not in SUPPORTED_BACKBONES:
            raise ValueError(f"Unknown backbone {self.model.backbone!r}")
        if self.run.mode not in TRAIN_MODES:
            raise ValueError(f"Unknown mode {self.run.mode!r}")
        if len(self.data.class_names) != self.data.num_classes:
            raise ValueError("class_names must match num_classes")
        if self.run.monitor not in ("accuracy", "macro_f1"):
            raise ValueError("run.monitor must be 'accuracy' or 'macro_f1'")
        return self
```

## Round-trip: to_dict / from_dict / save

`to_dict` is `asdict(self)`. `from_dict` rebuilds sections and validates — and
is exactly what checkpoint-loading uses to reconstruct a model (contract #2).

```python
_SECTION_TYPES = {"data": DataConfig, "aug": AugConfig, "model": ModelConfig,
                  "optim": OptimConfig, "run": RunConfig}

def _build_section(section_cls, values):
    values = values or {}
    known = {f.name for f in dataclasses.fields(section_cls)}
    unknown = set(values) - known
    if unknown:                      # a typo'd YAML key is a loud error
        raise ValueError(f"Unknown keys for {section_cls.__name__}: {sorted(unknown)}")
    return section_cls(**values)

def from_dict(raw: dict) -> Config:
    sections = {name: _build_section(cls, raw.get(name))
                for name, cls in _SECTION_TYPES.items()}
    return Config(**sections).validate()

def load_config(path) -> Config:
    with open(path) as fh:
        raw = yaml.safe_load(fh) or {}
    return from_dict(raw)
```

## Dotted CLI overrides with type coercion

Lets any leaf field be overridden as `section.field=value` without a bespoke
argparse flag per field. Coerce the string to the *existing* field's type so
`optim.lr=5e-4` becomes a float and `model.image_size=none` becomes `None`.

```python
def apply_overrides(cfg: Config, overrides: dict[str, str]) -> Config:
    data = cfg.to_dict()
    for dotted, raw_value in overrides.items():
        section, _, leaf = dotted.partition(".")
        if not leaf or section not in data or leaf not in data[section]:
            raise ValueError(f"Unknown override key: {dotted!r}")
        data[section][leaf] = _coerce(raw_value, data[section][leaf])
    sections = {name: cls(**data[name]) for name, cls in _SECTION_TYPES.items()}
    return Config(**sections).validate()

def _coerce(raw: str, like: Any) -> Any:
    if isinstance(like, bool): return raw.lower() in ("1", "true", "yes", "y")
    if isinstance(like, int):  return int(raw)
    if isinstance(like, float): return float(raw)
    if like is None:
        if raw.lower() in ("none", "null"): return None
        for caster in (int, float):
            try: return caster(raw)
            except ValueError: continue
    return raw
```

`bool` must be checked before `int` (in Python `bool` is a subclass of `int`).

## Entrypoint wiring

```python
cfg = load_config(args.config)
if overrides:
    cfg = apply_overrides(cfg, overrides)
run_dir = str(Path(cfg.run.output_dir) / run_timestamp())   # timestamped subdir
cfg = apply_overrides(cfg, {"run.output_dir": run_dir})
set_seed(cfg.run.seed, cfg.run.deterministic)
```
