# Trainer loop + batch augmentation (contracts #4/#5 + the loop)

## Model build: freeze logic and per-group LRs

Build the backbone through `timm`, strip its native head to expose pooled
features, and add a small `Dropout → Linear` head. Freeze the backbone for
linear probing; train it at a smaller LR for full fine-tuning.

```python
def build_model(cfg) -> tuple[nn.Module, dict]:
    backbone = timm.create_model(f"hf-hub:{cfg.hf_hub_id}", pretrained=cfg.model.pretrained)
    data_config = timm.data.resolve_model_data_config(backbone)  # input size, mean, std
    num_features = backbone.num_features
    backbone.reset_classifier(0)                                 # -> pooled features
    model = StyleClassifier(backbone, num_features, cfg.data.num_classes, cfg.model.head_dropout)
    model.set_backbone_trainable(cfg.run.mode == "full_finetune")
    return model, data_config

def set_backbone_trainable(self, trainable: bool) -> None:
    for p in self.backbone.parameters():
        p.requires_grad = trainable
    self.backbone.train(trainable)   # frozen backbone: also stop norm running-stat updates

def build_param_groups(model, cfg) -> list[dict]:
    groups = [{"params": list(model.head_parameters()), "lr": cfg.optim.lr}]
    if cfg.run.mode == "full_finetune":
        groups.append({"params": list(model.backbone_parameters()), "lr": cfg.optim.backbone_lr})
    return groups
```

## Warmup + cosine schedule as a LambdaLR fraction

Express the schedule as a multiplier on the base LR so `scheduler == "none"` is
a clean identity and `min_lr` becomes a floor fraction.

```python
def _lr_lambda(cfg):
    total, warmup = cfg.optim.epochs, cfg.optim.warmup_epochs
    floor = cfg.optim.min_lr / max(cfg.optim.lr, 1e-12)
    def fn(epoch: int) -> float:
        if cfg.optim.scheduler == "none": return 1.0
        if warmup > 0 and epoch < warmup: return (epoch + 1) / warmup
        progress = (epoch - warmup) / max(1, total - warmup)
        return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))
    return fn
```

## Train step: AMP + grad clip + batch mix

```python
self.optimizer.zero_grad(set_to_none=True)
mixed = self.mixaug(images, labels)                    # may be a no-op
with torch.amp.autocast("cuda", enabled=self.use_amp):
    logits = self.model(mixed.images)
    loss = mix_loss(self.criterion, logits, mixed)
self.scaler.scale(loss).backward()
if self.cfg.optim.grad_clip_norm > 0:
    self.scaler.unscale_(self.optimizer)
    nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.optim.grad_clip_norm)
self.scaler.step(self.optimizer); self.scaler.update()
```

In `linear_probe`, put the *model* in `.train()` but force `backbone.eval()`
each epoch so frozen norm layers keep stable statistics.

## Metric-based best checkpoint + early stop

```python
if monitored > self.best_metric:
    self.best_metric, self.best_epoch, self.epochs_since_best = monitored, epoch, 0
    if self.best_ckpt_path and self.best_ckpt_path.exists():
        self.best_ckpt_path.unlink()                   # drop old best (hundreds of MB)
    self.best_ckpt_path = self._save_checkpoint(self._ckpt_name("best", epoch, val_metrics),
                                                epoch, val_metrics)
else:
    self.epochs_since_best += 1
if patience > 0 and self.epochs_since_best >= patience:
    break                                              # early stop; epochs is a hard cap

def _save_checkpoint(self, name, epoch, metrics) -> Path:
    torch.save({"epoch": epoch, "model_state": self.model.state_dict(),
                "config": self.cfg.to_dict(), "metrics": metrics}, self.output_dir / name)
```

Filename embeds the score (`best_e7_macro_f10.6445.pt`) so runs compare at a
glance. The checkpoint carries its own config — that is contract #2.

## Batch augmentation contract (mixaug.py)

Operate on batched on-device tensors; return a dataclass; blend at the loss.

```python
@dataclass
class MixResult:
    images: torch.Tensor
    target_a: torch.Tensor
    target_b: torch.Tensor
    lam: float                 # weight of target_a; 1.0 == no mix applied

def mix_loss(criterion, logits, r: MixResult) -> torch.Tensor:
    if r.lam >= 1.0:
        return criterion(logits, r.target_a)            # exact fallback, keeps weights/smoothing
    return r.lam * criterion(logits, r.target_a) + (1 - r.lam) * criterion(logits, r.target_b)
```

For CutMix, after pasting the patch recompute `lam` from the **true** patch area
(`1 - area/(H*W)`) so the label weight matches the pixels replaced, then clamp
to `[min_ratio, 1 - min_ratio]` so neither source ever dominates. Place the box
fully in-frame (pick top-left in `[0, dim - cut + 1]`) rather than centering and
clipping — clipping would shrink the patch near edges and break the ratio floor.

## Metrics without sklearn (utils.py)

```python
def classification_metrics(targets, preds, num_classes) -> dict:
    accuracy = float((targets == preds).mean()) if len(targets) else 0.0
    per_class_f1 = []
    for c in range(num_classes):
        tp = int(((preds == c) & (targets == c)).sum())
        fp = int(((preds == c) & (targets != c)).sum())
        fn = int(((preds != c) & (targets == c)).sum())
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        per_class_f1.append(2 * p * r / (p + r) if (p + r) else 0.0)
    return {"accuracy": accuracy, "macro_f1": float(np.mean(per_class_f1)),
            "per_class_f1": [round(x, 4) for x in per_class_f1]}
```

Return per-class F1 so a macro metric never hides which class is collapsing.
