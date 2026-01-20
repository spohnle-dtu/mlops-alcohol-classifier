import json
from pathlib import Path
from typing import Tuple
import torch
import torch.nn as nn
from torch.optim import Adam
import hydra
from omegaconf import DictConfig, OmegaConf
import wandb

from src.alcohol_classifier.data import make_dataloaders
from src.alcohol_classifier.model import BeverageModel
from src.alcohol_classifier.utils import _set_seed, _get_device

def train_one_epoch(model, loader, optimizer, criterion, device) -> Tuple[float, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item()) * x.size(0)
        preds = torch.argmax(logits, dim=1)
        correct += int((preds == y).sum().item())
        total += int(y.numel())

    return total_loss / max(total, 1), correct / max(total, 1)

@torch.no_grad()
def validate(model, loader, criterion, device) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        logits = model(x)
        loss = criterion(logits, y)

        total_loss += float(loss.item()) * x.size(0)
        preds = torch.argmax(logits, dim=1)
        correct += int((preds == y).sum().item())
        total += int(y.numel())

    return total_loss / max(total, 1), correct / max(total, 1)

@hydra.main(config_path="../../configs", config_name="run", version_base="1.3")
def train(cfg: DictConfig) -> None:
    _set_seed(cfg.dataset.seed)
    device = _get_device(cfg.device)

    wandb.init(
        project=cfg.logger.project,
        entity=cfg.logger.entity,
        group=cfg.logger.group,
        name=cfg.logger.name,
        config=OmegaConf.to_container(cfg, resolve=True)
    )
    wandb.define_metric("epoch")
    wandb.define_metric("*", step_metric="epoch")

    train_loader, val_loader, class_names = make_dataloaders(cfg)
    
    model = BeverageModel(
        num_classes=len(class_names),
        dropout=cfg.model.dropout,
        pretrained=cfg.model.pretrained,
        freeze_backbone=cfg.model.freeze_backbone,
    ).to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Model: pretrained={cfg.model.pretrained}, freeze_backbone={cfg.model.freeze_backbone}")
    print(f"Params: trainable={trainable:,} / total={total:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam((p for p in model.parameters() if p.requires_grad), lr=cfg.model.lr)

    best_val_acc = -1.0

    for epoch in range(1, cfg.model.epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        va_loss, va_acc = validate(model, val_loader, criterion, device)

        wandb.log({
            "epoch": epoch,
            "train/loss": tr_loss, "train/acc": tr_acc,
            "val/loss": va_loss, "val/acc": va_acc
        })

        print(
            f"[{epoch:02d}/{cfg.model.epochs}] "
            f"train loss={tr_loss:.4f} acc={tr_acc:.3f} | "
            f"val loss={va_loss:.4f} acc={va_acc:.3f}"
        )

        if va_acc > best_val_acc:
            best_val_acc = va_acc
            Path(cfg.path_model).parent.mkdir(parents=True, exist_ok=True)
            torch.save({"state_dict": model.state_dict(), "class_names": class_names}, cfg.path_model)

    wandb.finish()
    
    print(f"✅ Saved best checkpoint to: {cfg.path_model}")
    print(f"✅ Logged training metrics as: {cfg.logger.name} in {cfg.logger.group}")

if __name__ == "__main__":
    train()