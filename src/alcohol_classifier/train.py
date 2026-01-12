from __future__ import annotations

import json
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
from torch.optim import Adam

from .data import DataConfig, make_dataloaders
#from .model import BeverageCNN
from .model import BeverageModelResnet



@dataclass
class TrainConfig:
    data: DataConfig = field(default_factory=DataConfig)

    # Model
    dropout: float = 0.5

    # Training
    epochs: int = 10
    lr: float = 1e-3
    device: str = "auto"  # "cpu", "cuda", "mps", or "auto"

    # Outputs
    out_dir: str = "checkpoints"
    best_name: str = "best.pt"
    metrics_path: str = "reports/train_metrics.json"
    seed: int = 42


def _set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _get_device(device: str) -> torch.device:
    if device != "auto":
        return torch.device(device)

    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


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


def save_checkpoint(path: Path, model: nn.Module, class_names) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "class_names": class_names,
        },
        path,
    )


def train(cfg: TrainConfig) -> None:
    _set_seed(cfg.seed)
    device = _get_device(cfg.device)

    train_loader, val_loader, class_names = make_dataloaders(cfg.data)
    num_classes = len(class_names)

    #model = BeverageCNN(num_classes=num_classes, dropout=cfg.dropout).to(device)
    model = BeverageModelResnet(num_classes=num_classes, dropout=cfg.dropout, pretrained=True).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=cfg.lr)

    out_dir = Path(cfg.out_dir)
    best_path = out_dir / cfg.best_name
    reports_path = Path(cfg.metrics_path)
    reports_path.parent.mkdir(parents=True, exist_ok=True)

    history = {"config": asdict(cfg), "class_names": class_names, "epochs": []}
    history["config"]["data"]["processed_path"] = str(history["config"]["data"]["processed_path"])
    best_val_acc = -1.0

    for epoch in range(1, cfg.epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        va_loss, va_acc = validate(model, val_loader, criterion, device)

        history["epochs"].append(
            {"epoch": epoch, "train_loss": tr_loss, "train_acc": tr_acc, "val_loss": va_loss, "val_acc": va_acc}
        )

        print(
            f"[{epoch:02d}/{cfg.epochs}] "
            f"train loss={tr_loss:.4f} acc={tr_acc:.3f} | "
            f"val loss={va_loss:.4f} acc={va_acc:.3f}"
        )

        if va_acc > best_val_acc:
            best_val_acc = va_acc
            save_checkpoint(best_path, model, class_names)

    with open(reports_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, default=str)


    print(f"✅ Saved best checkpoint to: {best_path}")
    print(f"✅ Saved training metrics to: {reports_path}")


def _parse_args():
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--processed-dir", default="data/processed",
                   help="Directory containing all_images.pt and all_labels.pt")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--val-fraction", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--device", default="auto")
    return p.parse_args()



if __name__ == "__main__":
    args = _parse_args()
    cfg = TrainConfig(
        data=DataConfig(
            processed_path=Path(args.processed_dir),
            batch_size=args.batch_size,
            val_fraction=args.val_fraction,
            seed=args.seed,
            num_workers=args.num_workers,
        ),
        epochs=args.epochs,
        lr=args.lr,
        device=args.device,
    )
    train(cfg)

