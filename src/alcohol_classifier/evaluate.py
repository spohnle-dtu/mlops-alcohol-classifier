from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import torch

from .data import DataConfig, make_dataloaders
from .model import BeverageModel
from .visualize import plot_confusion_matrix


@dataclass
class EvalConfig:
    checkpoint: str = "checkpoints/best.pt"
    data: DataConfig = field(default_factory=DataConfig)
    device: str = "auto"
    out_json: str = "reports/eval_metrics.json"
    out_cm_png: str = "reports/confusion_matrix.png"


def _get_device(device: str) -> torch.device:
    if device != "auto":
        return torch.device(device)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@torch.no_grad()
def evaluate(cfg: EvalConfig) -> None:
    device = _get_device(cfg.device)

    # DataConfig now uses processed_path (folder containing all_images.pt/all_labels.pt)
    _, val_loader, class_names_from_data = make_dataloaders(cfg.data)

    ckpt = torch.load(cfg.checkpoint, map_location=device)

    # Prefer class_names stored in checkpoint; fallback to data.py order
    class_names = ckpt.get("class_names", class_names_from_data)
    num_classes = len(class_names)

    # Build model with correct output size; no need to download pretrained weights during eval
    model = BeverageModel(num_classes=num_classes, pretrained=False).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    y_true_all = []
    y_pred_all = []

    for x, y in val_loader:
        x = x.to(device)
        y = y.to(device)

        logits = model(x)
        pred = torch.argmax(logits, dim=1)

        y_true_all.append(y.cpu())
        y_pred_all.append(pred.cpu())

    y_true = torch.cat(y_true_all)
    y_pred = torch.cat(y_pred_all)

    acc = float((y_true == y_pred).float().mean().item())

    cm = torch.zeros((num_classes, num_classes), dtype=torch.long)
    for t, p in zip(y_true.tolist(), y_pred.tolist()):
        cm[t, p] += 1

    out = {
        "accuracy": acc,
        "class_names": class_names,
        "confusion_matrix": cm.tolist(),
        "checkpoint": cfg.checkpoint,
    }

    Path(cfg.out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(cfg.out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    plot_confusion_matrix(cm, class_names, cfg.out_cm_png)

    print(f"✅ Accuracy: {acc:.3f}")
    print(f"✅ Saved eval metrics to: {cfg.out_json}")
    print(f"✅ Saved confusion matrix plot to: {cfg.out_cm_png}")


def _parse_args():
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="checkpoints/best.pt")
    p.add_argument("--processed-dir", default="data/processed",
                   help="Directory containing all_images.pt and all_labels.pt")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--val-fraction", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--device", default="auto")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    cfg = EvalConfig(
        checkpoint=args.checkpoint,
        data=DataConfig(
            processed_path=Path(args.processed_dir),
            batch_size=args.batch_size,
            val_fraction=args.val_fraction,
            seed=args.seed,
            num_workers=args.num_workers,
        ),
        device=args.device,
    )
    evaluate(cfg)
