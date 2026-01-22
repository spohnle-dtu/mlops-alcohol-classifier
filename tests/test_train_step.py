from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from omegaconf import OmegaConf
from torch.optim import Adam

from src.alcohol_classifier.data import make_dataloaders
from src.alcohol_classifier.model import BeverageModel
from src.alcohol_classifier.train import train_one_epoch

# #def _seed_tiny_processed_set(processed_dir: Path, n: int = 16, k: int = 3) -> None:
#     processed_dir.mkdir(parents=True, exist_ok=True)
#     images = torch.rand(n, 3, 224, 224)
#     labels = torch.randint(0, k, (n,), dtype=torch.long)
#     classes = ["beer", "whiskey", "wine"][:k]
#     torch.save(images, processed_dir / "images.pt")
#     torch.save(labels, processed_dir / "labels.pt")
#     torch.save(classes, processed_dir / "classes.pt")


def _seed_tiny_processed_set(processed_dir: Path, n: int = 16, k: int = 3) -> None:
    processed_dir.mkdir(parents=True, exist_ok=True)

    # Deterministic generation
    g = torch.Generator().manual_seed(0)

    images = torch.rand(n, 3, 224, 224, generator=g)

    # Ensure at least 2 samples per class for stratified split
    min_per_class = 2
    if n < k * min_per_class:
        raise ValueError("n too small to guarantee >=2 samples per class")

    base = torch.arange(k).repeat_interleave(min_per_class)  # length k*2
    remaining = n - base.numel()
    extra = torch.randint(0, k, (remaining,), generator=g, dtype=torch.long)

    labels = torch.cat([base, extra])
    labels = labels[torch.randperm(n, generator=g)]

    classes = ["beer", "whiskey", "wine"][:k]
    torch.save(images, processed_dir / "images.pt")
    torch.save(labels, processed_dir / "labels.pt")
    torch.save(classes, processed_dir / "classes.pt")


def _cfg(processed_dir: Path, *, batch_size=8, val_fraction=0.25, seed=0, num_workers=0):
    return OmegaConf.create(
        {
            "dataset": {
                "path_processed": str(processed_dir),
                "val_fraction": float(val_fraction),
                "seed": int(seed),
                "batch_size": int(batch_size),
                "num_workers": int(num_workers),
            }
        }
    )


def test_single_epoch_training_completes(tmp_path: Path):
    processed = tmp_path / "processed"
    _seed_tiny_processed_set(processed)

    cfg = _cfg(processed, batch_size=8, val_fraction=0.25, seed=0, num_workers=0)
    train_loader, _, class_names = make_dataloaders(cfg)

    model = BeverageModel(num_classes=len(class_names), pretrained=False)
    optimizer = Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    loss, acc = train_one_epoch(model, train_loader, optimizer, criterion, device=torch.device("cpu"))
    assert isinstance(loss, float) and loss >= 0.0
    assert isinstance(acc, float) and 0.0 <= acc <= 1.0
