from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Adam

from src.alcohol_classifier.data import DataConfig, make_dataloaders
from src.alcohol_classifier.model import BeverageModel
from src.alcohol_classifier.train import train_one_epoch


def _seed_tiny_processed_set(processed_dir: Path, n: int = 16, k: int = 3) -> None:
    """Create a tiny processed dataset for a fast training-step test."""
    processed_dir.mkdir(parents=True, exist_ok=True)
    images = torch.rand(n, 3, 224, 224)
    labels = torch.randint(0, k, (n,))
    torch.save(images, processed_dir / "all_images.pt")
    torch.save(labels, processed_dir / "all_labels.pt")


def test_single_epoch_training_completes(tmp_path: Path):
    processed = tmp_path / "processed"
    _seed_tiny_processed_set(processed)

    cfg = DataConfig(
        processed_path=processed,
        batch_size=8,
        val_fraction=0.25,
        seed=0,
        num_workers=0,
    )

    train_loader, _, class_names = make_dataloaders(cfg)

    model = BeverageModel(num_classes=len(class_names), pretrained=False)
    optimizer = Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    loss, acc = train_one_epoch(model, train_loader, optimizer, criterion, device=torch.device("cpu"))

    assert isinstance(loss, float) and loss >= 0.0
    assert isinstance(acc, float) and 0.0 <= acc <= 1.0

